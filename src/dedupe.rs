//! Tells whether a frame's text repeats the frame before it.

/// Remembers the text of the previous frame.
pub struct Dedupe {
    min_similarity: f64,
    previous: Option<String>,
}

impl Dedupe {
    /// Texts whose [`similarity`] reaches `min_similarity` count as the same content; at 1 (or above) only
    /// identical texts do.
    pub fn new(min_similarity: f64) -> Self {
        Self {
            min_similarity,
            previous: None,
        }
    }

    /// Whether `text` differs from the text passed in the call before. Comparing against the previous frame
    /// rather than the last one that was new keeps a page whose reading flickers between two variants from
    /// being reported again and again.
    pub fn is_new(&mut self, text: &str) -> bool {
        let repeated = self
            .previous
            .as_deref()
            .is_some_and(|previous| similarity_at_least(previous, text, self.min_similarity).is_some());
        self.previous = Some(text.to_owned());
        !repeated
    }
}

/// 1 for identical texts, 0 for texts with nothing in common: one minus the edit distance relative to the
/// longer text.
pub fn similarity(a: &str, b: &str) -> f64 {
    let (a, b) = (a.chars().collect::<Vec<_>>(), b.chars().collect::<Vec<_>>());
    let longer = a.len().max(b.len());
    if longer == 0 {
        return 1.0;
    }
    1.0 - edit_distance(&a, &b) as f64 / longer as f64
}

/// The [`similarity`] of two texts if it reaches `min_similarity`, and `None` if it doesn't. That is cheaper
/// to find out than the similarity itself, the more so the higher the bar is.
pub fn similarity_at_least(a: &str, b: &str, min_similarity: f64) -> Option<f64> {
    if a == b {
        return Some(1.0);
    }
    if min_similarity >= 1.0 {
        return None;
    }
    let (a, b) = (a.chars().collect::<Vec<_>>(), b.chars().collect::<Vec<_>>());
    let longer = a.len().max(b.len());
    // rounded up, the exact comparison follows
    let most_edits = ((1.0 - min_similarity) * longer as f64).ceil() as usize;
    let distance = edit_distance_up_to(&a, &b, most_edits)?;
    let similarity = 1.0 - distance as f64 / longer as f64;
    (similarity >= min_similarity).then_some(similarity)
}

/// The Levenshtein distance if it is no more than `limit`. Only the part of the table within `limit` of its
/// diagonal can belong to so short a way through it, so only that part is filled in, and the search stops
/// at the first row that has nothing within the limit.
fn edit_distance_up_to(a: &[char], b: &[char], limit: usize) -> Option<usize> {
    if a.len().abs_diff(b.len()) > limit {
        return None;
    }
    // stands for the cells that are not filled in; small enough to add 1 to
    const FAR: usize = usize::MAX / 2;
    let mut previous = vec![FAR; b.len() + 1];
    let mut current = vec![FAR; b.len() + 1];
    for (j, cell) in previous.iter_mut().enumerate().take(limit + 1) {
        *cell = j;
    }
    for i in 1..=a.len() {
        let (first, last) = (i.saturating_sub(limit), (i + limit).min(b.len()));
        // the neighbours of this row's part that the next cells look at must not be left over from before
        if first > 0 {
            current[first - 1] = FAR;
        }
        if last < b.len() {
            current[last + 1] = FAR;
        }
        let mut nearest = FAR;
        for j in first..=last {
            current[j] = if j == 0 {
                i
            } else {
                let substitution = previous[j - 1] + usize::from(a[i - 1] != b[j - 1]);
                substitution.min(previous[j] + 1).min(current[j - 1] + 1)
            };
            nearest = nearest.min(current[j]);
        }
        if nearest > limit {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    let distance = previous[b.len()];
    (distance <= limit).then_some(distance)
}

/// Levenshtein distance, one row of the table at a time.
fn edit_distance(a: &[char], b: &[char]) -> usize {
    let mut row = (0..=b.len()).collect::<Vec<_>>();
    for (i, &ca) in a.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let substitution = diagonal + usize::from(ca != cb);
            diagonal = row[j + 1];
            row[j + 1] = substitution.min(row[j] + 1).min(row[j + 1] + 1);
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_consecutive_repeats_are_dropped() {
        let mut dedupe = Dedupe::new(1.0);
        let seen = ["a", "a", "b", "b", "b", "a", "c"].map(|text| dedupe.is_new(text));
        assert_eq!(seen, [true, false, true, false, false, true, true]);
    }

    #[test]
    fn exact_mode_keeps_near_duplicates() {
        let mut dedupe = Dedupe::new(1.0);
        assert!(dedupe.is_new("total 1234"));
        assert!(dedupe.is_new("total 1284"));
    }

    #[test]
    fn similar_texts_count_as_repeats_when_allowed() {
        let mut dedupe = Dedupe::new(0.9);
        assert!(dedupe.is_new("the quick brown fox"));
        assert!(!dedupe.is_new("the quick brovn fox")); // 1 of 19 characters differs
        assert!(dedupe.is_new("an entirely different page"));
    }

    #[test]
    fn flicker_between_two_readings_is_reported_once() {
        let mut dedupe = Dedupe::new(0.9);
        // each reading is within 2 of 26 characters of the one before
        let readings = [
            "the same page read with 0O",
            "the same page read with O0",
            "the same page read with 00",
        ];
        let seen = [readings[0], readings[1], readings[2], readings[0]].map(|text| dedupe.is_new(text));
        assert_eq!(seen, [true, false, false, false]);
    }

    #[test]
    fn limited_edit_distance_agrees_with_the_full_one() {
        // texts over a small alphabet, so that all kinds of edits and coincidences come up
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut random = |below: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % below as u64) as usize
        };
        for _ in 0..3000 {
            let length = random(14);
            let a = (0..length)
                .map(|_| ['a', 'b', 'c'][random(3)])
                .collect::<Vec<_>>();
            // either a few edits of `a` or a text of its own
            let mut b = a.clone();
            if random(4) == 0 {
                let length = random(14);
                b = (0..length).map(|_| ['a', 'b', 'c'][random(3)]).collect();
            } else {
                for _ in 0..random(5) {
                    let at = random(b.len() + 1);
                    match random(3) {
                        0 => b.insert(at, ['a', 'b', 'c'][random(3)]),
                        1 if at < b.len() => drop(b.remove(at)),
                        2 if at < b.len() => b[at] = ['a', 'b', 'c'][random(3)],
                        _ => {}
                    }
                }
            }
            let full = edit_distance(&a, &b);
            for limit in 0..16 {
                let expected = (full <= limit).then_some(full);
                assert_eq!(
                    edit_distance_up_to(&a, &b, limit),
                    expected,
                    "{a:?} {b:?} limit {limit}"
                );
            }
        }
    }

    #[test]
    fn similarity_is_only_worked_out_where_it_reaches_the_bar() {
        assert_eq!(similarity_at_least("abcd", "abcd", 1.0), Some(1.0));
        assert_eq!(similarity_at_least("abcd", "abxd", 1.0), None);
        assert_eq!(similarity_at_least("abcd", "abxd", 0.75), Some(0.75));
        assert_eq!(similarity_at_least("abcd", "abxd", 0.76), None);
        assert_eq!(similarity_at_least("abcd", "wxyz", 0.0), Some(0.0));
        assert_eq!(similarity_at_least("", "", 0.9), Some(1.0));
        assert_eq!(similarity_at_least("", "a", 0.9), None);
        // the same answers as the plain similarity, for a bar that falls between two distances
        let page = "the quick brown fox jumps over the lazy dog";
        for (other, alike) in [
            ("the quick brown fox jumps over the lazy dog", true),
            ("the quick brovn fox jumps ovcr the lazy dog", true),
            ("the quick brovn fox jumps ovcr the lazv bog", true),
            ("the qvick brovn fox jumps ovcr the lazv bog", false),
            ("an entirely different page", false),
        ] {
            let found = similarity_at_least(page, other, 0.9);
            assert_eq!(found.is_some(), alike, "{other}");
            assert_eq!(found.is_some(), similarity(page, other) >= 0.9, "{other}");
            if let Some(found) = found {
                assert_eq!(found, similarity(page, other));
            }
        }
    }

    #[test]
    fn similarity_is_relative_to_the_longer_text() {
        assert_eq!(similarity("", ""), 1.0);
        assert_eq!(similarity("abcd", "abcd"), 1.0);
        assert_eq!(similarity("abcd", ""), 0.0);
        assert_eq!(similarity("abcd", "abxd"), 0.75);
        assert_eq!(similarity("kitten", "sitting"), 1.0 - 3.0 / 7.0);
    }
}
