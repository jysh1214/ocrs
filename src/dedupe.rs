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
        let repeated = self.previous.as_deref().is_some_and(|previous| {
            previous == text
                || (self.min_similarity < 1.0 && similarity(previous, text) >= self.min_similarity)
        });
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
    fn similarity_is_relative_to_the_longer_text() {
        assert_eq!(similarity("", ""), 1.0);
        assert_eq!(similarity("abcd", "abcd"), 1.0);
        assert_eq!(similarity("abcd", ""), 0.0);
        assert_eq!(similarity("abcd", "abxd"), 0.75);
        assert_eq!(similarity("kitten", "sitting"), 1.0 - 3.0 / 7.0);
    }
}
