//! Tells which reading the recent frames agree on.
//!
//! A single frame of a camera is not to be trusted: a frame taken while the camera moves is blurred, and
//! one taken while the page changes shows two pages at once. What such frames read differs from one to the
//! next, while the readings of a page that is in view are much the same. So a reading only counts once
//! most of the recent frames have it.

use std::collections::VecDeque;

use crate::dedupe::similarity_at_least;

pub struct Consensus {
    /// How many frames have to agree
    needed: usize,
    min_similarity: f64,
    /// What the last `2 * needed - 1` frames read, oldest first
    recent: VecDeque<String>,
    /// For each reading in `recent`, how similar every later one is to it, if similar enough to agree
    alike: VecDeque<Vec<Option<f64>>>,
}

impl Consensus {
    /// A reading counts once `needed` of the last `2 * needed - 1` frames agree on it, which is most of
    /// them; with 1 every frame is taken at its word. Two readings agree if their similarity (see
    /// [`crate::dedupe::similarity`]) is at least `min_similarity`.
    pub fn new(needed: usize, min_similarity: f64) -> Self {
        let needed = needed.max(1);
        Self {
            needed,
            min_similarity,
            recent: VecDeque::with_capacity(2 * needed),
            alike: VecDeque::with_capacity(2 * needed),
        }
    }

    /// Takes what the next frame reads and returns the reading that the recent frames agree on now, if
    /// there is one. Frames that agree rarely read exactly the same; of their readings this is the one
    /// that the most of them agree with and that is the most similar to those, which tends to be the one
    /// with the fewest misread characters.
    pub fn push(&mut self, text: String) -> Option<&str> {
        if self.recent.len() == 2 * self.needed - 1 {
            self.recent.pop_front();
            self.alike.pop_front();
        }
        for (earlier, alike) in self.recent.iter().zip(&mut self.alike) {
            alike.push(similarity_at_least(earlier, &text, self.min_similarity));
        }
        self.recent.push_back(text);
        self.alike.push_back(Vec::new());
        let (reading, agreeing) = self.leading(|_| true)?;
        (agreeing >= self.needed).then_some(reading)
    }

    /// The reading with text that the most of the recent frames agree with, whether or not those are enough
    /// for it to count. `None` if no frame read any text.
    pub fn likeliest(&self) -> Option<&str> {
        Some(self.leading(|reading| !reading.is_empty())?.0)
    }

    /// Forgets the frames so far, for when what comes next is known to be something else.
    pub fn clear(&mut self) {
        self.recent.clear();
        self.alike.clear();
    }

    /// Of the recent readings that are `eligible`, the one with the most support, and how many of the
    /// recent readings agree with it, itself included.
    fn leading(&self, eligible: impl Fn(&str) -> bool) -> Option<(&str, usize)> {
        // for every reading: how many agree with it, and how similar those are in sum
        let mut support = vec![(1, 0.0); self.recent.len()];
        for (index, alike) in self.alike.iter().enumerate() {
            for (later, similarity) in alike.iter().enumerate() {
                if let Some(similarity) = similarity {
                    for reading in [index, index + 1 + later] {
                        support[reading].0 += 1;
                        support[reading].1 += similarity;
                    }
                }
            }
        }
        // the last of equals: the reading of the newest frame where nothing speaks for another
        let (best, &(agreeing, _)) = support
            .iter()
            .enumerate()
            .filter(|(reading, _)| eligible(&self.recent[*reading]))
            .max_by(|(_, a), (_, b)| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)))?;
        Some((self.recent[best].as_str(), agreeing))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a consensus of `needed` frames makes of `readings`, one after the other.
    fn settle<'a>(needed: usize, min_similarity: f64, readings: &[&'a str]) -> Vec<Option<&'a str>> {
        let mut consensus = Consensus::new(needed, min_similarity);
        readings
            .iter()
            .map(|reading| {
                let settled = consensus.push((*reading).to_owned())?.to_owned();
                // the same text as one of the readings
                readings.iter().copied().find(|reading| *reading == settled)
            })
            .collect()
    }

    #[test]
    fn one_frame_is_enough_when_asked_for() {
        assert_eq!(
            settle(1, 1.0, &["a", "b", "b", ""]),
            [Some("a"), Some("b"), Some("b"), Some("")]
        );
    }

    #[test]
    fn reading_counts_from_the_third_frame_that_has_it() {
        assert_eq!(
            settle(3, 1.0, &["a", "a", "a", "a"]),
            [None, None, Some("a"), Some("a")]
        );
    }

    #[test]
    fn frame_between_two_pages_is_passed_over() {
        let readings = ["one", "one", "one", "o?e t?o", "two", "two", "two", "two"];
        assert_eq!(
            settle(3, 1.0, &readings),
            [
                None,
                None,
                Some("one"),
                // still what three of the last five frames read
                Some("one"),
                Some("one"),
                // two frames for either page
                None,
                Some("two"),
                Some("two")
            ]
        );
    }

    #[test]
    fn agreeing_frames_need_not_follow_each_other() {
        let readings = ["page", "blurred", "page", "shaken", "page", "moved", "page"];
        assert_eq!(
            settle(3, 1.0, &readings),
            [
                None,
                None,
                None,
                None,
                Some("page"),
                // two of the last five frames
                None,
                Some("page")
            ]
        );
    }

    #[test]
    fn readings_that_never_repeat_never_count() {
        let readings = [
            "blur one",
            "smear two",
            "shake three",
            "noise four",
            "haze five",
            "fog six",
        ];
        assert_eq!(settle(3, 0.9, &readings), [None; 6]);
        assert_eq!(settle(2, 0.9, &readings), [None; 6]);
    }

    #[test]
    fn empty_readings_agree_with_each_other() {
        assert_eq!(settle(2, 0.9, &["text", "", ""]), [None, None, Some("")]);
    }

    #[test]
    fn of_similar_readings_the_one_in_the_middle_is_taken() {
        // each misreads another character, or none
        let right = "Total due: 1,284.50 USD";
        let (first, second) = ("Total due: 1,2B4.50 USD", "Total dve: 1,284.50 USD");
        assert_eq!(settle(3, 0.9, &[first, second, right]), [None, None, Some(right)]);
        assert_eq!(settle(3, 0.9, &[right, first, second]), [None, None, Some(right)]);
        assert_eq!(settle(3, 0.9, &[first, right, second]), [None, None, Some(right)]);
        // and exact agreement is not to be had from them
        assert_eq!(settle(3, 1.0, &[first, second, right]), [None; 3]);
    }

    #[test]
    fn likeliest_reading_needs_no_majority() {
        let mut consensus = Consensus::new(3, 0.9);
        assert_eq!(consensus.likeliest(), None);
        for reading in [
            "Total due: 1,284.50 USD",
            "smeared beyond repair",
            "Total due: 1,284.50 USO",
        ] {
            assert_eq!(consensus.push(reading.to_owned()), None);
        }
        // two of three agree, and of those the later one stands
        assert_eq!(consensus.likeliest(), Some("Total due: 1,284.50 USO"));
        consensus.clear();
        assert_eq!(consensus.likeliest(), None);
        assert_eq!(consensus.push("next page".to_owned()), None);

        // frames that read nothing are no reading of the page, however many they are
        let mut consensus = Consensus::new(3, 0.9);
        for reading in ["", "", "one frame read this", ""] {
            consensus.push(reading.to_owned());
        }
        assert_eq!(consensus.likeliest(), Some("one frame read this"));
        consensus.clear();
        consensus.push(String::new());
        assert_eq!(consensus.likeliest(), None);
    }

    #[test]
    fn unlike_reading_among_similar_ones_has_no_say() {
        let page = "Total due: 1,284.50 USD";
        let readings = [
            page,
            "Tatol bue: l,2B4.5O U5D",
            "Total due: 1,284.50 USO",
            "Total due: 1,284.50 USD",
        ];
        assert_eq!(settle(3, 0.9, &readings), [None, None, None, Some(page)]);
    }
}
