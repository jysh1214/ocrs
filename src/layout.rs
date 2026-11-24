//! Puts recognised pieces of text into reading order.

use opencv::core::Point2f;

/// A piece of text and where it was found on the page.
#[derive(Clone, Debug)]
pub struct TextBox {
    /// Top-left, top-right, bottom-right, bottom-left
    pub corners: [Point2f; 4],
    pub text: String,
    /// Mean probability of the recognised characters
    pub confidence: f32,
}

impl TextBox {
    fn center_y(&self) -> f32 {
        self.corners.iter().map(|p| p.y).sum::<f32>() / 4.0
    }

    fn height(&self) -> f32 {
        let [tl, tr, br, bl] = self.corners;
        ((bl.y - tl.y) + (br.y - tr.y)) / 2.0
    }

    fn left(&self) -> f32 {
        self.corners[0].x.min(self.corners[3].x)
    }

    fn width(&self) -> f32 {
        let [tl, tr, br, bl] = self.corners;
        ((tr.x - tl.x).hypot(tr.y - tl.y) + (br.x - bl.x).hypot(br.y - bl.y)) / 2.0
    }
}

/// Pieces of text shorter than this say little about how wide their characters are.
const MIN_MEASURED_CHARACTERS: usize = 8;

/// How wide a character of the page is in pixels: the width of its pieces of text over their characters,
/// for the piece in the middle of them all. For text in a typewriter font, such as a terminal shows, that
/// is how many pixels a column of it gets, which decides how well it can be read. `None` for a page
/// without a piece of text that is long enough to tell.
pub fn character_width(boxes: &[TextBox]) -> Option<f32> {
    let mut widths = boxes
        .iter()
        .filter_map(|text_box| {
            let characters = text_box.text.chars().count();
            (characters >= MIN_MEASURED_CHARACTERS).then(|| text_box.width() / characters as f32)
        })
        .collect::<Vec<_>>();
    widths.sort_by(f32::total_cmp);
    widths.get(widths.len() / 2).copied()
}

/// The page as text: rows top to bottom, the boxes of a row left to right with a space between them.
pub fn page_text(boxes: &[TextBox]) -> String {
    let mut by_height = boxes.iter().collect::<Vec<_>>();
    by_height.sort_by(|a, b| a.center_y().total_cmp(&b.center_y()));

    let mut rows: Vec<Vec<&TextBox>> = Vec::new();
    for text_box in by_height {
        match rows.last_mut() {
            // measured against the row's first box, so a staircase of boxes doesn't chain into one row
            Some(row) if same_row(row[0], text_box) => row.push(text_box),
            _ => rows.push(vec![text_box]),
        }
    }
    rows.iter_mut()
        .map(|row| {
            row.sort_by(|a, b| a.left().total_cmp(&b.left()));
            row.iter().map(|b| b.text.as_str()).collect::<Vec<_>>().join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Two boxes sit on one row when their centres are less than half the smaller box's height apart.
fn same_row(a: &TextBox, b: &TextBox) -> bool {
    (a.center_y() - b.center_y()).abs() < 0.5 * a.height().min(b.height())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A piece of text in a box of this width, 30 pixels high.
    fn piece(text: &str, width: f32) -> TextBox {
        TextBox {
            corners: [(0.0, 0.0), (width, 0.0), (width, 30.0), (0.0, 30.0)].map(|(x, y)| Point2f::new(x, y)),
            text: text.to_owned(),
            confidence: 1.0,
        }
    }

    #[test]
    fn character_width_is_that_of_the_piece_in_the_middle() {
        // 12 pixels a character, but for a piece whose columns of figures were read as one space each
        let boxes = [
            piece("a line of thirty characters...", 360.0),
            piece("Dry thermometer deg C 3 hours", 500.0),
            piece("another twenty chars", 240.0),
        ];
        assert_eq!(character_width(&boxes), Some(12.0));
        // too short to measure
        assert_eq!(
            character_width(&[piece("3.", 40.0), piece("Returns", 90.0)]),
            None
        );
        assert_eq!(character_width(&[]), None);
    }

    #[test]
    fn character_width_follows_a_slanted_piece() {
        let mut slanted = piece("ten chars.", 0.0);
        slanted.corners =
            [(0.0, 0.0), (80.0, 60.0), (62.0, 84.0), (-18.0, 24.0)].map(|(x, y)| Point2f::new(x, y));
        assert_eq!(character_width(&[slanted]), Some(10.0));
    }

    fn text_box(x: f32, y: f32, width: f32, height: f32, text: &str) -> TextBox {
        let corners = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)];
        TextBox {
            corners: corners.map(|(x, y)| Point2f::new(x, y)),
            text: text.to_owned(),
            confidence: 1.0,
        }
    }

    #[test]
    fn empty_page_is_empty_text() {
        assert_eq!(page_text(&[]), "");
    }

    #[test]
    fn rows_run_top_to_bottom_and_left_to_right() {
        let boxes = [
            text_box(300.0, 52.0, 100.0, 20.0, "right"),
            text_box(10.0, 100.0, 200.0, 20.0, "second line"),
            text_box(10.0, 50.0, 100.0, 20.0, "left"),
            text_box(150.0, 48.0, 100.0, 20.0, "middle"),
        ];
        assert_eq!(page_text(&boxes), "left middle right\nsecond line");
    }

    #[test]
    fn tightly_spaced_lines_stay_separate() {
        // line pitch 22 with boxes 20 high: neighbouring lines overlap by nothing but are close
        let boxes = [
            text_box(10.0, 32.0, 100.0, 20.0, "b"),
            text_box(10.0, 10.0, 100.0, 20.0, "a"),
        ];
        assert_eq!(page_text(&boxes), "a\nb");
    }

    #[test]
    fn small_box_joins_the_row_it_is_centred_on() {
        let boxes = [
            text_box(10.0, 10.0, 100.0, 40.0, "BIG"),
            text_box(120.0, 22.0, 30.0, 14.0, "small"),
        ];
        assert_eq!(page_text(&boxes), "BIG small");
    }
}
