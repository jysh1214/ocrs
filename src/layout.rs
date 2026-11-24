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
