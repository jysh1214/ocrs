//! The chart as a web page, for a browser instead of a terminal.
//!
//! On a terminal the markers are made of character cells, so they grow and shrink with the text. On a page
//! the two have nothing to do with each other: the markers take a fixed share of the chart, and the text is
//! as large as it takes to get the number of characters asked for into a line.
//!
//! Everything is laid out here and only drawn by the page. Lengths are in hundredths of the chart's width
//! on both axes, which keeps squares square at whatever size a browser shows the chart.

use std::fmt::Write as _;

use anyhow::{Result, ensure};
use ocrs::markers::Pattern;

use crate::chart::{Rgb, Settings};

/// The width of the chart in the units of the layout.
pub const WIDTH: f64 = 100.0;

/// The notes below the text area are two lines, which are not worth showing smaller than this each.
const MIN_NOTE_LINE: f64 = 1.0;

/// What is up to the page alone. The rest is in [`Settings`], of which a page has no use for the `module`.
#[derive(Clone, Debug, PartialEq)]
pub struct Format {
    /// Width and height of the chart, of which only the ratio counts
    pub aspect: (f64, f64),
    /// Characters in a line of text, which is what sets the size of the text
    pub columns: usize,
    /// Height of a line of text in widths of a character
    pub line_height: f64,
    /// Side of a marker in hundredths of the chart's width
    pub marker: f64,
    /// The font of the text, as CSS names it
    pub font: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub left: f64,
    pub top: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    #[cfg(test)]
    pub fn right(&self) -> f64 {
        self.left + self.width
    }

    pub fn bottom(&self) -> f64 {
        self.top + self.height
    }

    /// The rectangle less `by` on every side.
    fn inset(&self, by: f64) -> Rect {
        Rect {
            left: self.left + by,
            top: self.top + by,
            width: self.width - 2.0 * by,
            height: self.height - 2.0 * by,
        }
    }
}

/// Where everything is on the chart.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    /// Height of the chart
    pub height: f64,
    /// Side of a marker
    pub marker: f64,
    /// Side of a marker module
    pub module: f64,
    /// The top-left corner of each marker, for the markers at the top-left, top-right, bottom-right and
    /// bottom-left of the text area
    pub corners: [(f64, f64); 4],
    /// The top-left corner of the marker that says what is on show: level with the top markers, half way
    /// between them
    pub signal: (f64, f64),
    /// What lies between the corners of the markers that face the middle, which is what `ocrs` cuts out
    pub area: Rect,
    /// The dark ground of light text, where the text is that way round: the area but for a module's width
    /// next to the markers, which it would otherwise run into
    pub panel: Option<Rect>,
    /// The top-left corner of the first character of a page
    pub text: (f64, f64),
    /// Width and height of a character
    pub cell: (f64, f64),
    /// Characters in a line
    pub columns: usize,
    /// Lines of text on a page
    pub lines: usize,
    /// From the top of one line of text to the top of the next, in heights of a character
    pub line_step: usize,
    /// Where the notes go, below the text area where `ocrs` does not look; `None` if markers this small
    /// leave no room for them
    pub notes: Option<Rect>,
}

impl Layout {
    /// The layout for markers of `marker_modules` modules a side, or what is wrong with what was asked for.
    pub fn new(marker_modules: usize, format: &Format, settings: &Settings) -> Result<Self> {
        let (across, down) = format.aspect;
        let height = WIDTH * down / across;
        let marker = format.marker;
        let module = marker / marker_modules as f64;
        let margin = settings.margin as f64 * module;
        // from an edge of the chart to the text area
        let edge = margin + marker;
        let area = Rect {
            left: edge,
            top: edge,
            width: WIDTH - 2.0 * edge,
            height: height - 2.0 * edge,
        };
        // the signal marker wants a module of paper between itself and the markers on either side
        ensure!(
            area.width >= marker + 2.0 * module && area.height > 0.0,
            "markers of {marker} % of the chart's width, {} modules away from its edges, leave no room \
             between them on a chart of {across}:{down}; a smaller --marker does",
            settings.margin
        );
        let ground = if settings.invert { area.inset(module) } else { area };
        ensure!(
            ground.width > 0.0 && ground.height > 0.0,
            "the markers leave no room for a dark panel between them; a smaller --marker does"
        );

        let pad = settings.pad as f64;
        let cell_width = ground.width / (format.columns as f64 + 4.0 * pad);
        let cell_height = cell_width * format.line_height;
        // rows that fit exactly are not to be lost to the last bit of a floating point number
        let rows = (ground.height / cell_height + 1e-9).floor() - 2.0 * pad;
        ensure!(
            rows >= 1.0,
            "with {} characters in a line, a line is too high for the text area of a chart of \
             {across}:{down}; more --columns, less --pad or a lower --line-height make room",
            format.columns
        );
        let rows = rows as usize;
        let line_step = settings.line_spacing + 1;
        // what the rows don't take up is shared between the top and the bottom
        let top = ground.top + (ground.height - rows as f64 * cell_height) / 2.0;

        let far_x = WIDTH - margin - marker;
        let far_y = height - margin - marker;
        // between the bottom markers, clear of them and of the text area
        let notes = Rect {
            left: area.left + 2.0 * module,
            top: area.bottom() + module,
            width: area.width - 4.0 * module,
            height: edge - 2.0 * module,
        };
        Ok(Self {
            height,
            marker,
            module,
            corners: [(margin, margin), (far_x, margin), (far_x, far_y), (margin, far_y)],
            signal: ((WIDTH - marker) / 2.0, margin),
            area,
            panel: settings.invert.then_some(ground),
            text: (ground.left + 2.0 * pad * cell_width, top),
            cell: (cell_width, cell_height),
            columns: format.columns,
            lines: (rows + settings.line_spacing) / line_step,
            line_step,
            notes: (notes.width > 0.0 && notes.height >= 2.0 * MIN_NOTE_LINE).then_some(notes),
        })
    }
}

/// The markers that tell `ocrs` what is on show.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signals {
    /// The start screen
    pub ready: Pattern,
    /// The end screen
    pub done: Pattern,
    /// The pages, by page number
    pub pages: Vec<Pattern>,
}

/// Everything a page shows.
#[derive(Clone, Debug, PartialEq)]
pub struct Show {
    /// What the text is called, for the title of the page
    pub title: String,
    pub layout: Layout,
    pub font: String,
    pub bold: bool,
    pub ink: Rgb,
    pub paper: Rgb,
    /// The markers at the top-left, top-right, bottom-right and bottom-left of the text area
    pub markers: [Pattern; 4],
    pub signals: Signals,
    /// The lines of the start screen
    pub start: Vec<String>,
    /// The lines of every page of the text
    pub pages: Vec<Vec<String>>,
    /// The lines of the end screen
    pub end: Vec<String>,
    /// Seconds after which a page makes way for the next one
    pub interval: Option<f64>,
    /// Whether the notes are on show to begin with
    pub notes: bool,
    /// What the layout measures, for the notes
    pub facts: String,
}

/// The page itself: all but the two places where a show goes in.
const TEMPLATE: &str = include_str!("page.html");
const TITLE_GOES_HERE: &str = "__TITLE__";
const SHOW_GOES_HERE: &str = "__SHOW__";

/// The web page that puts on `show`: one file with nothing to load from elsewhere.
pub fn page(show: &Show) -> String {
    let (head, rest) = TEMPLATE
        .split_once(TITLE_GOES_HERE)
        .expect("a place for the title");
    let (middle, tail) = rest.split_once(SHOW_GOES_HERE).expect("a place for the show");
    [head, &html_text(&show.title), middle, &json(show), tail].concat()
}

/// `show` as the object that the script of the page works with.
fn json(show: &Show) -> String {
    let layout = &show.layout;
    let at = |(left, top): (f64, f64)| Object::new().number("left", left).number("top", top);
    let rect = |rect: &Rect| {
        at((rect.left, rect.top))
            .number("width", rect.width)
            .number("height", rect.height)
            .end()
    };
    let or_null = |wanted: &Option<Rect>| wanted.as_ref().map_or("null".to_owned(), rect);
    let lines = |lines: &[String]| array(lines.iter().map(|line| quoted(line)));
    let colour = |colour: Rgb| quoted(&format!("#{colour}"));

    let corners = layout
        .corners
        .iter()
        .zip(&show.markers)
        .map(|(corner, pattern)| at(*corner).string("path", &path(pattern)).end());
    let signals = Object::new()
        .string("ready", &path(&show.signals.ready))
        .string("done", &path(&show.signals.done))
        .raw(
            "pages",
            array(show.signals.pages.iter().map(|pattern| quoted(&path(pattern)))),
        )
        .end();
    let text = at(layout.text)
        .raw("cell", array([number(layout.cell.0), number(layout.cell.1)]))
        .raw("step", layout.line_step.to_string())
        .end();
    Object::new()
        .string("title", &show.title)
        .number("height", layout.height)
        .raw("ink", colour(show.ink))
        .raw("paper", colour(show.paper))
        // plain to read from the chair, faint from where the camera is
        .raw("faint", colour(show.paper.towards(show.ink, 0.55)))
        .string("font", &show.font)
        .raw("bold", show.bold.to_string())
        .raw("side", show.markers[0].side().to_string())
        .number("marker", layout.marker)
        .raw("corners", array(corners))
        .raw("signal", at(layout.signal).end())
        .raw("signals", signals)
        .raw("panel", or_null(&layout.panel))
        .raw("text", text)
        .raw("notes", or_null(&layout.notes))
        .raw("showNotes", show.notes.to_string())
        .string("facts", &show.facts)
        .raw("interval", show.interval.map_or("null".to_owned(), number))
        .raw("start", lines(&show.start))
        .raw("pages", array(show.pages.iter().map(|page| lines(page))))
        .raw("end", lines(&show.end))
        .end()
}

/// The dark modules of `pattern` as an SVG path, on a grid of one unit a module: a rectangle for every run
/// of them in a row.
fn path(pattern: &Pattern) -> String {
    let side = pattern.side();
    let mut path = String::new();
    for row in 0..side {
        let mut column = 0;
        while column < side {
            let start = column;
            while column < side && pattern.is_dark(row, column) {
                column += 1;
            }
            if column > start {
                let run = column - start;
                write!(path, "M{start} {row}h{run}v1h-{run}z").expect("writing to a string");
            } else {
                column += 1;
            }
        }
    }
    path
}

/// A JSON object in the making.
struct Object(String);

impl Object {
    fn new() -> Self {
        Self("{".to_owned())
    }

    /// Adds a member whose value is JSON already.
    fn raw(mut self, key: &str, value: impl AsRef<str>) -> Self {
        if self.0.len() > 1 {
            self.0.push(',');
        }
        self.0.push_str(&quoted(key));
        self.0.push(':');
        self.0.push_str(value.as_ref());
        self
    }

    fn string(self, key: &str, value: &str) -> Self {
        self.raw(key, quoted(value))
    }

    fn number(self, key: &str, value: f64) -> Self {
        self.raw(key, number(value))
    }

    fn end(mut self) -> String {
        self.0.push('}');
        self.0
    }
}

fn array(items: impl IntoIterator<Item = String>) -> String {
    format!("[{}]", items.into_iter().collect::<Vec<_>>().join(","))
}

/// A number with as many of five decimals as it has, which on a chart 10,000 pixels wide is a thousandth
/// of a pixel.
fn number(value: f64) -> String {
    let text = format!("{value:.5}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text == "-0" {
        "0".to_owned()
    } else {
        text.to_owned()
    }
}

/// `text` as a JSON string that can stand in a script of a web page: nothing in it ends the script or
/// starts a comment, whatever the text is.
fn quoted(text: &str) -> String {
    let mut quoted = String::with_capacity(text.len() + 2);
    quoted.push('"');
    for c in text.chars() {
        match c {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            '\n' => quoted.push_str("\\n"),
            // the line separators are line ends to older scripts
            '<' | '>' | '&' | '\u{2028}' | '\u{2029}' => {
                write!(quoted, "\\u{:04x}", c as u32).expect("writing to a string")
            }
            c if c.is_control() => write!(quoted, "\\u{:04x}", c as u32).expect("writing to a string"),
            c => quoted.push(c),
        }
    }
    quoted.push('"');
    quoted
}

/// `text` as it is written between two tags.
fn html_text(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use ocrs::markers::{self, Locator, Signal};
    use ocrs::ocr::Models;
    use ocrs::{Device, Options, Outcome, Pipeline};
    use opencv::core::{self, CV_8UC3, Mat, Point, Point2f, Scalar, Size, Vector};
    use opencv::imgproc;
    use opencv::prelude::*;

    use super::*;

    fn format() -> Format {
        Format {
            aspect: (4.0, 3.0),
            columns: 80,
            line_height: 2.0,
            marker: 6.0,
            font: "monospace".to_owned(),
        }
    }

    fn settings() -> Settings {
        Settings {
            // of no use to a page
            module: 2,
            margin: 1,
            pad: 1,
            line_spacing: 0,
            invert: false,
            bold: false,
            ink: Rgb::BLACK,
            paper: Rgb::WHITE,
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn close_rect(rect: &Rect, (left, top, width, height): (f64, f64, f64, f64)) -> bool {
        close(rect.left, left)
            && close(rect.top, top)
            && close(rect.width, width)
            && close(rect.height, height)
    }

    #[test]
    fn text_area_lies_between_the_markers() {
        // markers of 6 % in 6 modules of 1 %, behind a margin of one module, on a chart of 100 by 75
        let layout = Layout::new(6, &format(), &settings()).unwrap();
        assert!(close(layout.height, 75.0) && close(layout.module, 1.0));
        assert_eq!(
            layout.corners,
            [(1.0, 1.0), (93.0, 1.0), (93.0, 68.0), (1.0, 68.0)]
        );
        assert_eq!(layout.signal, (47.0, 1.0));
        assert!(
            close_rect(&layout.area, (7.0, 7.0, 86.0, 61.0)),
            "{:?}",
            layout.area
        );
        assert_eq!(layout.panel, None);
        // 80 characters and twice two of padding share the 86: a character is 86/84 wide and twice as high
        let width = 86.0 / 84.0;
        assert!(close(layout.cell.0, width) && close(layout.cell.1, 2.0 * width));
        // 29 rows fit into the 61, of which two are padding
        assert_eq!((layout.columns, layout.lines, layout.line_step), (80, 27, 1));
        let spare = 61.0 - 27.0 * 2.0 * width;
        assert!(close(layout.text.0, 7.0 + 2.0 * width) && close(layout.text.1, 7.0 + spare / 2.0));
        // the notes keep a module away from the area and the bottom, two from the markers
        assert!(close_rect(&layout.notes.unwrap(), (9.0, 69.0, 82.0, 5.0)));
    }

    #[test]
    fn more_characters_make_smaller_text_and_leave_the_markers_alone() {
        let narrow = Layout::new(6, &format(), &settings()).unwrap();
        let wide = Layout::new(
            6,
            &Format {
                columns: 160,
                ..format()
            },
            &settings(),
        )
        .unwrap();
        assert_eq!(
            (wide.marker, wide.corners, wide.area),
            (narrow.marker, narrow.corners, narrow.area)
        );
        assert!(close(wide.cell.0, 86.0 / 164.0));
        // 86/164 wide and twice as high: 58 rows in 61, less the padding
        assert_eq!(wide.lines, 56);

        // and smaller markers leave more room for text of the same number of characters
        let small = Layout::new(
            6,
            &Format {
                marker: 3.0,
                ..format()
            },
            &settings(),
        )
        .unwrap();
        assert!(close(small.area.width, 93.0) && small.cell.0 > narrow.cell.0);
    }

    #[test]
    fn lines_follow_the_shape_of_the_chart_and_of_the_characters() {
        let lines = |format: Format, settings: Settings| Layout::new(6, &format, &settings).unwrap().lines;
        // 16:9 is 56.25 high, which leaves 42.25 between the markers
        let wide = Format {
            aspect: (16.0, 9.0),
            ..format()
        };
        assert_eq!(lines(wide, settings()), 18);
        // lines of one and a half widths: 39 in 61
        let tight = Format {
            line_height: 1.5,
            ..format()
        };
        assert_eq!(lines(tight, settings()), 37);
        // an empty row after every line: 27 rows are 14 lines
        let spaced = Settings {
            line_spacing: 1,
            ..settings()
        };
        let layout = Layout::new(6, &format(), &spaced).unwrap();
        assert_eq!((layout.lines, layout.line_step), (14, 2));
        // without padding the characters are 86/80 wide and 28 rows fit
        let bare = Settings { pad: 0, ..settings() };
        let layout = Layout::new(6, &format(), &bare).unwrap();
        assert!(close(layout.cell.0, 86.0 / 80.0) && close(layout.text.0, 7.0));
        assert_eq!(layout.lines, 28);
    }

    #[test]
    fn rows_that_fit_exactly_are_all_there() {
        // 10 characters without padding are 8.6 wide, and lines of 61/86 of that are 6.1 high: ten of them
        let exact = Format {
            columns: 10,
            line_height: 61.0 / 86.0,
            ..format()
        };
        let bare = Settings { pad: 0, ..settings() };
        assert_eq!(Layout::new(6, &exact, &bare).unwrap().lines, 10);
    }

    #[test]
    fn dark_panel_keeps_a_module_away_from_the_markers() {
        let inverted = Settings {
            invert: true,
            ..settings()
        };
        let layout = Layout::new(6, &format(), &inverted).unwrap();
        assert!(close_rect(&layout.panel.unwrap(), (8.0, 8.0, 84.0, 59.0)));
        assert!(close(layout.cell.0, 1.0) && close(layout.text.0, 10.0));
        // 29 rows of 2 in 59, less the padding
        assert_eq!(layout.lines, 27);
    }

    #[test]
    fn layout_without_room_is_turned_down() {
        let layout = |format: Format, settings: Settings| Layout::new(6, &format, &settings);
        // on a chart of 16:9, eight characters are so large that not one line fits with its padding
        let huge = Format {
            aspect: (16.0, 9.0),
            columns: 8,
            ..format()
        };
        let error = layout(huge.clone(), settings()).unwrap_err().to_string();
        assert!(error.contains("8 characters in a line"), "{error}");
        let bare = Settings { pad: 0, ..settings() };
        assert_eq!(layout(huge, bare).unwrap().lines, 1);

        // three markers of 30 % and their margins are wider than the chart
        let large = Format {
            marker: 30.0,
            ..format()
        };
        let error = layout(large, settings()).unwrap_err().to_string();
        assert!(error.contains("leave no room between them"), "{error}");
        // markers of 20 % are higher than half a chart of 3:1
        let flat = Format {
            aspect: (3.0, 1.0),
            marker: 20.0,
            ..format()
        };
        assert!(layout(flat, settings()).is_err());

        // small markers leave no room for notes, and that is all
        let small = Format {
            marker: 2.0,
            ..format()
        };
        assert_eq!(layout(small, settings()).unwrap().notes, None);
    }

    fn pattern(id: i32) -> Pattern {
        markers::pattern("4x4_50", id).unwrap()
    }

    /// Top-left, top-right, bottom-right, bottom-left
    const IDS: [i32; 4] = [7, 3, 11, 5];

    /// A show of one page with these lines, that says `signal` on that page.
    fn show(layout: Layout, lines: &[&str], signal: Signal) -> (Show, Pattern) {
        let signals = markers::Signals::new("4x4_50", IDS).unwrap();
        let show = Show {
            title: "notes.txt".to_owned(),
            layout,
            font: "monospace".to_owned(),
            bold: false,
            ink: Rgb::BLACK,
            paper: Rgb::WHITE,
            markers: IDS.map(pattern),
            signals: Signals {
                ready: pattern(signals.id(Signal::Ready)),
                done: pattern(signals.id(Signal::Done)),
                pages: (0..markers::PAGE_NUMBERS)
                    .map(|number| pattern(signals.id(Signal::Page(number))))
                    .collect(),
            },
            start: vec!["Enter starts.".to_owned()],
            pages: vec![lines.iter().map(|line| (*line).to_owned()).collect()],
            end: vec!["End.".to_owned()],
            interval: None,
            notes: true,
            facts: "80x27 characters".to_owned(),
        };
        let fifth = pattern(signals.id(signal));
        (show, fifth)
    }

    /// The first page of `show` as an image of `scale` pixels to a unit of the layout, with the marker
    /// `signal` between the top two. Text is drawn in a Hershey font that fills its cells about as a
    /// browser's font does.
    fn rasterize(show: &Show, signal: &Pattern, scale: f64) -> Result<Mat> {
        let layout = &show.layout;
        let pixels = |length: f64| (length * scale).round() as i32;
        let bgr = |Rgb(r, g, b): Rgb| Scalar::new(b as f64, g as f64, r as f64, 0.0);
        let size = Size::new(pixels(WIDTH), pixels(layout.height));
        let mut image = Mat::new_size_with_default(size, CV_8UC3, bgr(show.paper))?;
        let mut fill = |left: f64, top: f64, right: f64, bottom: f64, colour: Rgb| {
            let rect = core::Rect::from_points(
                Point::new(pixels(left), pixels(top)),
                Point::new(pixels(right), pixels(bottom)),
            );
            imgproc::rectangle(&mut image, rect, bgr(colour), imgproc::FILLED, imgproc::LINE_8, 0)
        };
        let patterns = show.markers.iter().chain([signal]);
        for (pattern, (left, top)) in patterns.zip(layout.corners.into_iter().chain([layout.signal])) {
            for row in 0..pattern.side() {
                for column in 0..pattern.side() {
                    if pattern.is_dark(row, column) {
                        let (x, y) = (
                            left + column as f64 * layout.module,
                            top + row as f64 * layout.module,
                        );
                        fill(x, y, x + layout.module, y + layout.module, show.ink)?;
                    }
                }
            }
        }
        let colour = match &layout.panel {
            Some(panel) => {
                fill(panel.left, panel.top, panel.right(), panel.bottom(), show.ink)?;
                show.paper
            }
            None => show.ink,
        };
        let (cell_width, cell_height) = layout.cell;
        for (index, line) in show.pages[0].iter().enumerate() {
            let font = imgproc::FONT_HERSHEY_DUPLEX;
            // as large as the line allows, but no wider than the characters of the line
            let mut size = cell_height * scale / 40.0;
            let drawn = imgproc::get_text_size(line, font, size, 2, &mut 0)?.width;
            let room = pixels(line.chars().count() as f64 * cell_width);
            if drawn > room {
                size *= room as f64 / drawn as f64;
            }
            let top = layout.text.1 + (index * layout.line_step) as f64 * cell_height;
            let baseline = Point::new(pixels(layout.text.0), pixels(top + 0.75 * cell_height));
            imgproc::put_text(
                &mut image,
                line,
                baseline,
                font,
                size,
                bgr(colour),
                2,
                imgproc::LINE_AA,
                false,
            )?;
        }
        Ok(image)
    }

    /// A camera frame of 1280 by 960 pixels that shows the chart a little smaller and in perspective in a
    /// dim room; returns where `area` of the chart is in it as well.
    fn photograph(chart: &Mat, area: &Rect, scale: f64) -> Result<(Mat, [Point2f; 4])> {
        let size = chart.size()?;
        let (w, h) = ((size.width - 1) as f32, (size.height - 1) as f32);
        let flat = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].map(|(x, y)| Point2f::new(x, y));
        let seen =
            [(70.0, 60.0), (1215.0, 35.0), (1250.0, 915.0), (40.0, 890.0)].map(|(x, y)| Point2f::new(x, y));
        let transform = imgproc::get_perspective_transform_def(
            &Vector::<Point2f>::from_slice(&flat),
            &Vector::<Point2f>::from_slice(&seen),
        )?;
        let mut frame = Mat::default();
        imgproc::warp_perspective(
            chart,
            &mut frame,
            &transform,
            Size::new(1280, 960),
            imgproc::INTER_AREA,
            core::BORDER_CONSTANT,
            Scalar::all(40.0),
        )?;
        let on_chart = |x: f64, y: f64| Point2f::new((x * scale) as f32, (y * scale) as f32);
        let corners = Vector::<Point2f>::from_slice(&[
            on_chart(area.left, area.top),
            on_chart(area.right(), area.top),
            on_chart(area.right(), area.bottom()),
            on_chart(area.left, area.bottom()),
        ]);
        let mut in_frame = Vector::<Point2f>::new();
        core::perspective_transform(&corners, &mut in_frame, &transform)?;
        Ok((
            frame,
            std::array::from_fn(|corner| in_frame.get(corner).expect("four corners")),
        ))
    }

    #[test]
    fn locator_finds_the_text_area_of_a_photographed_page() -> Result<()> {
        let inverted = Settings {
            invert: true,
            ..settings()
        };
        let wide = Format {
            aspect: (16.0, 9.0),
            marker: 4.0,
            ..format()
        };
        // markers of 5 % of a chart that is some 1150 pixels wide in the frame: modules of under 10 pixels
        let usual = Format {
            marker: 5.0,
            ..format()
        };
        let cases = [
            (usual, settings(), Signal::Ready),
            (format(), inverted, Signal::Page(9)),
            (wide, settings(), Signal::Done),
        ];
        for (format, settings, signal) in cases {
            let layout = Layout::new(6, &format, &settings)?;
            let (show, fifth) = show(layout, &["Some text", "to look past"], signal);
            let (frame, area) = photograph(&rasterize(&show, &fifth, 16.0)?, &show.layout.area, 16.0)?;

            let label = format!("{format:?}");
            let found = Locator::new("4x4_50", IDS)?.locate(&frame)?;
            let sighting = found.unwrap_or_else(|| panic!("{label}: markers not found"));
            assert_eq!(sighting.signal, Some(signal), "{label}");
            for (corner, expected) in sighting.area.iter().zip(area) {
                assert!(
                    (corner.x - expected.x).abs() <= 1.5 && (corner.y - expected.y).abs() <= 1.5,
                    "{label}: corner {corner:?} is not at {expected:?}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn pipeline_reads_a_photographed_page() -> Result<()> {
        let lines = [
            "Invoice 2026-0417",
            "",
            "Total due: 1,284.50 USD",
            "Thank you for your order!",
        ];
        let inverted = Settings {
            invert: true,
            ..settings()
        };
        for settings in [settings(), inverted] {
            // 48 characters in a line are some 20 pixels each in the frame
            let format = Format {
                columns: 48,
                marker: 5.0,
                ..format()
            };
            let layout = Layout::new(6, &format, &settings)?;
            let (show, fifth) = show(layout, &lines, Signal::Page(0));
            let (frame, _) = photograph(&rasterize(&show, &fifth, 16.0)?, &show.layout.area, 16.0)?;
            let mut pipeline = Pipeline::new(Options {
                device: Device::Cpu,
                dictionary: "4x4_50".to_owned(),
                marker_ids: IDS,
                page_size: None,
                stable_frames: 1,
                min_similarity: 0.9,
                models: Models::built_in()
                    .expect("no models in model/: tools/download_models.py fetches them"),
            })?;
            assert_eq!(
                pipeline.process(&frame)?,
                Outcome::Text(
                    "Invoice 2026-0417\nTotal due: 1,284.50 USD\nThank you for your order!".to_owned()
                ),
                "invert {}",
                settings.invert
            );
        }
        Ok(())
    }

    #[test]
    fn marker_path_draws_the_dark_modules_row_by_row() {
        let pattern = pattern(0);
        let path = path(&pattern);
        // the border is dark all around: a run of six at the top and at the bottom
        assert!(path.starts_with("M0 0h6v1h-6z"), "{path}");
        assert!(path.ends_with("M0 5h6v1h-6z"), "{path}");
        // every run covers dark modules only, and together they cover all of them
        let mut covered = vec![false; 36];
        for run in path.split('M').skip(1) {
            let (at, rest) = run.split_once('h').unwrap();
            let (column, row) = at.split_once(' ').unwrap();
            let (column, row): (usize, usize) = (column.parse().unwrap(), row.parse().unwrap());
            let length: usize = rest.split_once('v').unwrap().0.parse().unwrap();
            assert_eq!(rest, format!("{length}v1h-{length}z"));
            for column in column..column + length {
                assert!(pattern.is_dark(row, column) && !covered[row * 6 + column]);
                covered[row * 6 + column] = true;
            }
        }
        let dark = (0..36).map(|module| pattern.is_dark(module / 6, module % 6));
        assert_eq!(covered, dark.collect::<Vec<_>>());
    }

    #[test]
    fn numbers_and_strings_are_written_as_a_script_takes_them() {
        assert_eq!(number(75.0), "75");
        assert_eq!(number(86.0 / 84.0), "1.02381");
        assert_eq!(number(0.5), "0.5");
        assert_eq!(number(-0.000001), "0");
        assert_eq!(number(100.0), "100");
        assert_eq!(quoted("plain"), r#""plain""#);
        assert_eq!(quoted("a \"b\" \\ c\n\td"), r#""a \"b\" \\ c\n\u0009d""#);
        // nothing that a browser takes for the end of the script, or for markup
        assert_eq!(
            quoted("</script><!-- & \u{2028}"),
            r#""\u003c/script\u003e\u003c!-- \u0026 \u2028""#
        );
        assert_eq!(quoted("漢字 é"), "\"漢字 é\"");
        assert_eq!(html_text("a <b> & c"), "a &lt;b&gt; &amp; c");
    }

    #[test]
    fn page_carries_the_show() -> Result<()> {
        let layout = Layout::new(6, &format(), &settings())?;
        let (mut show, _) = show(
            layout,
            &["first line", "", "  third </script> \"line\""],
            Signal::Page(0),
        );
        show.title = "notes <1>.txt".to_owned();
        show.interval = Some(2.5);
        let page = page(&show);

        assert!(page.starts_with("<!doctype html>"));
        assert!(
            page.contains("<title>ocrs-show: notes &lt;1&gt;.txt</title>"),
            "no title"
        );
        // nothing is left to fill in, and the text can't end the script: the one script ends once
        assert!(!page.contains(TITLE_GOES_HERE) && !page.contains(SHOW_GOES_HERE));
        assert_eq!(page.matches("</script>").count(), 1);
        for part in [
            r#""title":"notes \u003c1\u003e.txt""#,
            r##""height":75,"ink":"#000000","paper":"#ffffff","faint":"#737373""##,
            r#""font":"monospace","bold":false,"side":6,"marker":6,"#,
            r#""corners":[{"left":1,"top":1,"path":"M0 0h6v1h-6z"#,
            r#""signal":{"left":47,"top":1},"signals":{"ready":"M0 0h6v1h-6z"#,
            r#""panel":null,"text":{"left":9.04762,"top":9.85714,"cell":[1.02381,2.04762],"step":1}"#,
            r#""notes":{"left":9,"top":69,"width":82,"height":5},"showNotes":true,"facts":"80x27 characters""#,
            r#""interval":2.5,"start":["Enter starts."]"#,
            r#""pages":[["first line","","  third \u003c/script\u003e \"line\""]],"end":["End."]}"#,
        ] {
            assert!(page.contains(part), "{part} is not in the page");
        }
        // a marker for every page number
        let signals = page
            .split_once(r#""signals":"#)
            .unwrap()
            .1
            .split_once(r#""panel""#)
            .unwrap()
            .0;
        assert_eq!(signals.matches("M0 0h6v1h-6z").count(), 2 + markers::PAGE_NUMBERS);

        // a dark panel, no interval, notes off
        let inverted = Settings {
            invert: true,
            ..settings()
        };
        let (mut show, _) = show_of(Layout::new(6, &format(), &inverted)?);
        show.notes = false;
        let page = super::page(&show);
        for part in [
            r#""panel":{"left":8,"top":8,"width":84,"height":59}"#,
            r#""showNotes":false"#,
            r#""interval":null"#,
        ] {
            assert!(page.contains(part), "{part} is not in the page");
        }
        Ok(())
    }

    fn show_of(layout: Layout) -> (Show, Pattern) {
        show(layout, &["text"], Signal::Page(0))
    }
}
