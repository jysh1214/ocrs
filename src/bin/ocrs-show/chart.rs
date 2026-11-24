//! What goes on the terminal: the four markers in its corners, a page of text in the area between them, and
//! between the top two markers a fifth one that tells `ocrs` which screen this is.
//!
//! The picture is made of dots of half a character cell each, one column wide and half a row high. A
//! terminal can draw those with the half block characters, and in the usual fonts they are about square. A
//! marker module is a square of dots, so the markers come out about square as well.

use std::fmt;
use std::ops::Range;
use std::str::FromStr;

use ocrs::markers::Pattern;
use unicode_width::UnicodeWidthStr;

use crate::text::{Line, fit};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    pub const BLACK: Rgb = Rgb(0, 0, 0);
    pub const WHITE: Rgb = Rgb(255, 255, 255);

    /// Perceived brightness, 0 to 255.
    pub fn luma(self) -> f32 {
        0.299 * self.0 as f32 + 0.587 * self.1 as f32 + 0.114 * self.2 as f32
    }

    /// The colour that lies `share` (0 to 1) of the way from this one to `other`.
    fn towards(self, other: Rgb, share: f32) -> Rgb {
        let mix = |from: u8, to: u8| (from as f32 + (to as f32 - from as f32) * share).round() as u8;
        Rgb(mix(self.0, other.0), mix(self.1, other.1), mix(self.2, other.2))
    }
}

impl FromStr for Rgb {
    type Err = String;

    /// `RRGGBB` in hexadecimal, with or without a `#` in front.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let digits = text.strip_prefix('#').unwrap_or(text);
        if digits.len() != 6 || !digits.bytes().all(|digit| digit.is_ascii_hexdigit()) {
            return Err(format!(
                "expected a colour as RRGGBB such as d0d0d0, got {text:?}"
            ));
        }
        let channel = |at: usize| u8::from_str_radix(&digits[at..at + 2], 16).expect("two hex digits");
        Ok(Rgb(channel(0), channel(2), channel(4)))
    }
}

impl fmt::Display for Rgb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02x}{:02x}{:02x}", self.0, self.1, self.2)
    }
}

/// How the chart is laid out and coloured; the command line options and keys of the same names set these.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Side of a marker module in dots: this many columns, half as many rows
    pub module: usize,
    /// Paper between the edges of the terminal and the markers, in modules
    pub margin: usize,
    /// Space between the text and the edge of its background: this many rows, twice as many columns
    pub pad: usize,
    /// Empty rows between two lines of text
    pub line_spacing: usize,
    /// Text in the paper colour on a panel in the ink colour, instead of ink on paper
    pub invert: bool,
    pub bold: bool,
    /// The dark colour: marker modules and text
    pub ink: Rgb,
    /// The light colour: everything else
    pub paper: Rgb,
}

/// Narrower text areas than this are not worth showing.
const MIN_TEXT_COLUMNS: usize = 8;

/// A rectangle of character cells: the columns `left..right` of the rows `top..bottom`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cells {
    pub left: usize,
    pub top: usize,
    pub right: usize,
    pub bottom: usize,
}

impl Cells {
    pub fn width(&self) -> usize {
        self.right - self.left
    }

    pub fn height(&self) -> usize {
        self.bottom - self.top
    }

    /// The rectangle less `columns` on its left and right and `rows` at its top and bottom, if that leaves
    /// anything.
    fn inset(&self, columns: usize, rows: usize) -> Option<Cells> {
        let inner = Cells {
            left: self.left + columns,
            top: self.top + rows,
            right: self.right.checked_sub(columns)?,
            bottom: self.bottom.checked_sub(rows)?,
        };
        (inner.left < inner.right && inner.top < inner.bottom).then_some(inner)
    }
}

/// Where everything is on a terminal of a given size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub columns: usize,
    pub rows: usize,
    /// Side of a module in dots
    module: usize,
    /// Side of a marker in dots
    pub marker: usize,
    /// Dots between an edge of the terminal and the markers next to it
    margin: usize,
    /// The cells that lie entirely between the corners of the markers that face the middle, which is the
    /// area `ocrs` cuts out. Where a marker ends in the middle of a row the area reaches half a row
    /// further than these cells.
    pub area: Cells,
    /// The part of the area that has the background of the text: all of it when that is paper, and all but
    /// a module's width next to the markers when it is ink, which would otherwise run into them
    panel: Cells,
    /// Where the text goes
    pub text: Cells,
}

impl Geometry {
    /// `None` if a terminal of `columns` by `rows` has no room for a usable text area between markers of
    /// `marker_modules` modules a side.
    pub fn new(columns: usize, rows: usize, marker_modules: usize, settings: &Settings) -> Option<Self> {
        let marker = marker_modules * settings.module;
        let margin = settings.margin * settings.module;
        // from an edge of the terminal to the text area, in dots and in the rows those take up
        let edge = margin + marker;
        let band = edge.div_ceil(2);
        let area = Cells {
            left: edge,
            top: band,
            right: columns.checked_sub(edge)?,
            bottom: rows.checked_sub(band)?,
        };
        let panel = if settings.invert {
            area.inset(settings.module, settings.module.div_ceil(2))?
        } else {
            area.inset(0, 0)?
        };
        let text = panel.inset(2 * settings.pad, settings.pad)?;
        // the signal marker wants a module of paper between itself and the markers on either side
        let room_for_signal = area.width() >= marker + 2 * settings.module;
        (text.width() >= MIN_TEXT_COLUMNS && room_for_signal).then_some(Self {
            columns,
            rows,
            module: settings.module,
            marker,
            margin,
            area,
            panel,
            text,
        })
    }

    /// How many lines of text a page has.
    pub fn lines_per_page(&self, line_spacing: usize) -> usize {
        (self.text.height() + line_spacing) / (line_spacing + 1)
    }

    /// The dot at the top-left of each marker, for the markers at the top-left, top-right, bottom-right
    /// and bottom-left of the text area.
    fn marker_origins(&self) -> [(usize, usize); 4] {
        let near = self.margin;
        let far_x = self.columns - self.margin - self.marker;
        let far_y = 2 * self.rows - self.margin - self.marker;
        [(near, near), (far_x, near), (far_x, far_y), (near, far_y)]
    }

    /// The dot at the top-left of the signal marker: level with the top markers, half way between them.
    fn signal_origin(&self) -> (usize, usize) {
        ((self.columns - self.marker) / 2, self.margin)
    }

    /// The cells for the notes, below the text area: between the markers but two modules clear of them,
    /// and clear of the row next to the area, which it may reach into.
    fn note_cells(&self) -> Option<Cells> {
        let cells = Cells {
            left: self.area.left + 2 * self.module,
            top: self.area.bottom + 1,
            right: self.area.right.checked_sub(2 * self.module)?,
            bottom: self.rows,
        };
        cells.inset(0, 0)
    }
}

/// A piece of text at a place on the screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    pub column: usize,
    pub row: usize,
    pub text: String,
    pub colour: Rgb,
    pub background: Rgb,
    pub bold: bool,
}

/// A terminal's worth of picture: a colour for every dot, and text to write on top of that.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Screen {
    pub columns: usize,
    pub rows: usize,
    /// Row by row, two rows of dots for every row of cells
    dots: Vec<Rgb>,
    pub runs: Vec<Run>,
}

impl Screen {
    fn new(columns: usize, rows: usize, colour: Rgb) -> Self {
        Self {
            columns,
            rows,
            dots: vec![colour; columns * rows * 2],
            runs: Vec::new(),
        }
    }

    /// The colour of the dot in column `x` and half-row `y`.
    pub fn dot(&self, x: usize, y: usize) -> Rgb {
        self.dots[y * self.columns + x]
    }

    fn fill_dots(&mut self, xs: Range<usize>, ys: Range<usize>, colour: Rgb) {
        for y in ys {
            let row = y * self.columns;
            self.dots[row + xs.start..row + xs.end].fill(colour);
        }
    }

    fn fill_cells(&mut self, cells: Cells, colour: Rgb) {
        self.fill_dots(cells.left..cells.right, 2 * cells.top..2 * cells.bottom, colour);
    }

    /// Writes `lines` into the middle of `cells`, as many as there are rows for and as much of each as
    /// there are columns for.
    fn centre(&mut self, cells: Cells, lines: &[String], colour: Rgb, background: Rgb) {
        let shown = lines.len().min(cells.height());
        let top = cells.top + (cells.height() - shown) / 2;
        for (index, line) in lines[..shown].iter().enumerate() {
            let text = fit(line, cells.width());
            if !text.is_empty() {
                self.runs.push(Run {
                    column: cells.left + (cells.width() - text.width()) / 2,
                    row: top + index,
                    text: text.to_owned(),
                    colour,
                    background,
                    bold: false,
                });
            }
        }
    }

    /// A screen that only says something, for when there is no room for the chart.
    pub fn notice(columns: usize, rows: usize, settings: &Settings, lines: &[String]) -> Self {
        let mut screen = Self::new(columns, rows, settings.paper);
        let everything = Cells {
            left: 0,
            top: 0,
            right: columns,
            bottom: rows,
        };
        screen.centre(everything, lines, settings.ink, settings.paper);
        screen
    }
}

/// The chart: `markers` (top-left, top-right, bottom-right, bottom-left) around `page`, the marker `signal`
/// between the top two, and below the text area, where `ocrs` does not look, as many of the `notes` for
/// whoever sits at the terminal as there is room for.
pub fn compose(
    geometry: &Geometry,
    settings: &Settings,
    markers: &[Pattern; 4],
    signal: &Pattern,
    page: &[Line],
    notes: &[String],
) -> Screen {
    let mut screen = Screen::new(geometry.columns, geometry.rows, settings.paper);
    let module = settings.module;
    let origins = geometry
        .marker_origins()
        .into_iter()
        .chain([geometry.signal_origin()]);
    for (pattern, (x, y)) in markers.iter().chain([signal]).zip(origins) {
        for row in 0..pattern.side() {
            for column in 0..pattern.side() {
                if pattern.is_dark(row, column) {
                    let (left, top) = (x + column * module, y + row * module);
                    screen.fill_dots(left..left + module, top..top + module, settings.ink);
                }
            }
        }
    }

    let (colour, background) = if settings.invert {
        screen.fill_cells(geometry.panel, settings.ink);
        (settings.paper, settings.ink)
    } else {
        (settings.ink, settings.paper)
    };
    let rows = (geometry.text.top..geometry.text.bottom).step_by(settings.line_spacing + 1);
    for (line, row) in page.iter().zip(rows) {
        if !line.text.is_empty() {
            screen.runs.push(Run {
                column: geometry.text.left,
                row,
                text: line.text.clone(),
                colour,
                background,
                bold: settings.bold,
            });
        }
    }

    if let Some(cells) = geometry.note_cells() {
        // plain to read from the chair, faint from where the camera is
        let faint = settings.paper.towards(settings.ink, 0.55);
        screen.centre(cells, notes, faint, settings.paper);
    }
    screen
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use ocrs::markers::{self, Locator, Signal, Signals};
    use ocrs::ocr::Models;
    use ocrs::{Device, Options, Outcome, Pipeline};
    use opencv::core::{self, CV_8UC3, Mat, Point, Point2f, Rect, Scalar, Size, Vector};
    use opencv::imgproc;
    use opencv::prelude::*;

    use super::*;

    fn settings() -> Settings {
        Settings {
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

    fn patterns(dictionary: &str, ids: [i32; 4]) -> [Pattern; 4] {
        ids.map(|id| markers::pattern(dictionary, id).unwrap())
    }

    /// The marker that says `signal` on a chart with these corner markers.
    fn signal_pattern(dictionary: &str, ids: [i32; 4], signal: Signal) -> Pattern {
        let id = Signals::new(dictionary, ids).unwrap().id(signal);
        markers::pattern(dictionary, id).unwrap()
    }

    /// The chart with the default markers and the signal for the first page.
    fn chart(geometry: &Geometry, settings: &Settings, page: &[Line], notes: &[String]) -> Screen {
        let ids = [0, 1, 2, 3];
        let signal = signal_pattern("4x4_50", ids, Signal::Page(0));
        compose(geometry, settings, &patterns("4x4_50", ids), &signal, page, notes)
    }

    /// More and longer notes than fit anywhere.
    fn notes() -> Vec<String> {
        vec!["x".repeat(200); 7]
    }

    fn lines(texts: &[&str]) -> Vec<Line> {
        let line = |(number, text): (usize, &&str)| Line {
            text: (*text).to_owned(),
            at: (number, 0),
        };
        texts.iter().enumerate().map(line).collect()
    }

    /// The screen as an image, with character cells of `cell` pixels (width, height). Text is drawn in a
    /// Hershey font of about the size a terminal's font has in such a cell.
    fn rasterize(screen: &Screen, cell: (i32, i32)) -> Result<Mat> {
        let (width, height) = cell;
        assert_eq!(height % 2, 0, "a dot is half a cell high");
        let size = Size::new(screen.columns as i32 * width, screen.rows as i32 * height);
        let mut image = Mat::new_size_with_default(size, CV_8UC3, Scalar::all(0.0))?;
        let bgr = |Rgb(r, g, b): Rgb| Scalar::new(b as f64, g as f64, r as f64, 0.0);
        for y in 0..2 * screen.rows {
            for x in 0..screen.columns {
                let dot = Rect::new(x as i32 * width, y as i32 * height / 2, width, height / 2);
                let colour = bgr(screen.dot(x, y));
                imgproc::rectangle(&mut image, dot, colour, imgproc::FILLED, imgproc::LINE_8, 0)?;
            }
        }
        for run in &screen.runs {
            let font = imgproc::FONT_HERSHEY_DUPLEX;
            // as large as the row allows, but no wider than the cells of the run: text on a terminal
            // never leaves its cells
            let mut scale = height as f64 / 40.0;
            let drawn = imgproc::get_text_size(&run.text, font, scale, 2, &mut 0)?.width;
            let room = run.text.width() as i32 * width;
            if drawn > room {
                scale *= room as f64 / drawn as f64;
            }
            let baseline = (run.row as i32 + 1) * height - height / 4;
            imgproc::put_text(
                &mut image,
                &run.text,
                Point::new(run.column as i32 * width, baseline),
                font,
                scale,
                bgr(run.colour),
                2,
                imgproc::LINE_AA,
                false,
            )?;
        }
        Ok(image)
    }

    /// A camera frame that shows the screen smaller and in perspective in a dim room.
    fn photograph(screen: &Mat) -> Result<Mat> {
        let size = screen.size()?;
        let (w, h) = ((size.width - 1) as f32, (size.height - 1) as f32);
        let flat = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].map(|(x, y)| Point2f::new(x, y));
        let seen = [(210.0, 130.0), (1650.0, 90.0), (1740.0, 960.0), (150.0, 900.0)]
            .map(|(x, y)| Point2f::new(x, y));
        let transform = imgproc::get_perspective_transform_def(
            &Vector::<Point2f>::from_slice(&flat),
            &Vector::<Point2f>::from_slice(&seen),
        )?;
        let mut frame = Mat::default();
        imgproc::warp_perspective(
            screen,
            &mut frame,
            &transform,
            Size::new(1920, 1080),
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            Scalar::all(40.0),
        )?;
        Ok(frame)
    }

    #[test]
    fn colours_are_six_hex_digits() {
        assert_eq!("d0D0d0".parse(), Ok(Rgb(208, 208, 208)));
        assert_eq!("#0a141e".parse(), Ok(Rgb(10, 20, 30)));
        assert_eq!(Rgb(10, 20, 30).to_string(), "0a141e");
        for wrong in ["", "fff", "#fffff", "+fffff", "gggggg", "ffffff0"] {
            assert!(wrong.parse::<Rgb>().is_err(), "{wrong:?}");
        }
    }

    #[test]
    fn text_area_lies_between_the_markers() {
        // markers of 6 modules of 2 dots: 12 columns by 6 rows, behind a margin of 2 columns by 1 row
        let geometry = Geometry::new(80, 24, 6, &settings()).unwrap();
        let area = Cells {
            left: 14,
            top: 7,
            right: 66,
            bottom: 17,
        };
        assert_eq!(geometry.area, area);
        assert_eq!(geometry.text, area.inset(2, 1).unwrap());
        assert_eq!(geometry.marker_origins(), [(2, 2), (66, 2), (66, 34), (2, 34)]);
        assert_eq!(geometry.lines_per_page(0), 8);
        assert_eq!(geometry.lines_per_page(1), 4);
        assert_eq!(geometry.lines_per_page(2), 3);

        // a dark panel keeps a module's distance from the markers, and the text its padding from the panel
        let inverted = Settings {
            invert: true,
            ..settings()
        };
        let geometry = Geometry::new(80, 24, 6, &inverted).unwrap();
        assert_eq!(geometry.panel, area.inset(2, 1).unwrap());
        assert_eq!(geometry.text, area.inset(4, 2).unwrap());
    }

    #[test]
    fn markers_that_end_in_the_middle_of_a_row_leave_that_row_out() {
        let odd = Settings {
            module: 1,
            margin: 0,
            pad: 0,
            ..settings()
        };
        // markers of 7 dots: three and a half rows
        let geometry = Geometry::new(40, 12, 7, &odd).unwrap();
        assert_eq!((geometry.area.top, geometry.area.bottom), (4, 8));
        assert_eq!((geometry.area.left, geometry.area.right), (7, 33));
    }

    #[test]
    fn terminal_without_room_has_no_geometry() {
        assert!(Geometry::new(80, 24, 6, &settings()).is_some());
        // no row left between the markers
        assert!(Geometry::new(80, 14, 6, &settings()).is_none());
        // too few columns for text
        assert!(Geometry::new(39, 24, 6, &settings()).is_none());
        // smaller than the markers themselves
        assert!(Geometry::new(10, 4, 6, &settings()).is_none());
    }

    #[test]
    fn page_is_written_from_the_top_with_the_line_spacing() {
        let spaced = Settings {
            line_spacing: 1,
            ..settings()
        };
        let geometry = Geometry::new(80, 24, 6, &spaced).unwrap();
        let page = lines(&["first", "", "third", "fourth", "does not fit"]);
        let screen = chart(&geometry, &spaced, &page, &[]);
        let placed = |run: &Run| (run.column, run.row, run.text.clone());
        assert_eq!(
            screen.runs.iter().map(placed).collect::<Vec<_>>(),
            [
                (16, 8, "first".to_owned()),
                (16, 12, "third".to_owned()),
                (16, 14, "fourth".to_owned())
            ]
        );
    }

    #[test]
    fn notes_stay_out_of_the_text_area() {
        let settings = settings();
        let geometry = Geometry::new(80, 24, 6, &settings).unwrap();
        let screen = chart(&geometry, &settings, &[], &notes());
        let rows = screen.runs.iter().map(|run| run.row).collect::<Vec<_>>();
        // the area is the rows 7 to 16 and the row below it stays empty, which leaves six rows for seven
        // lines
        assert_eq!(rows, [18, 19, 20, 21, 22, 23]);
        for run in &screen.runs {
            // two modules away from the markers on both sides
            assert!(run.column >= 18 && run.column + run.text.len() <= 62, "{run:?}");
        }
        assert_eq!(screen.runs[0].text, "x".repeat(44));

        // two lines go into the middle of the six rows
        let screen = chart(&geometry, &settings, &[], &notes()[..2]);
        let rows = screen.runs.iter().map(|run| run.row).collect::<Vec<_>>();
        assert_eq!(rows, [20, 21]);
    }

    #[test]
    fn signal_marker_sits_between_the_top_markers() {
        let settings = settings();
        let geometry = Geometry::new(80, 24, 6, &settings).unwrap();
        // 12 dots wide like the others, in the middle of 80 columns and level with the top markers
        assert_eq!(geometry.signal_origin(), (34, 2));
        let screen = chart(&geometry, &settings, &[], &[]);
        let ink_in_row = |y: usize| {
            (0..80)
                .filter(|&x| screen.dot(x, y) == Rgb::BLACK)
                .collect::<Vec<_>>()
        };
        // the top borders of three markers: columns 2 to 13, 34 to 45 and 66 to 77
        let borders = (2..14).chain(34..46).chain(66..78).collect::<Vec<_>>();
        assert_eq!(ink_in_row(2), borders);
        assert_eq!(ink_in_row(3), borders);
        assert_eq!(ink_in_row(1), [] as [usize; 0]);

        // the markers have to leave it room, with a module of paper on either side
        assert!(Geometry::new(44, 24, 6, &settings).is_some());
        assert!(Geometry::new(43, 24, 6, &settings).is_none());
    }

    #[test]
    fn notes_are_left_out_where_the_markers_leave_no_room() {
        let small = Settings {
            module: 1,
            margin: 0,
            ..settings()
        };
        // markers of three rows leave two rows for notes, below the row that is kept empty
        let geometry = Geometry::new(60, 20, 6, &small).unwrap();
        let screen = chart(&geometry, &small, &[], &notes());
        let rows = screen.runs.iter().map(|run| run.row).collect::<Vec<_>>();
        assert_eq!(rows, [18, 19]);
    }

    /// The corners of the text area in an image of the screen with cells of `cell` pixels.
    fn area_in_pixels(geometry: &Geometry, cell: (i32, i32)) -> [(f32, f32); 4] {
        let edge = (geometry.margin + geometry.marker) as f32;
        let (dot_width, dot_height) = (cell.0 as f32, cell.1 as f32 / 2.0);
        let (left, top) = (edge * dot_width, edge * dot_height);
        let right = (geometry.columns as f32 - edge) * dot_width;
        let bottom = (2.0 * geometry.rows as f32 - edge) * dot_height;
        [(left, top), (right, top), (right, bottom), (left, bottom)]
    }

    #[test]
    fn locator_finds_the_text_area_on_the_screen() -> Result<()> {
        struct Case {
            dictionary: &'static str,
            ids: [i32; 4],
            module: usize,
            margin: usize,
            invert: bool,
            cell: (i32, i32),
        }
        let case = |dictionary, ids, module, margin, invert, cell| Case {
            dictionary,
            ids,
            module,
            margin,
            invert,
            cell,
        };
        let cases = [
            case("4x4_50", [0, 1, 2, 3], 2, 1, false, (10, 20)),
            // the smallest markers there are: 6 columns by 3 rows
            case("4x4_50", [3, 2, 1, 0], 1, 1, false, (10, 20)),
            // markers of 21 dots end in the middle of a row
            case("5x5_100", [10, 20, 30, 40], 3, 2, false, (9, 18)),
            // cells that are not twice as high as wide make oblong markers
            case("apriltag_36h11", [5, 6, 7, 8], 2, 1, false, (8, 20)),
            case("6x6_250", [100, 7, 31, 249], 2, 1, false, (12, 20)),
            // a dark panel
            case("4x4_50", [0, 1, 2, 3], 2, 1, true, (10, 20)),
            case("4x4_50", [0, 1, 2, 3], 3, 1, true, (10, 20)),
        ];
        // every chart says something else about itself
        let signals = [
            Signal::Ready,
            Signal::Page(0),
            Signal::Page(7),
            Signal::Page(15),
            Signal::Done,
            Signal::Page(1),
            Signal::Page(2),
        ];
        for (case, signal) in cases.into_iter().zip(signals) {
            let Case {
                dictionary,
                ids,
                module,
                margin,
                invert,
                cell,
            } = case;
            let settings = Settings {
                module,
                margin,
                invert,
                ..settings()
            };
            let markers = patterns(dictionary, ids);
            let geometry = Geometry::new(120, 40, markers[0].side(), &settings).unwrap();
            let page = lines(&["Some text", "to look past"]);
            let fifth = signal_pattern(dictionary, ids, signal);
            let chart = compose(&geometry, &settings, &markers, &fifth, &page, &notes());
            let screen = rasterize(&chart, cell)?;

            let found = Locator::new(dictionary, ids)?.locate(&screen)?;
            let label = format!("{dictionary} module {module} invert {invert}");
            let sighting = found.unwrap_or_else(|| panic!("{label}: markers not found"));
            assert_eq!(sighting.signal, Some(signal), "{label}");
            for (corner, (x, y)) in sighting.area.iter().zip(area_in_pixels(&geometry, cell)) {
                // the detector reports the middle of the pixels along a marker's outline
                assert!(
                    (corner.x - x).abs() <= 1.5 && (corner.y - y).abs() <= 1.5,
                    "{label}: corner {corner:?} is not at ({x}, {y})"
                );
            }
        }
        Ok(())
    }

    /// Top-left, top-right, bottom-right, bottom-left
    const IDS: [i32; 4] = [7, 3, 11, 5];

    /// A camera frame of a chart on a terminal of 80 by 24 cells that says `signal` and shows `page`.
    fn frame(settings: &Settings, signal: Signal, page: &[&str]) -> Result<Mat> {
        let markers = patterns("4x4_50", IDS);
        let geometry = Geometry::new(80, 24, markers[0].side(), settings).unwrap();
        let notes = [
            "page 1/1 | module 2 | margin 1 | pad 1".to_owned(),
            "space/n next page   q quit".to_owned(),
        ];
        let fifth = signal_pattern("4x4_50", IDS, signal);
        let chart = compose(&geometry, settings, &markers, &fifth, &lines(page), &notes);
        photograph(&rasterize(&chart, (24, 48))?)
    }

    /// A pipeline for those frames, that takes a text once `stable_frames` frames agree on it.
    fn pipeline(stable_frames: usize) -> Result<Pipeline> {
        Pipeline::new(Options {
            device: Device::Cpu,
            dictionary: "4x4_50".to_owned(),
            marker_ids: IDS,
            page_size: None,
            stable_frames,
            min_similarity: 0.9,
            models: Models::default(),
        })
    }

    const PAGE: [&str; 3] = [
        "Invoice 2026-0417",
        "Total due: 1,284.50 USD",
        "Thank you for your order!",
    ];

    fn read_photograph(settings: &Settings) -> Result<Outcome> {
        pipeline(1)?.process(&frame(settings, Signal::Page(0), &PAGE)?)
    }

    const PAGE_TEXT: &str = "Invoice 2026-0417\nTotal due: 1,284.50 USD\nThank you for your order!";

    #[test]
    fn pipeline_reads_the_page_off_a_photographed_screen() -> Result<()> {
        assert_eq!(read_photograph(&settings())?, Outcome::Text(PAGE_TEXT.to_owned()));
        Ok(())
    }

    #[test]
    fn pipeline_follows_the_screens_of_a_show() -> Result<()> {
        let settings = settings();
        let ready = frame(&settings, Signal::Ready, &["Enter starts."])?;
        let first = frame(&settings, Signal::Page(0), &PAGE)?;
        let second = frame(&settings, Signal::Page(1), &["Second page", "delta epsilon"])?;
        // the same text as on the second page: what the chart says tells the two apart
        let third = frame(&settings, Signal::Page(2), &["Second page", "delta epsilon"])?;
        let done = frame(&settings, Signal::Done, &["End."])?;

        let mut pipeline = pipeline(3)?;
        let frames = [
            &ready, &ready, &first, &first, &first, &first, &second, &second, &second, &third, &third, &done,
            &done,
        ];
        let mut outcomes = Vec::new();
        for frame in frames {
            outcomes.push(pipeline.process(frame)?);
        }
        let second_text = "Second page\ndelta epsilon";
        assert_eq!(
            outcomes,
            [
                Outcome::Ready,
                Outcome::Ready,
                Outcome::Unsettled,
                Outcome::Unsettled,
                Outcome::Text(PAGE_TEXT.to_owned()),
                // read already: this frame is not even looked at
                Outcome::Duplicate,
                Outcome::Unsettled,
                Outcome::Unsettled,
                Outcome::Text(second_text.to_owned()),
                Outcome::Unsettled,
                Outcome::Unsettled,
                // the third page went away after two frames, which is one too few to be sure of it
                Outcome::Guess(second_text.to_owned()),
                Outcome::Done,
            ]
        );
        assert_eq!(pipeline.finish(), None);
        Ok(())
    }

    #[test]
    fn pages_that_go_missing_are_counted() -> Result<()> {
        use ocrs::Missed;
        let settings = settings();
        let ready = frame(&settings, Signal::Ready, &["Enter starts."])?;
        let done = frame(&settings, Signal::Done, &["End."])?;
        let page = |number: usize| frame(&settings, Signal::Page(number), &PAGE);
        let blank = |number: usize| frame(&settings, Signal::Page(number), &[]);

        let mut pipeline = pipeline(2)?;
        let mut missed = Vec::new();
        let mut see = |frame: &Mat| -> Result<Outcome> {
            let outcome = pipeline.process(frame)?;
            missed.push((pipeline.missed().unseen, pipeline.missed().unread));
            Ok(outcome)
        };
        see(&ready)?;
        // the first page is 0: starting with 2 leaves two out
        assert_eq!(see(&page(2)?)?, Outcome::Unsettled);
        assert_eq!(see(&page(2)?)?, Outcome::Text(PAGE_TEXT.to_owned()));
        // a page of which nothing can be read, which takes two frames to be sure of as well
        assert_eq!(see(&blank(3)?)?, Outcome::Unsettled);
        assert_eq!(see(&blank(3)?)?, Outcome::NoText);
        // from 3 to 5 is one left out, and the page before was not read
        assert_eq!(see(&page(5)?)?, Outcome::Unsettled);
        // from 5 over 15 to 0 are ten left out; one frame of page 5 was read, so it is guessed, not missed
        assert_eq!(see(&page(0)?)?, Outcome::Guess(PAGE_TEXT.to_owned()));
        assert_eq!(see(&page(0)?)?, Outcome::Unsettled);
        assert_eq!(see(&page(0)?)?, Outcome::Text(PAGE_TEXT.to_owned()));
        assert_eq!(see(&done)?, Outcome::Done);
        assert_eq!(
            missed,
            [
                (0, 0),
                (2, 0),
                (2, 0),
                (2, 0),
                (2, 0),
                (3, 1),
                (13, 1),
                (13, 1),
                (13, 1),
                (13, 1)
            ]
        );
        assert_eq!(pipeline.finish(), None);
        assert_eq!(
            pipeline.missed(),
            Missed {
                unseen: 13,
                unread: 1
            }
        );
        Ok(())
    }

    #[test]
    fn page_on_show_to_the_end_is_guessed_when_the_frames_run_out() -> Result<()> {
        let settings = settings();
        let first = frame(&settings, Signal::Page(4), &PAGE)?;
        let mut pipeline = pipeline(3)?;
        assert_eq!(pipeline.process(&first)?, Outcome::Unsettled);
        assert_eq!(pipeline.process(&first)?, Outcome::Unsettled);
        assert_eq!(pipeline.finish(), Some(PAGE_TEXT.to_owned()));
        assert_eq!(pipeline.finish(), None);
        Ok(())
    }

    #[test]
    fn frame_between_two_screens_is_passed_over() -> Result<()> {
        let settings = settings();
        let first = frame(&settings, Signal::Page(0), &PAGE)?;
        let second = frame(&settings, Signal::Page(1), &["Second page", "delta epsilon"])?;
        let mut between = Mat::default();
        core::add_weighted_def(&first, 0.5, &second, 0.5, 0.0, &mut between)?;

        let mut pipeline = pipeline(2)?;
        let mut outcomes = Vec::new();
        for frame in [&first, &first, &between, &second, &second] {
            outcomes.push(pipeline.process(frame)?);
        }
        let texts = outcomes
            .iter()
            .filter(|outcome| matches!(outcome, Outcome::Text(_) | Outcome::Guess(_)));
        assert_eq!(
            texts.collect::<Vec<_>>(),
            [
                &Outcome::Text(PAGE_TEXT.to_owned()),
                &Outcome::Text("Second page\ndelta epsilon".to_owned())
            ],
            "{outcomes:?}"
        );
        Ok(())
    }

    #[test]
    fn pipeline_reads_light_text_on_a_dark_panel() -> Result<()> {
        let inverted = Settings {
            invert: true,
            ..settings()
        };
        assert_eq!(read_photograph(&inverted)?, Outcome::Text(PAGE_TEXT.to_owned()));
        Ok(())
    }
}
