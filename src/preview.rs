//! A live view for pointing the camera, part of the `ocrs` program only: what the camera sees, drawn on the
//! terminal out of half block characters, with the markers that are made out in it.

use std::io::{self, IsTerminal, Write};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::style::{Color, Colors, Print, ResetColor, SetColors, SetForegroundColor};
use crossterm::terminal::{
    self, BeginSynchronizedUpdate, Clear, ClearType, DisableLineWrap, EnableLineWrap, EndSynchronizedUpdate,
    EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::{execute, queue};
use opencv::core::{Mat, Point, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;

use ocrs::markers::{Outline, Signal, Survey};

/// Rows below the picture: the markers, the advice, the facts and the latest remark.
const TEXT_ROWS: usize = 4;

/// Colours of the marks in the picture, as blue, green, red.
const CORNER: (u8, u8, u8) = (0, 230, 0);
const SIGNAL: (u8, u8, u8) = (255, 220, 0);
const STRANGER: (u8, u8, u8) = (255, 0, 255);
const UNREAD: (u8, u8, u8) = (0, 0, 255);

/// The terminal as a view of the camera, from [`Preview::open`] until this is dropped.
pub struct Preview {
    out: io::Stdout,
}

impl Preview {
    pub fn open() -> Result<Self> {
        let out = io::stdout();
        ensure!(
            out.is_terminal(),
            "--preview draws on the terminal, so the output can't be redirected"
        );
        terminal::enable_raw_mode().context("can't take over the terminal")?;
        // from here on `Drop` undoes what was done, also when the next step fails
        let mut preview = Self { out };
        execute!(preview.out, EnterAlternateScreen, DisableLineWrap, Hide)?;
        Ok(preview)
    }

    /// Shows a frame with what `survey` made out in it. `facts` and `remark` are lines of text to put
    /// below it.
    pub fn draw(&mut self, frame: &Mat, survey: &Survey, facts: &str, remark: &str) -> Result<()> {
        let window = terminal::window_size().context("can't ask the terminal for its size")?;
        let (columns, rows) = (window.columns as usize, window.rows as usize);
        let survey = worth_showing(survey, frame.size()?);
        let mut out = Vec::new();
        queue!(out, BeginSynchronizedUpdate, ResetColor, Clear(ClearType::All))?;
        let picture_rows = rows.saturating_sub(TEXT_ROWS);
        if columns > 0 && picture_rows > 0 {
            // a dot is one column wide and half a row high, and about square unless the terminal says so
            let dot_shape = if window.width > 0 && window.height > 0 {
                (window.width as f64 / columns as f64) / (window.height as f64 / rows as f64 / 2.0)
            } else {
                1.0
            };
            let size = picture_size(frame.size()?, columns, picture_rows, dot_shape);
            let picture = picture(frame, &survey, size)?;
            let left = (columns - size.width as usize) / 2;
            encode_picture(&picture, left as u16, &mut out)?;
        }
        for (line, row) in status(&survey, facts, remark).iter().zip(picture_rows..rows) {
            queue!(out, MoveTo(0, row as u16), ResetColor)?;
            let mut room = columns;
            for (colour, text) in line {
                let text = text.chars().take(room).collect::<String>();
                room -= text.chars().count();
                queue!(out, SetForegroundColor(*colour), Print(text))?;
            }
        }
        queue!(out, ResetColor, EndSynchronizedUpdate)?;
        self.out.write_all(&out)?;
        self.out.flush()?;
        Ok(())
    }

    /// Whether q, escape or control-c was pressed since the last call.
    pub fn quit_requested(&mut self) -> Result<bool> {
        let mut quit = false;
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()?
                && key.kind != KeyEventKind::Release
            {
                let interrupt =
                    key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
                quit |= interrupt || matches!(key.code, KeyCode::Char('q') | KeyCode::Esc);
            }
        }
        Ok(quit)
    }
}

impl Drop for Preview {
    fn drop(&mut self) {
        // nothing more can be done about a terminal that doesn't take these
        let _ = execute!(self.out, ResetColor, EnableLineWrap, Show, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

/// The survey without the unread squares that can hardly be markers: a detector that looks for dark squares
/// finds plenty of them in text and clutter, small ones and oblong ones, and marking all of those would
/// hide the few that say something. A marker that is worth pointing a camera at is at least a fiftieth of
/// the frame across, and no more than twice as long as wide even when seen at an angle.
fn worth_showing(survey: &Survey, frame: Size) -> Survey {
    let smallest = frame.width.max(frame.height) as f32 / 50.0;
    let like_a_marker = |outline: &Outline| {
        let sides = [0, 1, 2, 3].map(|corner| {
            let (from, to) = (outline[corner], outline[(corner + 1) % 4]);
            (to.x - from.x).hypot(to.y - from.y)
        });
        let shortest = sides.into_iter().fold(f32::INFINITY, f32::min);
        let longest = sides.into_iter().fold(0.0, f32::max);
        shortest >= smallest && longest <= 2.0 * shortest
    };
    Survey {
        unread: survey.unread.iter().copied().filter(like_a_marker).collect(),
        ..survey.clone()
    }
}

/// How many dots (width and height) a frame is shown in: as large as `columns` by `rows` cells allow at
/// two dots a cell, in the shape of the frame. `dot_shape` is the width of a dot over its height.
fn picture_size(frame: Size, columns: usize, rows: usize, dot_shape: f64) -> Size {
    let (widest, highest) = (columns as f64, 2.0 * rows as f64);
    // dots across for every dot down
    let across = frame.width as f64 / frame.height as f64 / dot_shape;
    let height = highest.min(widest / across);
    // whole cells
    let height = ((height / 2.0).floor() * 2.0).max(2.0);
    let width = (height * across).round().clamp(1.0, widest);
    Size::new(width as i32, height as i32)
}

/// The frame at one pixel a dot, with the outlines of what the survey made out: the text area and its
/// corner markers, the signal marker, markers with other ids, and squares that can't be read.
fn picture(frame: &Mat, survey: &Survey, size: Size) -> Result<Mat> {
    let mut picture = Mat::default();
    imgproc::resize(frame, &mut picture, size, 0.0, 0.0, imgproc::INTER_AREA)?;
    let frame = frame.size()?;
    let scale = (
        size.width as f32 / frame.width as f32,
        size.height as f32 / frame.height as f32,
    );
    let mut mark = |outline: &Outline, (blue, green, red): (u8, u8, u8)| -> Result<()> {
        let colour = Scalar::new(blue as f64, green as f64, red as f64, 0.0);
        for (from, to) in outline.iter().zip(outline.iter().cycle().skip(1)) {
            let [from, to] = [from, to].map(|point| {
                Point::new(
                    (point.x * scale.0).floor() as i32,
                    (point.y * scale.1).floor() as i32,
                )
            });
            imgproc::line(&mut picture, from, to, colour, 1, imgproc::LINE_8, 0)?;
        }
        Ok(())
    };
    // what matters most goes on last
    for outline in &survey.unread {
        mark(outline, UNREAD)?;
    }
    for (_, outline) in &survey.strangers {
        mark(outline, STRANGER)?;
    }
    if let Some(sighting) = survey.sighting() {
        mark(&sighting.area, CORNER)?;
    }
    for outline in survey.corners.iter().flatten() {
        mark(outline, CORNER)?;
    }
    if let Some((_, outline)) = &survey.signal {
        mark(outline, SIGNAL)?;
    }
    Ok(picture)
}

/// The escape sequences that draw `picture` from the top of the terminal, `left` columns in: two rows of
/// pixels to a row of cells, as an upper half block in the colour of the upper pixel on the colour of the
/// lower one.
fn encode_picture(picture: &Mat, left: u16, out: &mut impl Write) -> Result<()> {
    let (width, height) = (picture.cols() as usize, picture.rows() as usize);
    ensure!(
        picture.channels() == 3 && picture.is_continuous(),
        "a frame is expected in three channels"
    );
    let bytes = picture.data_bytes()?;
    let pixel = |column: usize, row: usize| {
        let at = (row * width + column) * 3;
        Color::Rgb {
            r: bytes[at + 2],
            g: bytes[at + 1],
            b: bytes[at],
        }
    };
    for row in 0..height / 2 {
        queue!(out, MoveTo(left, row as u16))?;
        let mut colours = None;
        for column in 0..width {
            let pair = (pixel(column, 2 * row), pixel(column, 2 * row + 1));
            if colours != Some(pair) {
                queue!(out, SetColors(Colors::new(pair.0, pair.1)))?;
                colours = Some(pair);
            }
            queue!(out, Print('▀'))?;
        }
    }
    Ok(())
}

fn colour((blue, green, red): (u8, u8, u8)) -> Color {
    Color::Rgb {
        r: red,
        g: green,
        b: blue,
    }
}

/// The lines below the picture, as pieces of text with their colours: which markers are found, what to do
/// about it, then `facts` and `remark`.
fn status(survey: &Survey, facts: &str, remark: &str) -> Vec<Vec<(Color, String)>> {
    let plain = Color::Reset;
    let missing = Color::Rgb { r: 255, g: 90, b: 90 };
    let mut markers = Vec::new();
    for (name, found) in ["TL", "TR", "BR", "BL"].iter().zip(&survey.corners) {
        let (shade, state) = if found.is_some() {
            (colour(CORNER), "found")
        } else {
            (missing, "-----")
        };
        markers.push((shade, format!("{name} {state}   ")));
    }
    let signal = match survey.signal {
        Some((Signal::Ready, _)) => Some("start screen".to_owned()),
        Some((Signal::Page(number), _)) => Some(format!("page, number {number}")),
        Some((Signal::Done, _)) => Some("end screen".to_owned()),
        None => None,
    };
    markers.push(match signal {
        Some(signal) => (colour(SIGNAL), format!("signal: {signal}")),
        None => (plain, "signal: none".to_owned()),
    });
    let legend = [
        (plain, "q quits   marks: ".to_owned()),
        (colour(CORNER), "text area and its markers  ".to_owned()),
        (colour(SIGNAL), "signal  ".to_owned()),
        (colour(STRANGER), "other id  ".to_owned()),
        (colour(UNREAD), "square that can't be read".to_owned()),
    ];
    let remark = if remark.is_empty() {
        legend.to_vec()
    } else {
        vec![(plain, remark.to_owned())]
    };
    vec![
        markers,
        vec![(plain, advice(survey))],
        vec![(plain, facts.to_owned())],
        remark,
    ]
}

/// What to do about what is in view.
fn advice(survey: &Survey) -> String {
    let found = survey.corners.iter().flatten().count();
    match (found, survey.signal) {
        (4, Some((Signal::Ready, _))) => {
            "start screen in view: enter in ocrs-show starts the pages".to_owned()
        }
        (4, Some((Signal::Page(_), _))) => "a page in view".to_owned(),
        (4, Some((Signal::Done, _))) => "end screen in view".to_owned(),
        (4, None) => "text area in view".to_owned(),
        (0, _) if !survey.strangers.is_empty() => {
            let ids = survey
                .strangers
                .iter()
                .map(|(id, _)| id.to_string())
                .collect::<Vec<_>>();
            format!(
                "markers with other ids in view ({}): compare --dict and --ids",
                ids.join(", ")
            )
        }
        (0, _) if survey.unread.is_empty() => "no markers in view".to_owned(),
        (found, _) => {
            let unread = if survey.unread.is_empty() {
                ""
            } else {
                "; the squares marked red may be markers that are too small, blurred or outshone"
            };
            format!("{found} of the 4 corner markers in view{unread}")
        }
    }
}

#[cfg(test)]
mod tests {
    use opencv::core::{CV_8UC3, Point2f, Vec3b};

    use super::*;

    /// A square marker outline from `(x, y)`.
    fn square(x: f32, y: f32, side: f32) -> Outline {
        [(x, y), (x + side, y), (x + side, y + side), (x, y + side)].map(|(x, y)| Point2f::new(x, y))
    }

    fn all_corners() -> [Option<Outline>; 4] {
        [
            Some(square(100.0, 100.0, 100.0)),
            Some(square(1000.0, 100.0, 100.0)),
            Some(square(1000.0, 700.0, 100.0)),
            Some(square(100.0, 700.0, 100.0)),
        ]
    }

    #[test]
    fn picture_fills_the_terminal_in_the_shape_of_the_frame() {
        let camera = Size::new(1280, 960);
        // 36 rows are 72 dots down, which makes 96 across for 4:3
        assert_eq!(picture_size(camera, 120, 36, 1.0), Size::new(96, 72));
        // a narrow terminal limits the width instead: 60 across, 45 down, of which 44 are whole cells
        assert_eq!(picture_size(camera, 60, 36, 1.0), Size::new(59, 44));
        // dots that are narrower than high need more of them across
        assert_eq!(picture_size(camera, 120, 36, 0.8), Size::new(120, 72));
        assert_eq!(
            picture_size(Size::new(1280, 720), 200, 50, 1.0),
            Size::new(178, 100)
        );
        // never nothing
        assert_eq!(picture_size(camera, 1, 1, 1.0), Size::new(1, 2));
    }

    #[test]
    fn markers_are_outlined_in_the_picture() -> Result<()> {
        let frame = Mat::new_rows_cols_with_default(960, 1280, CV_8UC3, Scalar::all(128.0))?;
        let survey = Survey {
            corners: all_corners(),
            signal: Some((Signal::Ready, square(600.0, 100.0, 100.0))),
            strangers: vec![(40, square(300.0, 400.0, 100.0))],
            unread: vec![square(700.0, 400.0, 100.0)],
        };
        // a tenth of the size
        let picture = picture(&frame, &survey, Size::new(128, 96))?;
        assert_eq!(picture.size()?, Size::new(128, 96));
        let at = |x: i32, y: i32| -> (u8, u8, u8) {
            let pixel = picture.at_2d::<Vec3b>(y, x).unwrap();
            (pixel[0], pixel[1], pixel[2])
        };
        let grey = (128, 128, 128);
        // the top-left marker, from (10, 10) to (20, 20), and nothing inside it
        assert_eq!([at(10, 10), at(15, 10), at(20, 15), at(10, 20)], [CORNER; 4]);
        assert_eq!(at(15, 15), grey);
        assert_eq!(at(110, 80), CORNER);
        // the text area between the corners that face the middle
        assert_eq!([at(40, 20), at(20, 45), at(100, 45), at(60, 70)], [CORNER; 4]);
        // where the signal marker stands on that line, as on a chart, its outline is on top
        assert_eq!(at(65, 20), SIGNAL);
        assert_eq!(at(60, 45), grey);
        assert_eq!(at(60, 10), SIGNAL);
        assert_eq!(at(30, 40), STRANGER);
        assert_eq!(at(70, 40), UNREAD);
        Ok(())
    }

    #[test]
    fn only_unread_squares_that_could_be_markers_are_shown() {
        let oblong = [(0.0, 0.0), (90.0, 0.0), (90.0, 30.0), (0.0, 30.0)].map(|(x, y)| Point2f::new(x, y));
        // a marker seen at an angle is a trapezium, but not that oblong
        let slanted = [(0.0, 0.0), (60.0, 5.0), (58.0, 50.0), (2.0, 60.0)].map(|(x, y)| Point2f::new(x, y));
        let survey = Survey {
            corners: all_corners(),
            unread: vec![
                square(10.0, 10.0, 40.0),
                square(500.0, 10.0, 20.0),
                oblong,
                slanted,
            ],
            ..Survey::default()
        };
        // a fiftieth of 1280 is 25.6
        let shown = worth_showing(&survey, Size::new(1280, 960));
        assert_eq!(shown.unread, [square(10.0, 10.0, 40.0), slanted]);
        assert_eq!(shown.corners, survey.corners);
    }

    #[test]
    fn picture_becomes_half_blocks() -> Result<()> {
        // two columns by two rows of cells; blue, green, red
        let mut picture = Mat::new_rows_cols_with_default(4, 2, CV_8UC3, Scalar::new(1.0, 2.0, 3.0, 0.0))?;
        *picture.at_2d_mut::<Vec3b>(1, 0)? = Vec3b::from([10, 20, 30]);
        let mut out = Vec::new();
        encode_picture(&picture, 7, &mut out)?;
        assert_eq!(
            String::from_utf8(out)?,
            "\x1b[1;8H\x1b[38;2;3;2;1;48;2;30;20;10m▀\x1b[38;2;3;2;1;48;2;3;2;1m▀\
             \x1b[2;8H\x1b[38;2;3;2;1;48;2;3;2;1m▀▀"
        );
        Ok(())
    }

    fn texts(lines: &[Vec<(Color, String)>]) -> Vec<String> {
        let text = |line: &Vec<(Color, String)>| line.iter().map(|(_, text)| text.as_str()).collect();
        lines.iter().map(text).collect()
    }

    #[test]
    fn status_names_the_markers_and_what_to_do() {
        let nothing = Survey::default();
        let lines = texts(&status(&nothing, "camera 1280x960", ""));
        assert_eq!(
            lines[0],
            "TL -----   TR -----   BR -----   BL -----   signal: none"
        );
        assert_eq!(lines[1], "no markers in view");
        assert_eq!(lines[2], "camera 1280x960");
        assert!(
            lines[3].starts_with("q quits   marks: text area and its markers"),
            "{}",
            lines[3]
        );

        let mut some = Survey {
            corners: all_corners(),
            unread: vec![square(0.0, 0.0, 10.0)],
            ..Survey::default()
        };
        some.corners[2] = None;
        let lines = texts(&status(&some, "", "frame 12: 20 lines"));
        assert_eq!(
            lines[0],
            "TL found   TR found   BR -----   BL found   signal: none"
        );
        assert_eq!(
            lines[1],
            "3 of the 4 corner markers in view; the squares marked red may be markers that are too small, \
             blurred or outshone"
        );
        assert_eq!(lines[3], "frame 12: 20 lines");
        // found markers and missing ones differ in colour
        let colours = status(&some, "", "")[0]
            .iter()
            .map(|(colour, _)| *colour)
            .collect::<Vec<_>>();
        assert_eq!(colours[0], colours[1]);
        assert_ne!(colours[1], colours[2]);

        let strangers = Survey {
            strangers: vec![(17, square(0.0, 0.0, 10.0)), (23, square(50.0, 0.0, 10.0))],
            ..Survey::default()
        };
        assert_eq!(
            advice(&strangers),
            "markers with other ids in view (17, 23): compare --dict and --ids"
        );
    }

    #[test]
    fn advice_follows_the_screens_of_a_show() {
        let with = |signal| Survey {
            corners: all_corners(),
            signal,
            ..Survey::default()
        };
        let marker = square(600.0, 100.0, 100.0);
        assert_eq!(advice(&with(None)), "text area in view");
        assert_eq!(
            advice(&with(Some((Signal::Ready, marker)))),
            "start screen in view: enter in ocrs-show starts the pages"
        );
        assert_eq!(advice(&with(Some((Signal::Page(3), marker)))), "a page in view");
        assert_eq!(advice(&with(Some((Signal::Done, marker)))), "end screen in view");
        let lines = texts(&status(&with(Some((Signal::Page(3), marker))), "", ""));
        assert_eq!(
            lines[0],
            "TL found   TR found   BR found   BL found   signal: page, number 3"
        );
    }
}
