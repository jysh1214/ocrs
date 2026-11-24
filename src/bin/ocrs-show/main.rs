//! `ocrs-show` puts a text file between four markers on the terminal, as something to point a camera at
//! and have `ocrs` read back.

mod chart;
mod terminal;
mod text;

use std::borrow::Cow;
use std::fs;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use clap::builder::RangedU64ValueParser;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use ocrs::markers::{self, PAGE_NUMBERS, Pattern, Signal, Signals};

use chart::{Geometry, Rgb, Screen, Settings};
use terminal::{Size, Terminal};
use text::{Line, Page, Wrap};

const MODULE: RangeInclusive<usize> = 1..=16;
const MARGIN: RangeInclusive<usize> = 0..=8;
const PAD: RangeInclusive<usize> = 0..=16;
const LINE_SPACING: RangeInclusive<usize> = 0..=8;

const DEFAULT_DICTIONARY: &str = "4x4_50";
const DEFAULT_IDS: [i32; 4] = [0, 1, 2, 3];

const KEYS: &str = "\
Keys:
  enter, space, n, right, down    next page; from the start screen to the first page
  p, left, up                     previous page
  g, G                            first, last page
  + -                             larger, smaller markers
  ] [                             more, less padding around the text
  } {                             more, less margin around the markers
  . ,                             more, fewer empty rows between the lines
  i                               light text on a dark panel, and back
  b                               bold text, and back
  w                               wrap or cut off long lines
  h                               hide or show the notes below the text
  q, esc                          quit, and print the options for the layout as it was then

The text has the size of the terminal's font: change that in the terminal and the layout follows.";

/// Shows a text file between four ArUco / AprilTag markers on the terminal, as a target to point a camera
/// at: `ocrs` reads the text between the markers back. It starts on a start screen and ends on an end
/// screen, and a fifth marker tells `ocrs` which screen or page is on show.
#[derive(Parser)]
#[command(version, after_help = KEYS)]
struct Cli {
    /// Text file to show; a form feed character in it starts a new page
    text: PathBuf,

    /// Marker dictionary, an OpenCV predefined one such as 4x4_50, 6x6_250 or apriltag_36h11
    #[arg(long, default_value = DEFAULT_DICTIONARY)]
    dict: String,

    /// Ids of the markers at the top-left, top-right, bottom-right and bottom-left corner of the text
    #[arg(long, value_parser = markers::parse_ids, default_value = "0,1,2,3", value_name = "TL,TR,BR,BL")]
    ids: [i32; 4],

    /// Size of a marker module: this many columns wide and half as many rows high
    #[arg(long, default_value_t = 2, value_parser = within(MODULE), value_name = "COLUMNS")]
    module: usize,

    /// Blank space between the edges of the terminal and the markers, in modules
    #[arg(long, default_value_t = 1, value_parser = within(MARGIN), value_name = "MODULES")]
    margin: usize,

    /// Blank space around the text: this many rows, and twice as many columns
    #[arg(long, default_value_t = 1, value_parser = within(PAD), value_name = "ROWS")]
    pad: usize,

    /// Empty rows between two lines of text
    #[arg(long, default_value_t = 0, value_parser = within(LINE_SPACING), value_name = "ROWS")]
    line_spacing: usize,

    /// What to do with lines that are wider than the text area
    #[arg(long, value_enum, default_value_t = Wrap::Word)]
    wrap: Wrap,

    /// Light text on a dark panel instead of dark text on a light background
    #[arg(long)]
    invert: bool,

    /// Bold text
    #[arg(long)]
    bold: bool,

    /// The dark colour, of the markers and the text
    #[arg(long, default_value_t = Rgb::BLACK, value_name = "RRGGBB")]
    ink: Rgb,

    /// The light colour, of the background; a grey such as c0c0c0 dims a screen that glares
    #[arg(long, default_value_t = Rgb::WHITE, value_name = "RRGGBB")]
    paper: Rgb,

    /// Turn to the next page by itself every so many seconds, from the first page on to the end screen
    #[arg(long, value_name = "SECONDS")]
    interval: Option<f64>,

    /// Start without the notes below the text area: the keys and the measurements
    #[arg(long)]
    no_notes: bool,
}

fn within(range: RangeInclusive<usize>) -> RangedU64ValueParser<usize> {
    RangedU64ValueParser::new().range(*range.start() as u64..=*range.end() as u64)
}

/// What a key press calls for.
#[derive(Debug, PartialEq, Eq)]
enum Reaction {
    Quit,
    Redraw,
    Nothing,
}

/// What is on show.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// The start screen, until a key brings the first page
    Ready,
    /// The page [`App::page`]
    Pages,
    /// The end screen, after the last page
    Done,
}

/// The markers that tell `ocrs` what is on show, see [`Signal`].
struct SignalMarkers {
    ready: Pattern,
    done: Pattern,
    /// By page number
    pages: Vec<Pattern>,
}

impl SignalMarkers {
    fn new(dictionary: &str, corner_ids: [i32; 4]) -> Result<Self> {
        let signals = Signals::new(dictionary, corner_ids)?;
        let pattern = |signal| markers::pattern(dictionary, signals.id(signal));
        Ok(Self {
            ready: pattern(Signal::Ready)?,
            done: pattern(Signal::Done)?,
            pages: (0..PAGE_NUMBERS)
                .map(|number| pattern(Signal::Page(number)))
                .collect::<Result<_>>()?,
        })
    }
}

struct App {
    file: PathBuf,
    dictionary: String,
    ids: [i32; 4],
    interval: Option<f64>,
    text: String,
    /// The markers of `ids`
    markers: [Pattern; 4],
    signals: SignalMarkers,
    settings: Settings,
    wrap: Wrap,
    /// Whether the keys and the measurements are shown
    notes: bool,
    /// Where in the text the page on show begins, see [`text::Line::at`]. The page is found again by this
    /// when a new layout puts different lines on every page.
    anchor: (usize, usize),
    /// The layout for the terminal as it was when it was last drawn on; `None` if it was too small
    geometry: Option<Geometry>,
    pages: Vec<Page>,
    stage: Stage,
    /// Index of the page that is on show, or will be when the pages are
    page: usize,
}

impl App {
    fn new(cli: Cli) -> Result<Self> {
        let text = fs::read_to_string(&cli.text)
            .with_context(|| format!("can't read {} as UTF-8 text", cli.text.display()))?;
        let mut patterns = Vec::with_capacity(cli.ids.len());
        for id in cli.ids {
            patterns.push(markers::pattern(&cli.dict, id)?);
        }
        ensure!(
            cli.ink.luma() < cli.paper.luma(),
            "--ink must be darker than --paper, markers are only found dark on light (--invert swaps the \
             colours of the text alone)"
        );
        if let Some(seconds) = cli.interval {
            ensure!(
                seconds.is_finite() && seconds > 0.0 && Duration::try_from_secs_f64(seconds).is_ok(),
                "--interval must be a number of seconds above 0"
            );
        }
        Ok(Self {
            signals: SignalMarkers::new(&cli.dict, cli.ids)?,
            file: cli.text,
            dictionary: cli.dict,
            ids: cli.ids,
            interval: cli.interval,
            text: text.trim_start_matches('\u{feff}').to_owned(),
            markers: patterns.try_into().expect("a pattern for each of the four ids"),
            settings: Settings {
                module: cli.module,
                margin: cli.margin,
                pad: cli.pad,
                line_spacing: cli.line_spacing,
                invert: cli.invert,
                bold: cli.bold,
                ink: cli.ink,
                paper: cli.paper,
            },
            wrap: cli.wrap,
            notes: !cli.no_notes,
            anchor: (0, 0),
            geometry: None,
            pages: Vec::new(),
            stage: Stage::Ready,
            page: 0,
        })
    }

    /// Lays the text out for a terminal of this size and finds the page that was on show in the result.
    fn lay_out(&mut self, columns: usize, rows: usize) {
        self.geometry = Geometry::new(columns, rows, self.markers[0].side(), &self.settings);
        self.pages = match &self.geometry {
            Some(geometry) => text::paginate(
                &self.text,
                geometry.text.width(),
                geometry.lines_per_page(self.settings.line_spacing),
                self.wrap,
            ),
            None => Vec::new(),
        };
        let begins_at_or_before_anchor = |page: &Page| page[0].at <= self.anchor;
        self.page = self
            .pages
            .iter()
            .rposition(begins_at_or_before_anchor)
            .unwrap_or(0);
    }

    /// Puts a page on show, whatever was before, if there is such a page.
    fn turn_to(&mut self, page: usize) {
        if let Some(lines) = self.pages.get(page) {
            self.stage = Stage::Pages;
            self.page = page;
            self.anchor = lines[0].at;
        }
    }

    /// On to the next screen: from the start screen to the first page, from the last page to the end
    /// screen.
    fn forward(&mut self) {
        match self.stage {
            Stage::Ready if !self.pages.is_empty() => self.turn_to(0),
            Stage::Pages if self.page + 1 < self.pages.len() => self.turn_to(self.page + 1),
            // only the end screen follows the last page, or a text without any
            Stage::Ready | Stage::Pages if self.geometry.is_some() => self.stage = Stage::Done,
            Stage::Ready | Stage::Pages | Stage::Done => {}
        }
    }

    /// Back to the page before, and from the end screen to the last page.
    fn back(&mut self) {
        match self.stage {
            Stage::Ready => {}
            Stage::Pages => self.turn_to(self.page.saturating_sub(1)),
            Stage::Done => self.turn_to(self.pages.len().saturating_sub(1)),
        }
    }

    fn on_key(&mut self, key: KeyEvent) -> Reaction {
        let settings = &mut self.settings;
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Reaction::Quit,
            KeyCode::Char('q') | KeyCode::Esc => return Reaction::Quit,
            KeyCode::Char(' ' | 'n')
            | KeyCode::Right
            | KeyCode::Down
            | KeyCode::PageDown
            | KeyCode::Enter => self.forward(),
            KeyCode::Char('p') | KeyCode::Left | KeyCode::Up | KeyCode::PageUp | KeyCode::Backspace => {
                self.back()
            }
            KeyCode::Char('g') | KeyCode::Home => self.turn_to(0),
            KeyCode::Char('G') | KeyCode::End => self.turn_to(self.pages.len().saturating_sub(1)),
            KeyCode::Char('+' | '=') => step(&mut settings.module, 1, MODULE),
            KeyCode::Char('-' | '_') => step(&mut settings.module, -1, MODULE),
            KeyCode::Char(']') => step(&mut settings.pad, 1, PAD),
            KeyCode::Char('[') => step(&mut settings.pad, -1, PAD),
            KeyCode::Char('}') => step(&mut settings.margin, 1, MARGIN),
            KeyCode::Char('{') => step(&mut settings.margin, -1, MARGIN),
            KeyCode::Char('.' | '>') => step(&mut settings.line_spacing, 1, LINE_SPACING),
            KeyCode::Char(',' | '<') => step(&mut settings.line_spacing, -1, LINE_SPACING),
            KeyCode::Char('i') => settings.invert = !settings.invert,
            KeyCode::Char('b') => settings.bold = !settings.bold,
            KeyCode::Char('w') => {
                self.wrap = match self.wrap {
                    Wrap::Word => Wrap::None,
                    Wrap::None => Wrap::Word,
                }
            }
            KeyCode::Char('h' | '?') => self.notes = !self.notes,
            _ => return Reaction::Nothing,
        }
        Reaction::Redraw
    }

    /// The picture for the layout made by [`App::lay_out`].
    fn screen(&self, size: &Size) -> Screen {
        let Some(geometry) = &self.geometry else {
            let lines = [
                format!(
                    "{}x{} cells are too few for these markers",
                    size.columns, size.rows
                ),
                "smaller font, or: - markers  { margin  [ padding".to_owned(),
            ];
            return Screen::notice(size.columns, size.rows, &self.settings, &lines);
        };
        let (signal, lines): (_, Cow<[Line]>) = match self.stage {
            Stage::Ready => {
                let pace = match self.interval {
                    Some(seconds) => format!(", a new one every {seconds} s"),
                    None => String::new(),
                };
                let message = format!(
                    "Ready to show {}: {}{pace}.\n\nPoint the camera at this screen. ocrs says so when it has \
                     it in view.\n\nEnter starts.",
                    self.file.display(),
                    self.page_count()
                );
                (&self.signals.ready, self.message(geometry, &message).into())
            }
            Stage::Pages => {
                let page = self.pages.get(self.page).map_or(&[][..], Vec::as_slice);
                (&self.signals.pages[self.page % PAGE_NUMBERS], page.into())
            }
            Stage::Done => {
                let message = format!(
                    "End of {}.\n\np goes back, g starts again, q quits.",
                    self.file.display()
                );
                (&self.signals.done, self.message(geometry, &message).into())
            }
        };
        let notes = if self.notes {
            self.notes(geometry, size)
        } else {
            Vec::new()
        };
        chart::compose(geometry, &self.settings, &self.markers, signal, &lines, &notes)
    }

    /// `2 pages`
    fn page_count(&self) -> String {
        match self.pages.len() {
            1 => "1 page".to_owned(),
            pages => format!("{pages} pages"),
        }
    }

    /// A message of the program's own, laid out like a page of the text.
    fn message(&self, geometry: &Geometry, message: &str) -> Page {
        let height = geometry.lines_per_page(self.settings.line_spacing);
        text::paginate(message, geometry.text.width(), height, Wrap::Word)
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// What is on show and what the layout measures, then the keys.
    fn notes(&self, geometry: &Geometry, size: &Size) -> Vec<String> {
        let settings = &self.settings;
        let stage = match self.stage {
            Stage::Ready => format!("start, {} to come", self.page_count()),
            Stage::Pages => format!("page {}/{}", self.page + 1, self.pages.len()),
            Stage::Done => format!("end, {} shown", self.page_count()),
        };
        let state = format!(
            "{stage} | module {} | margin {} | pad {} | spacing {}",
            settings.module, settings.margin, settings.pad, settings.line_spacing,
        );
        let mut sizes = format!(
            "{} | text {}x{} cells",
            marker_size(geometry, size),
            geometry.text.width(),
            geometry.text.height()
        );
        if let Some((width, height)) = size.cell {
            sizes.push_str(&format!(" | cell {}x{} px", short(width), short(height)));
        }
        let keys = [
            "enter/space/n next   p previous   g/G first/last page",
            "+/- markers   ]/[ padding   }/{ margin   ./, line spacing",
            "i invert   b bold   w wrap   h hide these notes   q quit",
        ];
        [state, sizes]
            .into_iter()
            .chain(keys.map(str::to_owned))
            .collect()
    }

    /// The command line that starts with the layout as it is now.
    fn command_line(&self) -> String {
        let settings = &self.settings;
        let mut words = vec!["ocrs-show".to_owned(), shell_word(&self.file.to_string_lossy())];
        if !self.dictionary.eq_ignore_ascii_case(DEFAULT_DICTIONARY) {
            words.push(format!("--dict {}", shell_word(&self.dictionary)));
        }
        if self.ids != DEFAULT_IDS {
            words.push(format!("--ids {}", self.ids.map(|id| id.to_string()).join(",")));
        }
        words.push(format!(
            "--module {} --margin {} --pad {} --line-spacing {}",
            settings.module, settings.margin, settings.pad, settings.line_spacing
        ));
        if self.wrap == Wrap::None {
            words.push("--wrap none".to_owned());
        }
        if settings.invert {
            words.push("--invert".to_owned());
        }
        if settings.bold {
            words.push("--bold".to_owned());
        }
        if settings.ink != Rgb::BLACK {
            words.push(format!("--ink {}", settings.ink));
        }
        if settings.paper != Rgb::WHITE {
            words.push(format!("--paper {}", settings.paper));
        }
        if let Some(seconds) = self.interval {
            words.push(format!("--interval {seconds}"));
        }
        if !self.notes {
            words.push("--no-notes".to_owned());
        }
        words.join(" ")
    }

    /// What the terminal and the layout measured when they were last drawn.
    fn measurements(&self, size: &Size) -> String {
        let mut parts = vec![format!("terminal {}x{} cells", size.columns, size.rows)];
        if let Some((width, height)) = size.cell {
            parts.push(format!("cell {}x{} px", short(width), short(height)));
        }
        match &self.geometry {
            Some(geometry) => {
                parts.push(marker_size(geometry, size));
                parts.push(format!(
                    "text {}x{} cells, {} lines a page",
                    geometry.text.width(),
                    geometry.text.height(),
                    geometry.lines_per_page(self.settings.line_spacing)
                ));
            }
            None => parts.push("too small for the markers".to_owned()),
        }
        parts.join(", ")
    }
}

/// Moves `value` one step up or down, as far as `range` goes.
fn step(value: &mut usize, by: isize, range: RangeInclusive<usize>) {
    *value = value
        .saturating_add_signed(by)
        .clamp(*range.start(), *range.end());
}

/// `marker 12x6 cells`, with the size in pixels if the terminal tells its cell size.
fn marker_size(geometry: &Geometry, size: &Size) -> String {
    let dots = geometry.marker as f64;
    let cells = format!("marker {}x{} cells", geometry.marker, short(dots / 2.0));
    match size.cell {
        Some((width, height)) => format!("{cells} = {:.0}x{:.0} px", dots * width, dots * height / 2.0),
        None => cells,
    }
}

/// A number with one decimal, or none if that would be a 0.
fn short(value: f64) -> String {
    let text = format!("{value:.1}");
    text.strip_suffix(".0").map(str::to_owned).unwrap_or(text)
}

/// `text` as one word for a shell.
fn shell_word(text: &str) -> String {
    let plain = |c: char| c.is_ascii_alphanumeric() || "_-./,:+=@%".contains(c);
    if !text.is_empty() && text.chars().all(plain) && !text.starts_with('-') {
        text.to_owned()
    } else {
        format!("'{}'", text.replace('\'', r"'\''"))
    }
}

/// Shows the chart and follows the keys until one of them quits; returns the size the terminal had then.
fn run(app: &mut App, terminal: &mut Terminal) -> Result<Size> {
    let interval = app.interval.map(Duration::from_secs_f64);
    let next_turn = || interval.map(|interval| Instant::now() + interval);
    let mut turn_at = next_turn();
    let mut size = terminal.size()?;
    let mut redraw = true;
    loop {
        if redraw {
            size = terminal.size()?;
            app.lay_out(size.columns, size.rows);
            terminal.draw(&app.screen(&size))?;
            redraw = false;
        }
        // the pages turn by themselves, the start screen and the end screen wait for a key
        if let Some(at) = turn_at
            && app.stage == Stage::Pages
            && !event::poll(at.saturating_duration_since(Instant::now()))?
        {
            app.forward();
            turn_at = next_turn();
            redraw = true;
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind != KeyEventKind::Release => match app.on_key(key) {
                Reaction::Quit => return Ok(size),
                Reaction::Redraw => {
                    // a page turned by hand stays for a whole interval too
                    turn_at = next_turn();
                    redraw = true;
                }
                Reaction::Nothing => {}
            },
            Event::Resize(..) => redraw = true,
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let mut app = App::new(Cli::parse())?;
    let mut terminal = Terminal::open()?;
    let outcome = run(&mut app, &mut terminal);
    // back to the normal screen before anything is printed
    drop(terminal);
    let size = outcome?;
    println!("{}", app.command_line());
    println!("{}", app.measurements(&size));
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// An app showing `text`, as started with `arguments`.
    fn app(text: &str, arguments: &[&str]) -> App {
        // tests run side by side, each needs a file of its own
        static FILES: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "ocrs-show-test-{}-{}.txt",
            std::process::id(),
            FILES.fetch_add(1, Ordering::Relaxed)
        ));
        fs::File::create(&path)
            .and_then(|mut file| file.write_all(text.as_bytes()))
            .unwrap();
        let mut command = vec!["ocrs-show", path.to_str().unwrap()];
        command.extend(arguments);
        let app = App::new(Cli::try_parse_from(command).unwrap());
        fs::remove_file(&path).unwrap();
        app.unwrap()
    }

    fn press(app: &mut App, key: char) -> Reaction {
        let reaction = app.on_key(KeyEvent::new(KeyCode::Char(key), KeyModifiers::NONE));
        if reaction == Reaction::Redraw {
            app.lay_out(80, 24);
        }
        reaction
    }

    fn numbered_lines(count: usize) -> String {
        (1..=count).map(|number| format!("line {number}\n")).collect()
    }

    fn first_line(app: &App) -> &str {
        &app.pages[app.page][0].text
    }

    /// Which signal the chart on an 80x24 terminal with the default layout shows: there the signal marker
    /// has its top-left dot in column 34 and half-row 2, and modules of 2 dots.
    fn signal_on(screen: &Screen) -> Signal {
        let signals = Signals::new(DEFAULT_DICTIONARY, DEFAULT_IDS).unwrap();
        let candidates = [Signal::Ready, Signal::Done]
            .into_iter()
            .chain((0..PAGE_NUMBERS).map(Signal::Page));
        let shown = |signal: &Signal| {
            let pattern = markers::pattern(DEFAULT_DICTIONARY, signals.id(*signal)).unwrap();
            (0..36).all(|module| {
                let (row, column) = (module / 6, module % 6);
                let dark = screen.dot(34 + 2 * column, 2 + 2 * row) == Rgb::BLACK;
                dark == pattern.is_dark(row, column)
            })
        };
        let mut matching = candidates.filter(shown);
        let signal = matching.next().expect("a signal marker");
        assert_eq!(matching.next(), None);
        signal
    }

    const SIZE: Size = Size {
        columns: 80,
        rows: 24,
        cell: None,
    };

    #[test]
    fn keys_turn_the_pages_from_the_start_screen_to_the_end_screen() {
        // 8 lines fit on a page of an 80x24 terminal
        let mut app = app(&numbered_lines(20), &[]);
        app.lay_out(80, 24);
        assert_eq!((app.stage, app.pages.len()), (Stage::Ready, 3));
        // nothing comes before the start screen
        press(&mut app, 'p');
        assert_eq!(app.stage, Stage::Ready);
        assert_eq!(
            app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            Reaction::Redraw
        );
        assert_eq!((app.stage, first_line(&app)), (Stage::Pages, "line 1"));
        press(&mut app, ' ');
        assert_eq!(first_line(&app), "line 9");
        press(&mut app, 'n');
        assert_eq!(first_line(&app), "line 17");
        press(&mut app, 'p');
        assert_eq!(first_line(&app), "line 9");
        press(&mut app, 'G');
        assert_eq!(first_line(&app), "line 17");
        // after the last page comes the end screen, and after that nothing
        press(&mut app, 'n');
        assert_eq!(app.stage, Stage::Done);
        press(&mut app, 'n');
        assert_eq!(app.stage, Stage::Done);
        press(&mut app, 'p');
        assert_eq!((app.stage, first_line(&app)), (Stage::Pages, "line 17"));
        press(&mut app, 'g');
        assert_eq!(first_line(&app), "line 1");
        assert_eq!(press(&mut app, 'x'), Reaction::Nothing);
        assert_eq!(press(&mut app, 'q'), Reaction::Quit);
    }

    #[test]
    fn every_screen_says_what_it_is() {
        // 17 pages, one more than there are page numbers
        let mut app = app(&numbered_lines(8 * 17), &[]);
        app.lay_out(80, 24);
        let texts = |app: &App| {
            let runs = app.screen(&SIZE).runs;
            runs.into_iter()
                .map(|run| run.text)
                .collect::<Vec<_>>()
                .join(" / ")
        };

        assert_eq!(signal_on(&app.screen(&SIZE)), Signal::Ready);
        let start = texts(&app);
        assert!(
            start.contains("17 pages") && start.contains("Enter starts."),
            "{start}"
        );
        assert!(start.contains("start, 17 pages to come | module 2"), "{start}");

        for page in 0..17 {
            press(&mut app, ' ');
            assert_eq!(
                signal_on(&app.screen(&SIZE)),
                Signal::Page(page % 16),
                "page {page}"
            );
        }
        assert!(texts(&app).contains("page 17/17 | module 2"));

        press(&mut app, ' ');
        assert_eq!(signal_on(&app.screen(&SIZE)), Signal::Done);
        let end = texts(&app);
        assert!(
            end.contains("End of") && end.contains("end, 17 pages shown | module 2"),
            "{end}"
        );
    }

    #[test]
    fn start_screen_tells_the_pace() {
        let mut app = app("one page", &["--interval", "2.5"]);
        app.lay_out(80, 24);
        let runs = app.screen(&SIZE).runs;
        let start = runs.into_iter().map(|run| run.text).collect::<Vec<_>>().join(" ");
        assert!(start.contains("1 page, a new one every 2.5 s."), "{start}");
    }

    #[test]
    fn pages_go_forward_up_to_the_end_screen() {
        let mut app = app(&numbered_lines(20), &[]);
        app.lay_out(80, 24);
        let mut seen = Vec::new();
        for _ in 0..5 {
            app.forward();
            app.lay_out(80, 24);
            seen.push((app.stage, first_line(&app).to_owned()));
        }
        let page = |line: &str| (Stage::Pages, line.to_owned());
        let end = (Stage::Done, "line 17".to_owned());
        assert_eq!(
            seen,
            [page("line 1"), page("line 9"), page("line 17"), end.clone(), end]
        );
    }

    #[test]
    fn text_without_pages_goes_from_the_start_to_the_end() {
        let mut app = app("\n\n", &[]);
        app.lay_out(80, 24);
        assert!(app.pages.is_empty());
        assert_eq!(signal_on(&app.screen(&SIZE)), Signal::Ready);
        press(&mut app, ' ');
        assert_eq!(app.stage, Stage::Done);
        press(&mut app, 'p');
        press(&mut app, 'g');
        assert_eq!(app.stage, Stage::Done);
    }

    #[test]
    fn page_on_show_survives_a_new_layout() {
        let mut app = app(&numbered_lines(100), &[]);
        app.lay_out(80, 24);
        // from the start screen to the sixth page
        for _ in 0..6 {
            press(&mut app, ' ');
        }
        assert_eq!(first_line(&app), "line 41");
        // a larger terminal, larger markers, more space between the lines: line 41 stays on show
        for (columns, rows) in [(120, 50), (60, 20), (200, 61)] {
            app.lay_out(columns, rows);
            let texts = app.pages[app.page].iter().map(|line| line.text.as_str());
            assert!(
                texts.clone().any(|text| text == "line 41"),
                "{:?}",
                texts.collect::<Vec<_>>()
            );
        }
        app.lay_out(80, 24);
        assert_eq!(first_line(&app), "line 41");
        press(&mut app, '.');
        assert!(app.pages[app.page].iter().any(|line| line.text == "line 41"));
        assert_eq!(app.settings.line_spacing, 1);
    }

    #[test]
    fn page_that_begins_inside_a_wrapped_line_is_found_again() {
        // one line of the file that fills more than two pages
        let mut app = app(&"word ".repeat(400), &[]);
        app.lay_out(80, 24);
        assert!(app.pages.len() > 2);
        for page in 0..3 {
            press(&mut app, ' ');
            assert_eq!(app.page, page);
        }
        press(&mut app, 'p');
        assert_eq!(app.page, 1);
    }

    #[test]
    fn keys_change_the_layout_within_limits() {
        let mut app = app("text", &["--module", "15", "--pad", "0"]);
        app.lay_out(80, 24);
        for _ in 0..3 {
            press(&mut app, '+');
            press(&mut app, '[');
            press(&mut app, '}');
            press(&mut app, ',');
        }
        let settings = &app.settings;
        assert_eq!(
            (
                settings.module,
                settings.pad,
                settings.margin,
                settings.line_spacing
            ),
            (16, 0, 4, 0)
        );
        // markers of 96 columns don't fit in 80: there is a notice instead of the chart, and no page
        assert_eq!(app.geometry, None);
        let size = Size {
            columns: 80,
            rows: 24,
            cell: None,
        };
        assert!(app.screen(&size).runs[0].text.contains("80x24 cells are too few"));
        // and no way off the start screen
        press(&mut app, ' ');
        assert_eq!(app.stage, Stage::Ready);
        for _ in 0..14 {
            press(&mut app, '-');
        }
        assert!(app.geometry.is_some());
        assert_eq!(first_line(&app), "text");
    }

    #[test]
    fn command_line_repeats_the_layout() {
        let mut app = app(
            "text",
            &[
                "--dict",
                "apriltag_36h11",
                "--ids",
                "4,5,6,7",
                "--paper",
                "#C0c0C0",
            ],
        );
        app.lay_out(80, 24);
        for key in ['+', ']', '.', 'i', 'w', 'h'] {
            press(&mut app, key);
        }
        let line = app.command_line();
        let options = line.split_once(".txt ").expect("the file comes first").1;
        assert_eq!(
            options,
            "--dict apriltag_36h11 --ids 4,5,6,7 --module 3 --margin 1 --pad 2 --line-spacing 1 --wrap none \
             --invert --paper c0c0c0 --no-notes"
        );
        // and it parses back into the same settings
        let again = Cli::try_parse_from(line.split(' ')).unwrap();
        assert_eq!(
            (again.module, again.pad, again.invert, again.paper),
            (3, 2, true, Rgb(192, 192, 192))
        );
    }

    #[test]
    fn notes_tell_the_page_and_the_sizes() {
        let mut app = app(&numbered_lines(20), &[]);
        app.lay_out(80, 24);
        press(&mut app, ' ');
        press(&mut app, ' ');
        let size = Size {
            columns: 80,
            rows: 24,
            cell: Some((9.0, 19.5)),
        };
        let geometry = app.geometry.clone().unwrap();
        assert_eq!(
            app.notes(&geometry, &size)[..2],
            [
                "page 2/3 | module 2 | margin 1 | pad 1 | spacing 0",
                "marker 12x6 cells = 108x117 px | text 48x8 cells | cell 9x19.5 px"
            ]
        );
        // hidden notes leave only the page on the screen
        press(&mut app, 'h');
        assert_eq!(app.screen(&size).runs.len(), 8);
        press(&mut app, '?');
        assert_eq!(app.screen(&size).runs.len(), 8 + 5);
        assert_eq!(
            app.measurements(&size),
            "terminal 80x24 cells, cell 9x19.5 px, marker 12x6 cells = 108x117 px, text 48x8 cells, 8 lines a page"
        );
    }

    #[test]
    fn options_are_checked() {
        let start = |arguments: &[&str]| {
            let mut command = vec!["ocrs-show", "Cargo.toml"];
            command.extend(arguments);
            Cli::try_parse_from(command)
                .map_err(anyhow::Error::from)
                .and_then(App::new)
        };
        assert!(start(&[]).is_ok());
        for wrong in [
            &["--module", "0"][..],
            &["--module", "17"],
            &["--ids", "1,2,3"],
            &["--ids", "1,2,3,1"],
            &["--ids", "0,1,2,50"],
            &["--dict", "9x9"],
            &["--ink", "ffffff", "--paper", "000000"],
            &["--paper", "white"],
            &["--interval", "0"],
            &["--interval", "nan"],
            &["--wrap", "char"],
        ] {
            assert!(start(wrong).is_err(), "{wrong:?}");
        }
        assert_eq!(shell_word("notes/page 1.txt"), "'notes/page 1.txt'");
        assert_eq!(shell_word("it's"), r"'it'\''s'");
        assert_eq!(shell_word("-o"), "'-o'");
    }
}
