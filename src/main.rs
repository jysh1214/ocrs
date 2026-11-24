mod preview;

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use clap::builder::RangedU64ValueParser;
use opencv::core::{Mat, Size, Vector};
use opencv::prelude::*;
use opencv::{imgcodecs, videoio};

use ocrs::markers;
use ocrs::ocr::Models;
use ocrs::{Device, Missed, Options, Outcome, Pipeline};

use preview::Preview;

/// Reads the text between four ArUco / AprilTag markers in every frame of a video and writes it to a text
/// file, skipping frames that show the same text as the frame before.
#[derive(Parser)]
#[command(version)]
struct Cli {
    /// Video file, image, image sequence (page_%03d.png) or camera index
    input: String,

    /// Text file to write; rewritten from scratch on every run
    #[arg(short, long)]
    output: PathBuf,

    /// Where to run the image warping and the neural networks
    #[arg(long, value_enum, default_value_t = Device::default())]
    device: Device,

    /// Frame size to ask a camera for [default: what the camera gives by itself, usually 640x480]
    #[arg(long, value_parser = parse_size, value_name = "WIDTHxHEIGHT")]
    camera_size: Option<Size>,

    /// Show what the camera sees on this terminal while reading, with the markers that are found in it
    /// marked, for pointing the camera; q quits
    #[arg(long)]
    preview: bool,

    /// Marker dictionary, an OpenCV predefined one such as 4x4_50, 6x6_250 or apriltag_36h11
    #[arg(long, default_value = "4x4_50")]
    dict: String,

    /// Ids of the markers at the top-left, top-right, bottom-right and bottom-left corner of the text
    #[arg(long, value_parser = markers::parse_ids, default_value = "0,1,2,3", value_name = "TL,TR,BR,BL")]
    ids: [i32; 4],

    /// Size in pixels to rectify the text area to [default: its size in the first frame that shows it]
    #[arg(long, value_parser = parse_size, value_name = "WIDTHxHEIGHT")]
    page_size: Option<Size>,

    /// How many frames must read a text before it is recorded: it counts once this many of the last 2N-1
    /// frames agree on it. That passes over frames that are blurred or caught between two pages; 1 records
    /// what any single frame reads [default: 3, and 1 for pictures]
    #[arg(long, value_name = "FRAMES", value_parser = RangedU64ValueParser::<usize>::new().range(1..=100))]
    stable: Option<usize>,

    /// How alike (0 to 1) two texts must be to count as the same content, for frames to agree on a text
    /// and for a text to repeat the one before it. Below 1 a few characters may differ, which absorbs what
    /// a camera's noise does to the reading of an unchanged page; 1 takes identical texts only
    #[arg(long, default_value_t = 0.9, value_name = "RATIO")]
    similarity: f64,

    /// Line to put between the texts of two frames, e.g. "" for a blank line [default: nothing]
    #[arg(long, value_name = "LINE", allow_hyphen_values = true)]
    separator: Option<String>,

    /// Save the rectified text area of every recorded frame to this directory, to check the marker setup
    #[arg(long, value_name = "DIR")]
    dump_pages: Option<PathBuf>,

    /// Text detection model (ONNX) to use instead of the built-in one
    #[arg(long, value_name = "FILE")]
    det_model: Option<PathBuf>,

    /// Text recognition model (ONNX, prepared with tools/freeze_onnx_shapes.py) instead of the built-in one
    #[arg(long, value_name = "FILE", requires = "rec_dict")]
    rec_model: Option<PathBuf>,

    /// Character list of the recognition model given with --rec-model, one character per line
    #[arg(long, value_name = "FILE", requires = "rec_model")]
    rec_dict: Option<PathBuf>,
}

fn parse_size(text: &str) -> Result<Size, String> {
    let parsed = text.split_once(['x', 'X']).and_then(|(width, height)| {
        let (width, height) = (
            width.trim().parse::<i32>().ok()?,
            height.trim().parse::<i32>().ok()?,
        );
        (width > 0 && height > 0).then(|| Size::new(width, height))
    });
    parsed.ok_or_else(|| format!("expected WIDTHxHEIGHT such as 1600x900, got {text:?}"))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(
        (0.0..=1.0).contains(&cli.similarity),
        "--similarity must be between 0 and 1"
    );

    let mut models = Models::default();
    let read = |path: &PathBuf| fs::read(path).with_context(|| format!("can't read {}", path.display()));
    if let Some(path) = &cli.det_model {
        models.detector = read(path)?;
    }
    if let (Some(model), Some(dictionary)) = (&cli.rec_model, &cli.rec_dict) {
        models.recognizer = read(model)?;
        models.dictionary = String::from_utf8(read(dictionary)?)
            .with_context(|| format!("{} is not UTF-8 text", dictionary.display()))?;
    }
    let stable_frames = cli.stable.unwrap_or(if is_picture(&cli.input) { 1 } else { 3 });
    let mut pipeline = Pipeline::new(Options {
        device: cli.device,
        dictionary: cli.dict,
        marker_ids: cli.ids,
        page_size: cli.page_size,
        stable_frames,
        min_similarity: cli.similarity,
        models,
    })?;

    let mut capture = open(&cli.input, cli.camera_size)?;
    if let Some(dir) = &cli.dump_pages {
        fs::create_dir_all(dir).with_context(|| format!("can't create {}", dir.display()))?;
    }
    let file = File::create(&cli.output).with_context(|| format!("can't create {}", cli.output.display()))?;
    let mut recorder = Recorder {
        output: BufWriter::new(file),
        separator: cli.separator.as_deref(),
        dump_pages: cli.dump_pages.as_deref(),
        recorded: 0,
    };

    let mut log = Log {
        preview: if cli.preview { Some(Preview::open()?) } else { None },
        held: Vec::new(),
    };

    let (mut frames, mut located, mut unsettled) = (0u64, 0u64, 0u64);
    // whether the last frame with markers showed the start screen of ocrs-show
    let mut waiting = false;
    let (mut frame_rate, mut frame_at) = (0.0, Instant::now());
    let mut frame = Mat::default();
    while capture.read(&mut frame)? && !frame.empty() {
        frames += 1;
        let missed_before = pipeline.missed();
        let outcome = pipeline
            .process(&frame)
            .with_context(|| format!("frame {frames}"))?;
        say_missed(
            &mut log,
            &format!("frame {frames}: "),
            missed_before,
            pipeline.missed(),
        );
        let was_waiting = waiting;
        if outcome != Outcome::NoMarkers {
            located += 1;
            waiting = outcome == Outcome::Ready;
        }
        unsettled += u64::from(outcome == Outcome::Unsettled);
        let done = outcome == Outcome::Done;
        match outcome {
            Outcome::Text(text) => {
                recorder.record(&text, frames, pipeline.last_page())?;
                log.say(format!(
                    "frame {frames}: {} lines{}",
                    text.lines().count(),
                    pipeline
                        .last_character_width()
                        .map_or(String::new(), |width| format!(", {width:.1} px a column"))
                ));
            }
            Outcome::Guess(text) => {
                recorder.record(&text, frames, pipeline.last_page())?;
                log.say(format!(
                    "frame {frames}: {} lines, the likeliest reading of a page that went away before its frames \
                     agreed",
                    text.lines().count()
                ));
            }
            Outcome::Ready if !was_waiting => {
                log.say(format!(
                    "frame {frames}: start screen in view, waiting for the first page"
                ));
            }
            Outcome::Done => log.say(format!("frame {frames}: end screen in view")),
            _ => {}
        }
        if let Some(preview) = &mut log.preview {
            // smoothed over the last few frames
            let now = Instant::now();
            let rate = 1.0 / now.duration_since(frame_at).as_secs_f64().max(1e-3);
            frame_rate = if frames == 1 {
                rate
            } else {
                0.8 * frame_rate + 0.2 * rate
            };
            frame_at = now;
            let size = frame.size()?;
            let facts = format!(
                "{}x{} | frame {frames} | {frame_rate:.1} frames a second | {} texts recorded{}",
                size.width,
                size.height,
                recorder.recorded,
                pipeline
                    .last_character_width()
                    .map_or(String::new(), |width| format!(" | text {width:.1} px a column"))
            );
            let remark = log.held.last().map_or("", String::as_str);
            preview.draw(&frame, pipeline.last_survey(), &facts, remark)?;
            if preview.quit_requested()? {
                log.say(format!("frame {frames}: stopped by a key"));
                break;
            }
        }
        if done {
            break;
        }
    }
    let missed_before = pipeline.missed();
    if let Some(text) = pipeline.finish() {
        recorder.record(&text, frames, pipeline.last_page())?;
        log.say(format!(
            "{} lines, the likeliest reading of the last page, which was not on show until its frames agreed",
            text.lines().count()
        ));
    }
    say_missed(&mut log, "", missed_before, pipeline.missed());

    log.say(format!(
        "{frames} frames read, {located} with all four markers, {} recorded in {}",
        recorder.recorded,
        cli.output.display()
    ));
    let missed = pipeline.missed();
    if missed != Missed::default() {
        log.say(format!(
            "NOT RECORDED: {} of the pages that ocrs-show had on show ({} unseen, {} seen but not read)",
            missed.unseen + missed.unread,
            missed.unseen,
            missed.unread
        ));
    }
    if frames > 0 && located == 0 {
        log.say("no frame showed all four markers, check --dict and --ids against the markers in use");
    } else if recorder.recorded == 0 && unsettled > 0 {
        log.say(format!(
            "the frames never agreed on a text: {stable_frames} of {} in a row have to, see --stable and \
             --similarity",
            2 * stable_frames - 1
        ));
    }
    log.close();
    if frames == 0 {
        bail!("no frame could be read from {}", cli.input);
    }
    Ok(())
}

/// Says which pages of `ocrs-show` went missing between two states of the count.
fn say_missed(log: &mut Log, prefix: &str, before: Missed, now: Missed) {
    let pages = |count: u64| {
        if count == 1 {
            "1 page".to_owned()
        } else {
            format!("{count} pages")
        }
    };
    if now.unseen > before.unseen {
        log.say(format!(
            "{prefix}{} went by unseen, going by the page numbers (or the pages were turned back)",
            pages(now.unseen - before.unseen)
        ));
    }
    if now.unread > before.unread {
        log.say(format!(
            "{prefix}{} went by without a frame that could be read",
            pages(now.unread - before.unread)
        ));
    }
}

/// Where the remarks about the run go: to stderr, at once or, while the preview has the terminal, when it
/// has given it back.
struct Log {
    preview: Option<Preview>,
    /// The remarks made while the preview was on
    held: Vec<String>,
}

impl Log {
    fn say(&mut self, remark: impl Into<String>) {
        let remark = remark.into();
        if self.preview.is_some() {
            self.held.push(remark);
        } else {
            eprintln!("{remark}");
        }
    }

    /// Ends the preview, if there is one, and prints what was said meanwhile.
    fn close(&mut self) {
        self.preview = None;
        for remark in self.held.drain(..) {
            eprintln!("{remark}");
        }
    }
}

impl Drop for Log {
    /// Also when an error ends the run, the terminal comes back before anything is printed.
    fn drop(&mut self) {
        self.close();
    }
}

/// Appends the texts to the output file, and the pages they were read from to the directory for those.
struct Recorder<'a> {
    output: BufWriter<File>,
    separator: Option<&'a str>,
    dump_pages: Option<&'a Path>,
    recorded: u64,
}

impl Recorder<'_> {
    /// `frame` is the number of the frame at which the text is recorded, `page` what was read last.
    fn record(&mut self, text: &str, frame: u64, page: &Mat) -> Result<()> {
        if let (Some(separator), true) = (self.separator, self.recorded > 0) {
            writeln!(self.output, "{separator}")?;
        }
        writeln!(self.output, "{text}")?;
        // every text is on disk before the next frame is looked at
        self.output.flush()?;
        self.recorded += 1;
        if let Some(dir) = self.dump_pages {
            let path = dir.join(format!("frame_{frame:06}.png"));
            let saved = imgcodecs::imwrite(&path, page, &Vector::new())?;
            ensure!(saved, "can't write {}", path.display());
        }
        Ok(())
    }
}

/// Whether the input is one picture or a sequence of them, going by its name. Each picture is a view of
/// its own, so there is nothing for consecutive ones to agree on, unlike the frames of a camera or video.
fn is_picture(input: &str) -> bool {
    const PICTURES: [&str; 15] = [
        "png", "jpg", "jpeg", "jpe", "bmp", "dib", "tif", "tiff", "webp", "pbm", "pgm", "ppm", "pnm", "jp2",
        "avif",
    ];
    Path::new(input)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            PICTURES
                .iter()
                .any(|picture| picture.eq_ignore_ascii_case(extension))
        })
}

/// Opens a camera for a plain number, a file or image sequence for anything else. A camera is asked for
/// frames of `camera_size`, and says on stderr what size its frames have.
fn open(input: &str, camera_size: Option<Size>) -> Result<videoio::VideoCapture> {
    let Ok(camera) = input.parse::<i32>() else {
        ensure!(
            camera_size.is_none(),
            "--camera-size is for a camera, which {input} is not"
        );
        let capture = videoio::VideoCapture::from_file(input, videoio::CAP_ANY)?;
        ensure!(capture.is_opened()?, "can't open {input}");
        return Ok(capture);
    };
    let mut capture = videoio::VideoCapture::new(camera, videoio::CAP_ANY)?;
    ensure!(capture.is_opened()?, "can't open camera {camera}");
    let size = match camera_size {
        Some(wanted) => request_size(&mut capture, wanted)?,
        None => frame_size(&capture)?,
    };
    eprintln!("camera {camera}: {}x{} frames", size.width, size.height);
    if let Some(wanted) = camera_size.filter(|wanted| *wanted != size) {
        eprintln!(
            "  it has no {}x{} frames, this is the nearest size (`v4l2-ctl --list-formats-ext` lists them)",
            wanted.width, wanted.height
        );
    }
    Ok(capture)
}

/// What is asked of a camera apart from frames. A trait, so that the asking can be tested without a camera.
trait Camera {
    fn property(&self, id: i32) -> Result<f64>;
    /// Whether the camera took the value
    fn set_property(&mut self, id: i32, value: f64) -> Result<bool>;
}

impl Camera for videoio::VideoCapture {
    fn property(&self, id: i32) -> Result<f64> {
        Ok(self.get(id)?)
    }

    fn set_property(&mut self, id: i32, value: f64) -> Result<bool> {
        Ok(self.set(id, value)?)
    }
}

/// The size of the frames a camera delivers.
fn frame_size(camera: &impl Camera) -> Result<Size> {
    let pixels = |id| Ok::<_, anyhow::Error>(camera.property(id)?.round() as i32);
    Ok(Size::new(
        pixels(videoio::CAP_PROP_FRAME_WIDTH)?,
        pixels(videoio::CAP_PROP_FRAME_HEIGHT)?,
    ))
}

/// Motion JPEG as OpenCV and V4L2 name pixel formats, by the four characters `MJPG`.
const MOTION_JPEG: f64 = u32::from_le_bytes(*b"MJPG") as f64;

/// Asks a camera for frames of `wanted` size. Returns the size it delivers from then on, which is the
/// nearest it has if it doesn't have that one.
///
/// A camera has each of its sizes in certain pixel formats only, and USB cameras tend to have the large
/// ones as Motion JPEG alone, because uncompressed frames of that size don't fit through the cable fast
/// enough. OpenCV asks in the uncompressed format it prefers, so when that gives another size the question
/// is repeated for Motion JPEG. Uncompressed frames are kept where they do come in the size asked for: they
/// are free of compression artefacts, which matters more for reading text than the frame rate does.
fn request_size(camera: &mut impl Camera, wanted: Size) -> Result<Size> {
    fn ask(camera: &mut impl Camera, wanted: Size) -> Result<Size> {
        // the camera changes its size when the second of the two arrives
        camera.set_property(videoio::CAP_PROP_FRAME_WIDTH, wanted.width as f64)?;
        camera.set_property(videoio::CAP_PROP_FRAME_HEIGHT, wanted.height as f64)?;
        frame_size(camera)
    }

    let given = ask(camera, wanted)?;
    if given == wanted {
        return Ok(given);
    }
    let format = camera.property(videoio::CAP_PROP_FOURCC)?;
    if format == MOTION_JPEG || !camera.set_property(videoio::CAP_PROP_FOURCC, MOTION_JPEG)? {
        return Ok(given);
    }
    let compressed = ask(camera, wanted)?;
    if compressed == wanted {
        return Ok(compressed);
    }
    // no better for it: back to what the camera gave first
    camera.set_property(videoio::CAP_PROP_FOURCC, format)?;
    ask(camera, wanted)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: i32 = videoio::CAP_PROP_FRAME_WIDTH;
    const HEIGHT: i32 = videoio::CAP_PROP_FRAME_HEIGHT;
    const FORMAT: i32 = videoio::CAP_PROP_FOURCC;
    const YUYV: f64 = u32::from_le_bytes(*b"YUYV") as f64;

    /// A camera that has some sizes in each of its pixel formats and settles on the nearest one to what it
    /// is asked for, the way a V4L2 camera behaves behind OpenCV.
    struct Fake {
        formats: Vec<(f64, Vec<(i32, i32)>)>,
        format: f64,
        size: (i32, i32),
        /// Width and height asked for and not yet in effect, 0 if not asked for
        asked: (i32, i32),
    }

    impl Fake {
        /// Starts like OpenCV does: in the first format, at the size nearest to 640x480.
        fn new(formats: &[(f64, &[(i32, i32)])]) -> Self {
            let mut fake = Self {
                formats: formats
                    .iter()
                    .map(|(format, sizes)| (*format, sizes.to_vec()))
                    .collect(),
                format: formats[0].0,
                size: (0, 0),
                asked: (0, 0),
            };
            fake.settle((640, 480));
            fake
        }

        fn settle(&mut self, wanted: (i32, i32)) {
            let sizes = &self
                .formats
                .iter()
                .find(|(format, _)| *format == self.format)
                .unwrap()
                .1;
            let off = |size: &&(i32, i32)| (size.0 - wanted.0).abs() + (size.1 - wanted.1).abs();
            self.size = *sizes.iter().min_by_key(off).unwrap();
        }
    }

    impl Camera for Fake {
        fn property(&self, id: i32) -> Result<f64> {
            Ok(match id {
                WIDTH => self.size.0 as f64,
                HEIGHT => self.size.1 as f64,
                FORMAT => self.format,
                _ => 0.0,
            })
        }

        fn set_property(&mut self, id: i32, value: f64) -> Result<bool> {
            match id {
                WIDTH => self.asked.0 = value as i32,
                HEIGHT => self.asked.1 = value as i32,
                FORMAT => {
                    if !self.formats.iter().any(|(format, _)| *format == value) {
                        return Ok(false);
                    }
                    self.format = value;
                    self.settle(self.size);
                }
                _ => return Ok(false),
            }
            if self.asked.0 > 0 && self.asked.1 > 0 {
                self.settle(self.asked);
                self.asked = (0, 0);
            }
            Ok(true)
        }
    }

    /// Uncompressed up to 1280x720, Motion JPEG up to 1920x1080: a common webcam.
    fn webcam() -> Fake {
        Fake::new(&[
            (YUYV, &[(640, 480), (1280, 720)]),
            (MOTION_JPEG, &[(640, 480), (1280, 720), (1920, 1080)]),
        ])
    }

    #[test]
    fn camera_starts_small() -> Result<()> {
        assert_eq!(frame_size(&webcam())?, Size::new(640, 480));
        Ok(())
    }

    #[test]
    fn size_the_camera_has_uncompressed_stays_uncompressed() -> Result<()> {
        let mut camera = webcam();
        assert_eq!(
            request_size(&mut camera, Size::new(1280, 720))?,
            Size::new(1280, 720)
        );
        assert_eq!(camera.format, YUYV);
        Ok(())
    }

    #[test]
    fn size_the_camera_only_has_compressed_switches_to_motion_jpeg() -> Result<()> {
        let mut camera = webcam();
        assert_eq!(
            request_size(&mut camera, Size::new(1920, 1080))?,
            Size::new(1920, 1080)
        );
        assert_eq!(camera.format, MOTION_JPEG);
        Ok(())
    }

    #[test]
    fn size_the_camera_does_not_have_gives_the_nearest_uncompressed() -> Result<()> {
        let mut camera = webcam();
        assert_eq!(
            request_size(&mut camera, Size::new(1400, 800))?,
            Size::new(1280, 720)
        );
        assert_eq!(camera.format, YUYV);
        Ok(())
    }

    #[test]
    fn camera_without_motion_jpeg_gives_the_nearest_size() -> Result<()> {
        let mut camera = Fake::new(&[(YUYV, &[(640, 480), (800, 600)])]);
        assert_eq!(
            request_size(&mut camera, Size::new(1920, 1080))?,
            Size::new(800, 600)
        );
        assert_eq!(camera.format, YUYV);
        Ok(())
    }

    #[test]
    fn camera_with_motion_jpeg_alone_is_asked_once() -> Result<()> {
        let mut camera = Fake::new(&[(MOTION_JPEG, &[(640, 480), (1920, 1080)])]);
        assert_eq!(
            request_size(&mut camera, Size::new(1920, 1080))?,
            Size::new(1920, 1080)
        );
        assert_eq!(
            request_size(&mut camera, Size::new(3840, 2160))?,
            Size::new(1920, 1080)
        );
        Ok(())
    }

    #[test]
    fn camera_size_is_refused_for_a_file() {
        let error = open("recording.mp4", Some(Size::new(1920, 1080))).unwrap_err();
        assert!(
            error.to_string().contains("--camera-size is for a camera"),
            "{error}"
        );
    }

    #[test]
    fn pictures_are_told_from_videos_and_cameras_by_name() {
        for picture in ["page.png", "scans/scan_%04d.JPG", "a.b.tiff"] {
            assert!(is_picture(picture), "{picture}");
        }
        for other in ["0", "2", "recording.mp4", "clip.mkv", "png", "pictures.png/video"] {
            assert!(!is_picture(other), "{other}");
        }
    }

    #[test]
    fn sizes_are_width_x_height() {
        assert_eq!(parse_size("1920x1080"), Ok(Size::new(1920, 1080)));
        assert_eq!(parse_size(" 640 X 480 "), Ok(Size::new(640, 480)));
        for wrong in ["1920", "1920x", "0x1080", "1920x-1080", "wide x high"] {
            assert!(parse_size(wrong).is_err(), "{wrong:?}");
        }
    }
}
