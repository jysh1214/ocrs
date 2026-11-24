use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use opencv::core::{Mat, Size, Vector};
use opencv::prelude::*;
use opencv::{imgcodecs, videoio};

use ocrs::ocr::Models;
use ocrs::{Device, Options, Outcome, Pipeline};

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
    #[arg(long, value_enum, default_value_t = Device::Cuda)]
    device: Device,

    /// Marker dictionary, an OpenCV predefined one such as 4x4_50, 6x6_250 or apriltag_36h11
    #[arg(long, default_value = "4x4_50")]
    dict: String,

    /// Ids of the markers at the top-left, top-right, bottom-right and bottom-left corner of the text
    #[arg(long, value_delimiter = ',', num_args = 4, default_values_t = [0, 1, 2, 3], value_name = "TL,TR,BR,BL")]
    ids: Vec<i32>,

    /// Size in pixels to rectify the text area to [default: its size in the first frame that shows it]
    #[arg(long, value_parser = parse_size, value_name = "WIDTHxHEIGHT")]
    page_size: Option<Size>,

    /// How alike (0 to 1) the texts of two consecutive frames must be to count as the same content. Below 1
    /// a few characters may differ, which absorbs what a camera's noise does to the reading of an unchanged
    /// page; 1 only skips a frame whose text is identical
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
    let marker_ids: [i32; 4] = cli
        .ids
        .as_slice()
        .try_into()
        .expect("clap takes exactly four ids");
    let mut pipeline = Pipeline::new(Options {
        device: cli.device,
        dictionary: cli.dict,
        marker_ids,
        page_size: cli.page_size,
        min_similarity: cli.similarity,
        models,
    })?;

    let mut capture = open(&cli.input)?;
    if let Some(dir) = &cli.dump_pages {
        fs::create_dir_all(dir).with_context(|| format!("can't create {}", dir.display()))?;
    }
    let file = File::create(&cli.output).with_context(|| format!("can't create {}", cli.output.display()))?;
    let mut output = BufWriter::new(file);

    let (mut frames, mut located, mut recorded) = (0u64, 0u64, 0u64);
    let mut frame = Mat::default();
    while capture.read(&mut frame)? && !frame.empty() {
        frames += 1;
        let outcome = pipeline
            .process(&frame)
            .with_context(|| format!("frame {frames}"))?;
        if outcome != Outcome::NoMarkers {
            located += 1;
        }
        let Outcome::Text(text) = outcome else { continue };
        if let (Some(separator), true) = (&cli.separator, recorded > 0) {
            writeln!(output, "{separator}")?;
        }
        writeln!(output, "{text}")?;
        // every frame's text is on disk before the next frame is looked at
        output.flush()?;
        recorded += 1;
        eprintln!("frame {frames}: {} lines", text.lines().count());
        if let Some(dir) = &cli.dump_pages {
            let path = dir.join(format!("frame_{frames:06}.png"));
            let saved = imgcodecs::imwrite(&path, pipeline.last_page(), &Vector::new())?;
            ensure!(saved, "can't write {}", path.display());
        }
    }

    eprintln!(
        "{frames} frames read, {located} with all four markers, {recorded} recorded in {}",
        cli.output.display()
    );
    if frames == 0 {
        bail!("no frame could be read from {}", cli.input);
    }
    if located == 0 {
        eprintln!("no frame showed all four markers, check --dict and --ids against the markers in use");
    }
    Ok(())
}

/// Opens a camera for a plain number, a file or image sequence for anything else.
fn open(input: &str) -> Result<videoio::VideoCapture> {
    let capture = match input.parse::<i32>() {
        Ok(camera) => videoio::VideoCapture::new(camera, videoio::CAP_ANY)?,
        Err(_) => videoio::VideoCapture::from_file(input, videoio::CAP_ANY)?,
    };
    ensure!(capture.is_opened()?, "can't open {input}");
    Ok(capture)
}
