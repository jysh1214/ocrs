//! End to end on synthetic frames, on the CPU: a page with four markers and some text, seen at an angle.

use anyhow::Result;
use opencv::core::{self, CV_8UC3, Mat, Point, Point2f, Rect, Scalar, Size, Vector};
use opencv::prelude::*;
use opencv::{imgproc, objdetect};

use ocrs::ocr::Models;
use ocrs::{Device, Options, Outcome, Pipeline};

const PAGE: Size = Size {
    width: 1280,
    height: 720,
};
const MARKER: i32 = 96;
const MARGIN: i32 = 24;
/// Top-left, top-right, bottom-right, bottom-left
const IDS: [i32; 4] = [7, 3, 11, 5];

/// A white page with the markers in its corners and `lines` between them.
fn page(lines: &[&str]) -> Result<Mat> {
    let mut page = Mat::new_size_with_default(PAGE, CV_8UC3, Scalar::all(255.0))?;
    let dictionary = objdetect::get_predefined_dictionary(objdetect::PredefinedDictionaryType::DICT_4X4_50)?;
    let (right, bottom) = (PAGE.width - MARGIN - MARKER, PAGE.height - MARGIN - MARKER);
    for (id, (x, y)) in IDS.into_iter().zip([
        (MARGIN, MARGIN),
        (right, MARGIN),
        (right, bottom),
        (MARGIN, bottom),
    ]) {
        let mut marker = Mat::default();
        objdetect::generate_image_marker(&dictionary, id, MARKER, &mut marker, 1)?;
        let mut colour = Mat::default();
        imgproc::cvt_color_def(&marker, &mut colour, imgproc::COLOR_GRAY2BGR)?;
        colour.copy_to(&mut page.roi_mut(Rect::new(x, y, MARKER, MARKER))?)?;
    }
    for (row, line) in lines.iter().enumerate() {
        let origin = Point::new(MARGIN + MARKER + 40, MARGIN + MARKER + 70 + 60 * row as i32);
        imgproc::put_text(
            &mut page,
            line,
            origin,
            imgproc::FONT_HERSHEY_DUPLEX,
            1.2,
            Scalar::all(0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(page)
}

/// A camera frame that shows `page` smaller, off-centre and in perspective on a grey table.
fn photograph(page: &Mat) -> Result<Mat> {
    let frame_size = Size::new(1920, 1080);
    let (w, h) = ((PAGE.width - 1) as f32, (PAGE.height - 1) as f32);
    let flat = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].map(|(x, y)| Point2f::new(x, y));
    let seen =
        [(210.0, 130.0), (1650.0, 90.0), (1740.0, 960.0), (150.0, 900.0)].map(|(x, y)| Point2f::new(x, y));
    let transform = imgproc::get_perspective_transform_def(
        &Vector::<Point2f>::from_slice(&flat),
        &Vector::<Point2f>::from_slice(&seen),
    )?;
    let mut frame = Mat::default();
    imgproc::warp_perspective(
        page,
        &mut frame,
        &transform,
        frame_size,
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        Scalar::all(90.0),
    )?;
    Ok(frame)
}

fn pipeline() -> Result<Pipeline> {
    pipeline_on(Device::Cpu)
}

fn pipeline_on(device: Device) -> Result<Pipeline> {
    pipeline_with(device, 1, 1.0)
}

/// A pipeline that wants `stable_frames` frames to agree, at `min_similarity`, before a text counts.
fn pipeline_with(device: Device, stable_frames: usize, min_similarity: f64) -> Result<Pipeline> {
    Pipeline::new(Options {
        device,
        dictionary: "4x4_50".to_owned(),
        marker_ids: IDS,
        page_size: None,
        stable_frames,
        min_similarity,
        models: Models::built_in().expect("no models in model/: tools/download_models.py fetches them"),
    })
}

#[test]
fn reads_the_text_between_the_markers() -> Result<()> {
    let lines = [
        "Invoice 2026-0417",
        "Total due: 1,284.50 USD",
        "Thank you for your order!",
    ];
    let frame = photograph(&page(&lines)?)?;
    let mut pipeline = pipeline()?;
    let outcome = pipeline.process(&frame)?;
    assert_eq!(outcome, Outcome::Text(lines.join("\n")));
    Ok(())
}

#[test]
fn consecutive_frames_with_the_same_text_are_reported_once() -> Result<()> {
    let first = photograph(&page(&["First page", "alpha beta gamma"])?)?;
    let second = photograph(&page(&["Second page", "delta epsilon"])?)?;
    let mut pipeline = pipeline()?;
    let mut new_texts = Vec::new();
    for frame in [&first, &first, &first, &second, &second, &first] {
        match pipeline.process(frame)? {
            Outcome::Text(text) => new_texts.push(text),
            outcome => assert_eq!(outcome, Outcome::Duplicate),
        }
    }
    assert_eq!(
        new_texts,
        [
            "First page\nalpha beta gamma",
            "Second page\ndelta epsilon",
            "First page\nalpha beta gamma"
        ]
    );
    Ok(())
}

#[test]
fn text_counts_once_most_of_the_recent_frames_agree() -> Result<()> {
    let first = photograph(&page(&["First page", "alpha beta gamma"])?)?;
    let second = photograph(&page(&["Second page", "delta epsilon"])?)?;
    // what a camera sees while the page changes: both at once
    let mut between = Mat::default();
    core::add_weighted_def(&first, 0.5, &second, 0.5, 0.0, &mut between)?;

    let mut pipeline = pipeline_with(Device::Cpu, 3, 0.9)?;
    let frames = [
        &first, &first, &first, &first, &between, &second, &second, &second, &second,
    ];
    let mut outcomes = Vec::new();
    for frame in frames {
        outcomes.push(pipeline.process(frame)?);
    }
    let text = |lines: &str| Outcome::Text(lines.to_owned());
    assert_eq!(
        outcomes,
        [
            Outcome::Unsettled,
            Outcome::Unsettled,
            text("First page\nalpha beta gamma"),
            Outcome::Duplicate,
            // whatever the frame between the pages reads, three of the last five frames read the first page
            Outcome::Duplicate,
            Outcome::Duplicate,
            // two frames for either page
            Outcome::Unsettled,
            text("Second page\ndelta epsilon"),
            Outcome::Duplicate,
        ]
    );
    assert_eq!(pipeline.finish(), None);
    Ok(())
}

#[test]
fn frame_without_all_markers_is_skipped() -> Result<()> {
    let mut frame = photograph(&page(&["Hidden"])?)?;
    // cover the top-left marker
    imgproc::rectangle(
        &mut frame,
        Rect::new(150, 60, 260, 240),
        Scalar::all(90.0),
        imgproc::FILLED,
        imgproc::LINE_8,
        0,
    )?;
    assert_eq!(pipeline()?.process(&frame)?, Outcome::NoMarkers);
    Ok(())
}

#[test]
fn page_without_text_is_not_content() -> Result<()> {
    assert_eq!(pipeline()?.process(&photograph(&page(&[])?)?)?, Outcome::NoText);
    Ok(())
}

/// The same frames through the CUDA code paths (GPU warp, networks on the CUDA backend).
#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs a CUDA GPU, run with `cargo test -- --ignored`"]
fn cuda_reads_the_same_text_as_the_cpu() -> Result<()> {
    let pages = [
        vec![
            "Invoice 2026-0417",
            "Total due: 1,284.50 USD",
            "Thank you for your order!",
        ],
        vec!["Second page", "delta epsilon"],
    ];
    let mut pipeline = pipeline_on(Device::Cuda)?;
    for lines in pages {
        let frame = photograph(&page(&lines)?)?;
        assert_eq!(pipeline.process(&frame)?, Outcome::Text(lines.join("\n")));
        assert_eq!(pipeline.process(&frame)?, Outcome::Duplicate);
    }
    Ok(())
}
