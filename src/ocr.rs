//! Reads the text on a page: a DB text detector finds the lines, a CTC recogniser reads each of them. Both
//! are PP-OCR models run by OpenCV's DNN module, on its CUDA backend unless the CPU is asked for.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use opencv::core::{
    self, CV_8UC3, CV_32F, Mat, Point, Point2f, Rect, RotatedRect, Scalar, Size, Size2f, ToInputArray, Vector,
};
use opencv::dnn::{self, DataLayout, Image2BlobParams, ImagePaddingMode, Net};
use opencv::imgproc;
use opencv::prelude::*;

use crate::Device;
use crate::layout::TextBox;

/// The networks and the recogniser's character list, as file contents.
pub struct Models {
    /// DB text detector, ONNX
    pub detector: Vec<u8>,
    /// CTC text recogniser, ONNX, already passed through `tools/freeze_onnx_shapes.py`
    pub recognizer: Vec<u8>,
    /// The recogniser's characters, one per line in the order of its output classes
    pub dictionary: String,
}

impl Default for Models {
    /// The English models that are compiled into the program, see `models/README.md`.
    fn default() -> Self {
        Self {
            detector: include_bytes!("../models/PP-OCRv5_det_mobile.onnx").to_vec(),
            recognizer: include_bytes!("../models/en_PP-OCRv5_rec_mobile.onnx").to_vec(),
            dictionary: include_str!("../models/ppocrv5_en_dict.txt").to_owned(),
        }
    }
}

pub struct Ocr {
    detector: Detector,
    recognizer: Recognizer,
}

impl Ocr {
    pub fn new(device: Device, models: &Models) -> Result<Self> {
        if device == Device::Cuda {
            // without this check OpenCV would log a warning and quietly run the networks on the CPU
            let targets = dnn::get_available_targets(dnn::Backend::DNN_BACKEND_CUDA)?;
            ensure!(
                targets
                    .iter()
                    .any(|target| target == dnn::Target::DNN_TARGET_CUDA),
                "this OpenCV build has no CUDA backend for neural networks (it is left out when cuDNN is missing at build time)"
            );
        }
        Ok(Self {
            detector: Detector::new(device, &models.detector)
                .context("can't load the text detection model")?,
            recognizer: Recognizer::new(device, &models.recognizer, &models.dictionary)?,
        })
    }

    /// The text on an upright BGR page, in no particular order.
    pub fn read(&mut self, page: &Mat) -> Result<Vec<TextBox>> {
        let mut boxes = Vec::new();
        for corners in self.detector.detect(page)? {
            let (text, confidence) = self.recognizer.read(page, &corners)?;
            let text = text.trim();
            if !text.is_empty() && confidence >= MIN_CONFIDENCE {
                boxes.push(TextBox {
                    corners,
                    text: text.to_owned(),
                    confidence,
                });
            }
        }
        Ok(boxes)
    }
}

/// Readings the recogniser is less sure of than this are noise more often than text (PaddleOCR's default).
const MIN_CONFIDENCE: f32 = 0.5;

fn load_net(onnx: &Vector<u8>, device: Device) -> Result<Net> {
    let mut net = dnn::read_net_from_onnx_buffer(onnx)?;
    let (backend, target) = match device {
        Device::Cuda => (dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA),
        Device::Cpu => (dnn::DNN_BACKEND_OPENCV, dnn::DNN_TARGET_CPU),
    };
    net.set_preferable_backend(backend)?;
    net.set_preferable_target(target)?;
    Ok(net)
}

/// Pages larger than this are scaled down for detection; lines are still cut out of the full-size page.
const MAX_DETECTION_SIDE: i32 = 1920;
// Post-processing of the detector's output as in PaddleOCR.
/// Probability from which a pixel belongs to text
const TEXT_THRESHOLD: f64 = 0.3;
/// Mean probability a region needs to count as text
const MIN_REGION_SCORE: f64 = 0.6;
/// Regions thinner than this many pixels are noise
const MIN_REGION_SIDE: f32 = 3.0;
/// How far to grow a region back out, see [`text_regions`]
const UNCLIP_RATIO: f32 = 1.5;
const MAX_REGIONS: usize = 1000;

struct Detector {
    net: Net,
}

impl Detector {
    fn new(device: Device, onnx: &[u8]) -> Result<Self> {
        Ok(Self {
            net: load_net(&Vector::from_slice(onnx), device)?,
        })
    }

    /// Text lines on the page, as corners in the order top-left, top-right, bottom-right, bottom-left.
    fn detect(&mut self, page: &Mat) -> Result<Vec<[Point2f; 4]>> {
        let page_size = page.size()?;
        // ImageNet normalisation, applied to the BGR channels in this order like PaddleOCR does
        let (mean, std) = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
        let blob = dnn::blob_from_image_with_params(
            page,
            Image2BlobParams {
                scalefactor: Scalar::new(
                    1.0 / (255.0 * std[0]),
                    1.0 / (255.0 * std[1]),
                    1.0 / (255.0 * std[2]),
                    1.0,
                ),
                size: detection_size(page_size),
                mean: Scalar::new(255.0 * mean[0], 255.0 * mean[1], 255.0 * mean[2], 0.0),
                swap_rb: false,
                ddepth: CV_32F,
                datalayout: DataLayout::DNN_LAYOUT_NCHW,
                paddingmode: ImagePaddingMode::DNN_PMODE_NULL,
                border_value: Scalar::default(),
            },
        )?;
        self.net.set_input_def(&blob)?;
        // one text probability per input pixel, as [1, 1, height, width]
        let output = self.net.forward_single_def()?;
        let shape = output.mat_size();
        ensure!(
            shape.len() >= 2,
            "the detection model's output has shape {:?}",
            &*shape
        );
        let probabilities = output.reshape_nd(1, &shape[shape.len() - 2..])?;
        text_regions(&probabilities, page_size)
    }
}

/// Boxes around the text regions in a map of per-pixel text probabilities, scaled to a page of the given
/// size. A DB detector predicts regions that are shrunk, so that neighbouring lines stay apart, by a margin
/// of `area / perimeter` times a ratio; the boxes are grown back by that margin.
fn text_regions(
    probabilities: &(impl MatTraitConst + ToInputArray),
    page: Size,
) -> Result<Vec<[Point2f; 4]>> {
    let mut text = Mat::default();
    core::compare(
        probabilities,
        &Scalar::all(TEXT_THRESHOLD),
        &mut text,
        core::CMP_GT,
    )?;
    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours_def(
        &text,
        &mut contours,
        imgproc::RETR_LIST,
        imgproc::CHAIN_APPROX_SIMPLE,
    )?;

    let scale_x = page.width as f32 / probabilities.cols() as f32;
    let scale_y = page.height as f32 / probabilities.rows() as f32;
    let mut regions = Vec::new();
    for contour in contours.iter().take(MAX_REGIONS) {
        let outline = imgproc::min_area_rect(&contour)?;
        let (width, height) = (outline.size.width, outline.size.height);
        if width.min(height) < MIN_REGION_SIDE || region_score(probabilities, &outline)? < MIN_REGION_SCORE {
            continue;
        }
        let margin = UNCLIP_RATIO * (width * height) / (2.0 * (width + height));
        let grown = RotatedRect {
            center: outline.center,
            size: Size2f::new(width + 2.0 * margin, height + 2.0 * margin),
            angle: outline.angle,
        };
        regions
            .push(upright_corners(&grown).map(|corner| Point2f::new(corner.x * scale_x, corner.y * scale_y)));
    }
    Ok(regions)
}

/// Mean probability inside a rectangle.
fn region_score(probabilities: &impl MatTraitConst, rectangle: &RotatedRect) -> Result<f64> {
    let corners =
        upright_corners(rectangle).map(|corner| Point::new(corner.x.round() as i32, corner.y.round() as i32));
    let map = Rect::new(0, 0, probabilities.cols(), probabilities.rows());
    let bounds = imgproc::bounding_rect(&Vector::<Point>::from_slice(&corners))? & map;
    let mut inside = Mat::zeros(bounds.height, bounds.width, core::CV_8UC1)?.to_mat()?;
    let polygon = corners
        .iter()
        .map(|corner| Point::new(corner.x - bounds.x, corner.y - bounds.y))
        .collect::<Vector<Point>>();
    imgproc::fill_convex_poly_def(&mut inside, &polygon, Scalar::all(1.0))?;
    Ok(core::mean(&probabilities.roi(bounds)?, &inside)?[0])
}

/// Network input size for a page: the page size, scaled down if it exceeds [`MAX_DETECTION_SIDE`], with
/// both sides rounded to the multiple of 32 the network needs.
fn detection_size(page: Size) -> Size {
    let scale = (MAX_DETECTION_SIDE as f64 / page.width.max(page.height) as f64).min(1.0);
    let side = |length: i32| ((length as f64 * scale / 32.0).round() as i32).max(1) * 32;
    Size::new(side(page.width), side(page.height))
}

/// Corners of a rectangle that is roughly axis-aligned, named by where they are. `RotatedRect` leaves open
/// which side it calls the width, and on a rectified page text runs left to right anyway.
fn upright_corners(rectangle: &RotatedRect) -> [Point2f; 4] {
    let (sin, cos) = rectangle.angle.to_radians().sin_cos();
    let (half_width, half_height) = (rectangle.size.width / 2.0, rectangle.size.height / 2.0);
    let mut corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(u, v): (f32, f32)| {
        let (dx, dy) = (u * half_width, v * half_height);
        Point2f::new(
            rectangle.center.x + dx * cos - dy * sin,
            rectangle.center.y + dx * sin + dy * cos,
        )
    });
    corners.sort_by(|a, b| a.x.total_cmp(&b.x));
    let [mut top_left, mut bottom_left, mut top_right, mut bottom_right] = corners;
    if top_left.y > bottom_left.y {
        std::mem::swap(&mut top_left, &mut bottom_left);
    }
    if top_right.y > bottom_right.y {
        std::mem::swap(&mut top_right, &mut bottom_right);
    }
    [top_left, top_right, bottom_right, bottom_left]
}

/// Height of the recogniser's input; lines are scaled to it.
const LINE_HEIGHT: i32 = 48;
/// Recogniser input widths. A line is padded to the next one; one longer than the last is squeezed into it.
const LINE_WIDTHS: [i32; 9] = [160, 320, 640, 960, 1280, 1600, 1920, 2240, 2560];

struct Recognizer {
    device: Device,
    onnx: Vector<u8>,
    characters: Vec<String>,
    /// One network per input width in use. OpenCV sets a network up again whenever its input shape changes,
    /// which on the GPU costs far more than keeping a few copies of this small model around.
    nets: HashMap<i32, Net>,
}

impl Recognizer {
    fn new(device: Device, onnx: &[u8], dictionary: &str) -> Result<Self> {
        // Not `lines()` plus trimming: a line holding a single space is a character too. PaddleOCR appends
        // the space after the dictionary's own entries.
        let mut characters = dictionary
            .split('\n')
            .map(|line| line.strip_suffix('\r').unwrap_or(line))
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        characters.push(" ".to_owned());

        let mut recognizer = Self {
            device,
            onnx: Vector::from_slice(onnx),
            characters,
            nets: HashMap::new(),
        };
        // loading one network up front reports a broken model now instead of at the first line of text
        recognizer
            .net(LINE_WIDTHS[0])
            .context("can't load the text recognition model")?;
        Ok(recognizer)
    }

    fn net(&mut self, width: i32) -> Result<&mut Net> {
        if !self.nets.contains_key(&width) {
            self.nets.insert(width, load_net(&self.onnx, self.device)?);
        }
        Ok(self.nets.get_mut(&width).expect("inserted above"))
    }

    /// Text of the line with the given corners, and how sure the network is of it.
    fn read(&mut self, page: &Mat, corners: &[Point2f; 4]) -> Result<(String, f32)> {
        let length = |a: Point2f, b: Point2f| (a.x - b.x).hypot(a.y - b.y);
        let [tl, tr, br, bl] = *corners;
        let width = (length(tl, tr) + length(bl, br)) / 2.0;
        let height = (length(tl, bl) + length(tr, br)) / 2.0;
        if width < 1.0 || height < 1.0 {
            return Ok((String::new(), 0.0));
        }
        let widest = LINE_WIDTHS[LINE_WIDTHS.len() - 1];
        let line_width = ((width * LINE_HEIGHT as f32 / height).round() as i32).clamp(1, widest);
        let input_width = LINE_WIDTHS
            .into_iter()
            .find(|&w| w >= line_width)
            .unwrap_or(widest);

        // Cut the line out and scale it to the input height in one step. The rest of the input stays grey,
        // which the normalisation below turns into the zeros the model was trained to see as padding.
        let mut input =
            Mat::new_rows_cols_with_default(LINE_HEIGHT, input_width, CV_8UC3, Scalar::all(127.5))?;
        let (right, bottom) = ((line_width - 1) as f32, (LINE_HEIGHT - 1) as f32);
        let target =
            [(0.0, 0.0), (right, 0.0), (right, bottom), (0.0, bottom)].map(|(x, y)| Point2f::new(x, y));
        let transform = imgproc::get_perspective_transform_def(
            &Vector::<Point2f>::from_slice(corners),
            &Vector::<Point2f>::from_slice(&target),
        )?;
        imgproc::warp_perspective(
            page,
            &mut input.roi_mut(Rect::new(0, 0, line_width, LINE_HEIGHT))?,
            &transform,
            Size::new(line_width, LINE_HEIGHT),
            imgproc::INTER_LINEAR,
            core::BORDER_REPLICATE,
            Scalar::default(),
        )?;
        // BGR as it is, scaled to -1..1
        let blob = dnn::blob_from_image(
            &input,
            1.0 / 127.5,
            Size::default(),
            Scalar::all(127.5),
            false,
            false,
            CV_32F,
        )?;

        let classes = self.characters.len() + 1;
        let net = self.net(input_width)?;
        net.set_input_def(&blob)?;
        let probabilities = net.forward_single_def()?;
        let shape = probabilities.mat_size();
        if shape.len() != 3 || shape[2] as usize != classes {
            bail!(
                "the recognition model's output has shape {:?}, expected [1, steps, {classes}] for a dictionary of {} characters plus space",
                &*shape,
                classes - 2
            );
        }
        Ok(decode(probabilities.data_typed::<f32>()?, &self.characters))
    }
}

/// Greedy CTC decoding of per-step class probabilities (class 0 is the blank, class `i` is
/// `characters[i - 1]`): the best class of every step, with repeats merged and blanks dropped. Also returns
/// the mean probability of the characters kept.
fn decode(probabilities: &[f32], characters: &[String]) -> (String, f32) {
    let mut text = String::new();
    let (mut total, mut count) = (0.0, 0);
    let mut previous = 0;
    for step in probabilities.chunks_exact(characters.len() + 1) {
        let (best, &probability) = step
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("a step has at least the blank class");
        if best != 0 && best != previous {
            text.push_str(&characters[best - 1]);
            total += probability;
            count += 1;
        }
        previous = best;
    }
    (text, if count == 0 { 0.0 } else { total / count as f32 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_size_is_a_multiple_of_32_and_capped() {
        assert_eq!(detection_size(Size::new(1280, 720)), Size::new(1280, 736));
        assert_eq!(detection_size(Size::new(1000, 500)), Size::new(992, 512));
        assert_eq!(detection_size(Size::new(3840, 2160)), Size::new(1920, 1088));
        assert_eq!(detection_size(Size::new(10, 10)), Size::new(32, 32));
    }

    fn rectangle(angle: f32, width: f32, height: f32) -> RotatedRect {
        RotatedRect {
            center: Point2f::new(100.0, 50.0),
            size: core::Size2f::new(width, height),
            angle,
        }
    }

    #[test]
    fn corners_are_named_by_position_not_by_parameterisation() {
        let expected =
            [(60.0, 40.0), (140.0, 40.0), (140.0, 60.0), (60.0, 60.0)].map(|(x, y)| Point2f::new(x, y));
        // the same rectangle, described as 80x20 and as 20x80 turned by a quarter
        for described in [
            rectangle(0.0, 80.0, 20.0),
            rectangle(90.0, 20.0, 80.0),
            rectangle(-90.0, 20.0, 80.0),
        ] {
            for (corner, wanted) in upright_corners(&described).iter().zip(expected) {
                assert!(
                    (corner.x - wanted.x).abs() < 1e-3 && (corner.y - wanted.y).abs() < 1e-3,
                    "{corner:?}"
                );
            }
        }
    }

    #[test]
    fn slightly_tilted_line_keeps_its_corner_order() {
        let [tl, tr, br, bl] = upright_corners(&rectangle(5.0, 80.0, 20.0));
        assert!(tl.x < tr.x && bl.x < br.x && tl.y < bl.y && tr.y < br.y);
        assert!(tr.y > tl.y, "a clockwise tilt lowers the right end");
    }

    #[test]
    fn text_regions_are_grown_back_and_scaled_to_the_page() -> Result<()> {
        // a 200x100 probability map with a confident 100x20 region, a weak one and a speck
        let mut probabilities = Mat::zeros(100, 200, CV_32F)?.to_mat()?;
        probabilities
            .roi_mut(Rect::new(40, 30, 100, 20))?
            .set_to(&Scalar::all(0.9), &core::no_array())?;
        probabilities
            .roi_mut(Rect::new(40, 70, 100, 20))?
            .set_to(&Scalar::all(0.4), &core::no_array())?;
        probabilities
            .roi_mut(Rect::new(180, 10, 2, 2))?
            .set_to(&Scalar::all(0.9), &core::no_array())?;

        let regions = text_regions(&probabilities, Size::new(400, 200))?;
        let [[tl, tr, br, bl]] = regions[..] else {
            panic!("expected one region, got {regions:?}")
        };
        // the outline spans the pixel centres 40..=139 x 30..=49, i.e. 99 x 19; the margin is
        // 1.5 * 99 * 19 / (2 * 118) = 11.96 on every side, and the page is twice the size of the map
        let margin = 1.5 * 99.0 * 19.0 / 236.0;
        let expected = [
            (2.0 * (40.0 - margin), 2.0 * (30.0 - margin)),
            (2.0 * (139.0 + margin), 2.0 * (30.0 - margin)),
            (2.0 * (139.0 + margin), 2.0 * (49.0 + margin)),
            (2.0 * (40.0 - margin), 2.0 * (49.0 + margin)),
        ];
        for (corner, (x, y)) in [tl, tr, br, bl].into_iter().zip(expected) {
            assert!(
                (corner.x - x).abs() < 0.01 && (corner.y - y).abs() < 0.01,
                "{corner:?} vs ({x}, {y})"
            );
        }
        Ok(())
    }

    #[test]
    fn ctc_decoding_merges_repeats_and_drops_blanks() {
        let characters = ["a", "b", " "].map(str::to_owned);
        // one row per step, columns: blank, a, b, space
        let steps: [[f32; 4]; 8] = [
            [0.1, 0.9, 0.0, 0.0], // a
            [0.1, 0.8, 0.1, 0.0], // a again: same character
            [0.9, 0.1, 0.0, 0.0], // blank
            [0.0, 0.7, 0.3, 0.0], // a after a blank: a second a
            [0.0, 0.0, 0.2, 0.8], // space
            [0.0, 0.1, 0.9, 0.0], // b
            [1.0, 0.0, 0.0, 0.0], // blank
            [1.0, 0.0, 0.0, 0.0], // blank
        ];
        let (text, confidence) = decode(&steps.concat(), &characters);
        assert_eq!(text, "aa b");
        assert!((confidence - (0.9 + 0.7 + 0.8 + 0.9) / 4.0).abs() < 1e-6);
    }

    #[test]
    fn nothing_but_blanks_decodes_to_nothing() {
        let characters = ["a".to_owned()];
        assert_eq!(decode(&[1.0, 0.0, 0.9, 0.1], &characters), (String::new(), 0.0));
    }
}
