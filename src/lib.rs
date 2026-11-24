//! OCR for frames whose text area is framed by four ArUco / AprilTag markers.
//!
//! Every frame goes through the same steps: [`markers`] finds the framed area, [`rectify`] warps it into an
//! upright page, [`ocr`] reads the page and [`layout`] puts the text into reading order. [`Pipeline`] chains
//! them and drops a frame whose text repeats the previous one.

pub mod dedupe;
pub mod layout;
pub mod markers;
pub mod ocr;
pub mod rectify;

use anyhow::{Result, ensure};
use opencv::core::{self, Mat, Size};

use dedupe::Dedupe;
use markers::Locator;
use ocr::{Models, Ocr};
use rectify::Rectifier;

/// Where the image warping and the neural networks run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Device {
    /// The first CUDA GPU
    Cuda,
    /// The CPU only, for machines without a CUDA GPU and for tests
    Cpu,
}

impl Device {
    /// Fails for [`Device::Cuda`] when OpenCV cannot use a GPU, rather than letting it fall back to the CPU.
    pub fn ensure_available(self) -> Result<()> {
        if self == Device::Cuda {
            ensure!(
                core::get_cuda_enabled_device_count()? > 0,
                "no CUDA device is available to OpenCV (use --device cpu to run without one)"
            );
        }
        Ok(())
    }
}

/// How the frames are turned into text, see the command line options of the same names.
pub struct Options {
    pub device: Device,
    pub dictionary: String,
    /// Marker ids at the top-left, top-right, bottom-right and bottom-left corner of the text area
    pub marker_ids: [i32; 4],
    /// Size of the rectified page, by default the size the area has in the first frame that shows it
    pub page_size: Option<Size>,
    /// Texts at least this similar (0..=1) count as the same content, 1 means identical
    pub min_similarity: f64,
    pub models: Models,
}

/// What became of one frame.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Not all four markers were found
    NoMarkers,
    /// The framed area contains no readable text
    NoText,
    /// Same text as the previous frame that had any
    Duplicate,
    /// New content, in reading order with one line of text per line
    Text(String),
}

pub struct Pipeline {
    device: Device,
    page_size: Option<Size>,
    locator: Locator,
    rectifier: Option<Rectifier>,
    ocr: Ocr,
    dedupe: Dedupe,
    page: Mat,
}

impl Pipeline {
    pub fn new(options: Options) -> Result<Self> {
        options.device.ensure_available()?;
        Ok(Self {
            device: options.device,
            page_size: options.page_size,
            locator: Locator::new(&options.dictionary, options.marker_ids)?,
            rectifier: None,
            ocr: Ocr::new(options.device, &options.models)?,
            dedupe: Dedupe::new(options.min_similarity),
            page: Mat::default(),
        })
    }

    /// Reads one BGR frame.
    pub fn process(&mut self, frame: &Mat) -> Result<Outcome> {
        let Some(quad) = self.locator.locate(frame)? else {
            return Ok(Outcome::NoMarkers);
        };
        // The page size is settled once: a size that follows the markers from frame to frame would make the
        // networks see a new input shape (and set themselves up again) every time the camera moves a little.
        let rectifier = match &mut self.rectifier {
            Some(rectifier) => rectifier,
            slot => {
                let size = self.page_size.unwrap_or_else(|| rectify::natural_size(&quad));
                slot.insert(Rectifier::new(self.device, size)?)
            }
        };
        rectifier.rectify(frame, &quad, &mut self.page)?;
        let text = layout::page_text(&self.ocr.read(&self.page)?);
        Ok(if text.is_empty() {
            Outcome::NoText
        } else if self.dedupe.is_new(&text) {
            Outcome::Text(text)
        } else {
            Outcome::Duplicate
        })
    }

    /// The rectified page of the frame processed last (empty before the first frame with markers).
    pub fn last_page(&self) -> &Mat {
        &self.page
    }
}
