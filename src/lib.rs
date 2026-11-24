//! OCR for frames whose text area is framed by four ArUco / AprilTag markers.
//!
//! Every frame goes through the same steps: [`markers`] finds the framed area, [`rectify`] warps it into an
//! upright page, [`ocr`] reads the page and [`layout`] puts the text into reading order. [`Pipeline`] chains
//! them, waits until the frames agree on what they read ([`consensus`]) and drops what repeats the text
//! before it ([`dedupe`]).

pub mod consensus;
pub mod dedupe;
pub mod layout;
pub mod markers;
pub mod ocr;
pub mod rectify;

use anyhow::{Result, ensure};
use opencv::core::{self, Mat, Size};

use consensus::Consensus;
use dedupe::Dedupe;
use markers::{Locator, Quad, Signal, Survey};
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
    /// How many frames have to agree on a text before it counts, see [`Consensus::new`]. 1 takes every
    /// frame at its word, which is right for pictures that each show another page.
    pub stable_frames: usize,
    /// Texts at least this similar (0..=1) count as the same content, 1 means identical
    pub min_similarity: f64,
    pub models: Models,
}

/// What became of one frame.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Not all four markers were found
    NoMarkers,
    /// The start screen of `ocrs-show`: the pages are still to come
    Ready,
    /// The end screen of `ocrs-show`: there is nothing more to read
    Done,
    /// The framed area contains no readable text
    NoText,
    /// There is text, but the recent frames don't agree on it; or the frame does not tell which page of
    /// `ocrs-show` it shows
    Unsettled,
    /// Same text as before, or a page of `ocrs-show` that has been reported
    Duplicate,
    /// New content, in reading order with one line of text per line
    Text(String),
    /// The likeliest reading of a page of `ocrs-show` that went away before enough frames agreed on it
    Guess(String),
}

/// The page of an `ocrs-show` chart that the last frames showed.
struct Shown {
    /// As in [`Signal::Page`]
    number: usize,
    /// Whether its text has been reported
    reported: bool,
}

pub struct Pipeline {
    device: Device,
    page_size: Option<Size>,
    locator: Locator,
    rectifier: Option<Rectifier>,
    ocr: Ocr,
    consensus: Consensus,
    dedupe: Dedupe,
    /// Whether a chart that says which page it shows has been seen. From then on the pages are told apart
    /// by what the chart says rather than by their text.
    signalled: bool,
    shown: Option<Shown>,
    survey: Survey,
    page: Mat,
    character_width: Option<f32>,
}

impl Pipeline {
    pub fn new(options: Options) -> Result<Self> {
        options.device.ensure_available()?;
        ensure!(
            options.stable_frames >= 1,
            "a text has to be read by one frame at least"
        );
        Ok(Self {
            device: options.device,
            page_size: options.page_size,
            locator: Locator::new(&options.dictionary, options.marker_ids)?,
            rectifier: None,
            ocr: Ocr::new(options.device, &options.models)?,
            consensus: Consensus::new(options.stable_frames, options.min_similarity),
            dedupe: Dedupe::new(options.min_similarity),
            signalled: false,
            shown: None,
            survey: Survey::default(),
            page: Mat::default(),
            character_width: None,
        })
    }

    /// Reads one BGR frame.
    pub fn process(&mut self, frame: &Mat) -> Result<Outcome> {
        self.survey = self.locator.survey(frame)?;
        let Some(sighting) = self.survey.sighting() else {
            return Ok(Outcome::NoMarkers);
        };
        let Some(signal) = sighting.signal else {
            if self.signalled {
                // a chart that tells its pages apart, in a frame that doesn't show which one this is
                return Ok(Outcome::Unsettled);
            }
            // any other text area: new content is what the frames agree on and what differs from before
            let text = self.read(frame, &sighting.area)?;
            let Some(settled) = self.consensus.push(text) else {
                return Ok(Outcome::Unsettled);
            };
            return Ok(if settled.is_empty() {
                Outcome::NoText
            } else if self.dedupe.is_new(settled) {
                Outcome::Text(settled.to_owned())
            } else {
                Outcome::Duplicate
            });
        };

        self.signalled = true;
        let number = match signal {
            Signal::Page(number) => Some(number),
            Signal::Ready | Signal::Done => None,
        };
        if self.shown.as_ref().map(|shown| shown.number) != number {
            // another screen than before: the page that was on show is over, read or not
            let guess = self.leave_page();
            self.shown = number.map(|number| Shown {
                number,
                reported: false,
            });
            if let Some(guess) = guess {
                return Ok(Outcome::Guess(guess));
            }
        }
        match (signal, &mut self.shown) {
            (Signal::Ready, _) => Ok(Outcome::Ready),
            (Signal::Done, _) => Ok(Outcome::Done),
            // nothing left to find out about this page
            (Signal::Page(_), Some(Shown { reported: true, .. })) => Ok(Outcome::Duplicate),
            (Signal::Page(_), _) => {
                let text = self.read(frame, &sighting.area)?;
                let Some(settled) = self.consensus.push(text) else {
                    return Ok(Outcome::Unsettled);
                };
                let outcome = if settled.is_empty() {
                    Outcome::NoText
                } else {
                    Outcome::Text(settled.to_owned())
                };
                if let Some(shown) = &mut self.shown {
                    shown.reported = true;
                }
                Ok(outcome)
            }
        }
    }

    /// To be called after the last frame: the likeliest reading of a page of `ocrs-show` that was on show
    /// to the end without enough frames agreeing on it.
    pub fn finish(&mut self) -> Option<String> {
        self.leave_page()
    }

    /// Ends the page of `ocrs-show` that is on show. Returns its likeliest reading if it has text and has
    /// not been reported.
    fn leave_page(&mut self) -> Option<String> {
        let unreported = self.shown.take().is_some_and(|shown| !shown.reported);
        let guess = self
            .consensus
            .likeliest()
            .filter(|text| unreported && !text.is_empty())
            .map(str::to_owned);
        self.consensus.clear();
        guess
    }

    /// The text in `area` of the frame, in reading order.
    fn read(&mut self, frame: &Mat, area: &Quad) -> Result<String> {
        // The page size is settled once: a size that follows the markers from frame to frame would make the
        // networks see a new input shape (and set themselves up again) every time the camera moves a little.
        let rectifier = match &mut self.rectifier {
            Some(rectifier) => rectifier,
            slot => {
                let size = self.page_size.unwrap_or_else(|| rectify::natural_size(area));
                slot.insert(Rectifier::new(self.device, size)?)
            }
        };
        rectifier.rectify(frame, area, &mut self.page)?;
        let boxes = self.ocr.read(&self.page)?;
        self.character_width = layout::character_width(&boxes);
        Ok(layout::page_text(&boxes))
    }

    /// How many pixels wide a character is on the page read last, see [`layout::character_width`]. The
    /// page has the size that the text area has in the frames unless another one was asked for, so these
    /// are pixels of the camera.
    pub fn last_character_width(&self) -> Option<f32> {
        self.character_width
    }

    /// The markers that were made out in the frame processed last.
    pub fn last_survey(&self) -> &Survey {
        &self.survey
    }

    /// The rectified page of the frame read last (empty before the first frame that was read).
    pub fn last_page(&self) -> &Mat {
        &self.page
    }
}
