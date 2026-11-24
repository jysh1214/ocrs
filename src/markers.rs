//! Finds the text area from the four fiducial markers around it.

use anyhow::{Context, Result, anyhow, ensure};
use opencv::core::{Mat, Point2f, Vector};
use opencv::objdetect::{
    self, ArucoDetector, CornerRefineMethod, DetectorParameters, PredefinedDictionaryType as Dictionary,
    RefineParameters,
};
use opencv::prelude::*;

/// Corners of a quadrilateral in image coordinates: top-left, top-right, bottom-right, bottom-left.
pub type Quad = [Point2f; 4];

/// OpenCV's predefined dictionaries by their name without the `DICT_` prefix.
const DICTIONARIES: &[(&str, Dictionary)] = &[
    ("4X4_50", Dictionary::DICT_4X4_50),
    ("4X4_100", Dictionary::DICT_4X4_100),
    ("4X4_250", Dictionary::DICT_4X4_250),
    ("4X4_1000", Dictionary::DICT_4X4_1000),
    ("5X5_50", Dictionary::DICT_5X5_50),
    ("5X5_100", Dictionary::DICT_5X5_100),
    ("5X5_250", Dictionary::DICT_5X5_250),
    ("5X5_1000", Dictionary::DICT_5X5_1000),
    ("6X6_50", Dictionary::DICT_6X6_50),
    ("6X6_100", Dictionary::DICT_6X6_100),
    ("6X6_250", Dictionary::DICT_6X6_250),
    ("6X6_1000", Dictionary::DICT_6X6_1000),
    ("7X7_50", Dictionary::DICT_7X7_50),
    ("7X7_100", Dictionary::DICT_7X7_100),
    ("7X7_250", Dictionary::DICT_7X7_250),
    ("7X7_1000", Dictionary::DICT_7X7_1000),
    ("ARUCO_ORIGINAL", Dictionary::DICT_ARUCO_ORIGINAL),
    ("ARUCO_MIP_36h12", Dictionary::DICT_ARUCO_MIP_36h12),
    ("APRILTAG_16h5", Dictionary::DICT_APRILTAG_16h5),
    ("APRILTAG_25h9", Dictionary::DICT_APRILTAG_25h9),
    ("APRILTAG_36h10", Dictionary::DICT_APRILTAG_36h10),
    ("APRILTAG_36h11", Dictionary::DICT_APRILTAG_36h11),
];

/// Looks a dictionary up by name; case and the `DICT_` prefix don't matter (`4x4_50`, `DICT_APRILTAG_36h11`).
pub fn parse_dictionary(name: &str) -> Result<Dictionary> {
    let wanted = name.trim();
    let wanted = match wanted.get(..5) {
        Some(prefix) if prefix.eq_ignore_ascii_case("DICT_") => &wanted[5..],
        _ => wanted,
    };
    DICTIONARIES
        .iter()
        .find(|(known, _)| known.eq_ignore_ascii_case(wanted))
        .map(|&(_, dictionary)| dictionary)
        .with_context(|| {
            let known = DICTIONARIES.iter().map(|(name, _)| *name).collect::<Vec<_>>();
            format!(
                "unknown marker dictionary {name:?}, expected one of: {}",
                known.join(", ")
            )
        })
}

/// The ids of the four markers as they are given on a command line: `TL,TR,BR,BL`.
pub fn parse_ids(text: &str) -> Result<[i32; 4]> {
    let ids = text
        .split(',')
        .map(|id| {
            id.trim()
                .parse::<i32>()
                .map_err(|_| anyhow!("{id:?} is not a marker id, expected four numbers as TL,TR,BR,BL"))
        })
        .collect::<Result<Vec<_>>>()?;
    let ids: [i32; 4] = ids
        .try_into()
        .map_err(|ids: Vec<i32>| anyhow!("expected four marker ids as TL,TR,BR,BL, got {}", ids.len()))?;
    ensure_distinct(&ids)?;
    Ok(ids)
}

/// The four markers of a text area are told apart by id alone, so their ids have to differ.
fn ensure_distinct(ids: &[i32; 4]) -> Result<()> {
    for (index, id) in ids.iter().enumerate() {
        ensure!(
            !ids[..index].contains(id),
            "the four marker ids must differ, {id} is given twice"
        );
    }
    Ok(())
}

/// Width in modules of the black border around a marker's bits, the one the detector expects by default.
const BORDER_MODULES: i32 = 1;

/// One marker as a square grid of black and white modules, its border included.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pattern {
    side: usize,
    dark: Vec<bool>,
}

impl Pattern {
    /// Modules along one side, border included: 6 for a 4x4 dictionary.
    pub fn side(&self) -> usize {
        self.side
    }

    /// Whether the module in `row` and `column`, counted from the top-left corner, is black.
    pub fn is_dark(&self, row: usize, column: usize) -> bool {
        self.dark[row * self.side + column]
    }
}

/// The marker `id` of a dictionary, to draw it on something that is not an image.
pub fn pattern(dictionary: &str, id: i32) -> Result<Pattern> {
    let predefined = objdetect::get_predefined_dictionary(parse_dictionary(dictionary)?)?;
    let count = predefined.bytes_list().rows();
    ensure!(
        (0..count).contains(&id),
        "dictionary {dictionary} has the marker ids 0 to {}, not {id}",
        count - 1
    );
    let side = predefined.marker_size() + 2 * BORDER_MODULES;
    // drawn at one pixel per module, the image is the grid of modules
    let mut image = Mat::default();
    objdetect::generate_image_marker(&predefined, id, side, &mut image, BORDER_MODULES)?;
    let mut dark = Vec::with_capacity((side * side) as usize);
    for row in 0..side {
        for column in 0..side {
            dark.push(*image.at_2d::<u8>(row, column)? < 128);
        }
    }
    Ok(Pattern {
        side: side as usize,
        dark,
    })
}

/// What a chart of `ocrs-show` says about the screen it is on, with a fifth marker between the top two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signal {
    /// The start screen: the pages are still to come
    Ready,
    /// A page. Pages that follow each other have numbers that follow each other, and after
    /// `PAGE_NUMBERS - 1` comes 0 again.
    Page(usize),
    /// The end screen: all pages have been shown
    Done,
}

/// How many page numbers there are before they start over.
pub const PAGE_NUMBERS: usize = 16;

/// The marker ids that stand for the signals: the first ones of the dictionary that the corners of the
/// text area don't use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signals {
    ready: i32,
    done: i32,
    pages: [i32; PAGE_NUMBERS],
}

impl Signals {
    /// `corner_ids` are the markers around the text area, which leave the others for signals.
    pub fn new(dictionary: &str, corner_ids: [i32; 4]) -> Result<Self> {
        let predefined = objdetect::get_predefined_dictionary(parse_dictionary(dictionary)?)?;
        let mut free = (0..predefined.bytes_list().rows()).filter(|id| !corner_ids.contains(id));
        let mut next = || {
            free.next()
                .with_context(|| format!("dictionary {dictionary} has too few markers for signals"))
        };
        let (ready, done) = (next()?, next()?);
        let mut pages = [0; PAGE_NUMBERS];
        for page in &mut pages {
            *page = next()?;
        }
        Ok(Self { ready, done, pages })
    }

    /// The id of the marker that stands for `signal`.
    pub fn id(&self, signal: Signal) -> i32 {
        match signal {
            Signal::Ready => self.ready,
            Signal::Done => self.done,
            Signal::Page(number) => self.pages[number % PAGE_NUMBERS],
        }
    }

    /// What the marker `id` stands for, if it is a signal.
    pub fn signal(&self, id: i32) -> Option<Signal> {
        if id == self.ready {
            Some(Signal::Ready)
        } else if id == self.done {
            Some(Signal::Done)
        } else {
            self.pages.iter().position(|page| *page == id).map(Signal::Page)
        }
    }
}

/// What a frame shows of a text area.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sighting {
    /// The area enclosed by the four markers
    pub area: Quad,
    /// What the chart says about itself, if it is one of `ocrs-show` and says one thing only: a frame
    /// taken while the screen changes can show two signals on top of each other
    pub signal: Option<Signal>,
}

/// The corners of a marker in a frame.
pub type Outline = [Point2f; 4];

/// Everything that is made out of markers in a frame: more than is needed to find the text area, for
/// showing somebody who points a camera what it takes for what.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Survey {
    /// The markers at the top-left, top-right, bottom-right and bottom-left corner of the text area, as
    /// far as they are found
    pub corners: [Option<Outline>; 4],
    /// The marker that says what a chart of `ocrs-show` has on show, if there is exactly one
    pub signal: Option<(Signal, Outline)>,
    /// Other markers of the dictionary, with their ids
    pub strangers: Vec<(i32, Outline)>,
    /// Squares that look like markers and can't be read as one: markers that are too small, blurred or
    /// outshone, and whatever else is dark and square
    pub unread: Vec<Outline>,
}

impl Survey {
    /// The text area, if all four of its markers are found.
    pub fn sighting(&self) -> Option<Sighting> {
        let [
            Some(top_left),
            Some(top_right),
            Some(bottom_right),
            Some(bottom_left),
        ] = self.corners
        else {
            return None;
        };
        Some(Sighting {
            area: enclosed_area(&[top_left, top_right, bottom_right, bottom_left]),
            signal: self.signal.map(|(signal, _)| signal),
        })
    }
}

/// Detects the four markers and reports the area they enclose.
pub struct Locator {
    detector: ArucoDetector,
    ids: [i32; 4],
    signals: Signals,
}

impl Locator {
    /// `ids` are the markers at the top-left, top-right, bottom-right and bottom-left corner of the area.
    /// Telling them apart by id is what gives the page its orientation, so they have to differ.
    pub fn new(dictionary: &str, ids: [i32; 4]) -> Result<Self> {
        ensure_distinct(&ids)?;
        let signals = Signals::new(dictionary, ids)?;
        let dictionary = objdetect::get_predefined_dictionary(parse_dictionary(dictionary)?)?;
        let mut parameters = DetectorParameters::default()?;
        // marker corners end up as page corners, so they are worth the sub-pixel pass
        parameters.set_corner_refinement_method(CornerRefineMethod::CORNER_REFINE_SUBPIX as i32);
        let detector = ArucoDetector::new(&dictionary, &parameters, RefineParameters::new_def()?)?;
        Ok(Self {
            detector,
            ids,
            signals,
        })
    }

    /// The area enclosed by the markers, or `None` unless all four are visible in `frame`.
    pub fn locate(&self, frame: &Mat) -> Result<Option<Sighting>> {
        Ok(self.survey(frame)?.sighting())
    }

    /// Everything that can be made out of markers in `frame`.
    pub fn survey(&self, frame: &Mat) -> Result<Survey> {
        let mut outlines = Vector::<Vector<Point2f>>::new();
        let mut found = Vector::<i32>::new();
        let mut rejected = Vector::<Vector<Point2f>>::new();
        self.detector
            .detect_markers(frame, &mut outlines, &mut found, &mut rejected)?;

        let outline = |points: Vector<Point2f>| -> Result<Outline> {
            ensure!(
                points.len() == 4,
                "a marker was reported with {} corners",
                points.len()
            );
            Ok(std::array::from_fn(|corner| {
                points.get(corner).expect("four corners")
            }))
        };
        let mut survey = Survey::default();
        let mut signals = Vec::new();
        for (id, points) in found.iter().zip(&outlines) {
            let outline = outline(points)?;
            if let Some(corner) = self.ids.iter().position(|own| *own == id) {
                // the first of two markers with the same id
                survey.corners[corner].get_or_insert(outline);
            } else if let Some(signal) = self.signals.signal(id) {
                signals.push((signal, id, outline));
            } else {
                survey.strangers.push((id, outline));
            }
        }
        match signals.as_slice() {
            [(signal, _, outline)] => survey.signal = Some((*signal, *outline)),
            // two signals at once say nothing
            several => survey
                .strangers
                .extend(several.iter().map(|(_, id, outline)| (*id, *outline))),
        }
        // the white patches inside a marker are squares of their own to the detector, and no news
        let read = survey
            .corners
            .iter()
            .flatten()
            .chain(survey.signal.iter().map(|(_, outline)| outline))
            .chain(survey.strangers.iter().map(|(_, outline)| outline))
            .copied()
            .collect::<Vec<_>>();
        for points in &rejected {
            let outline = outline(points)?;
            let middle_x = outline.iter().map(|corner| corner.x).sum::<f32>() / 4.0;
            let middle_y = outline.iter().map(|corner| corner.y).sum::<f32>() / 4.0;
            let inside = |marker: &Outline| {
                let (xs, ys) = (marker.map(|corner| corner.x), marker.map(|corner| corner.y));
                let within = |value: f32, range: [f32; 4]| {
                    value >= range.into_iter().fold(f32::INFINITY, f32::min)
                        && value <= range.into_iter().fold(f32::NEG_INFINITY, f32::max)
                };
                within(middle_x, xs) && within(middle_y, ys)
            };
            if !read.iter().any(inside) {
                survey.unread.push(outline);
            }
        }
        Ok(survey)
    }
}

/// Of each marker outline, the corner nearest to the middle of all four markers. With the markers sitting
/// just outside the corners of the text area, those are the corners of that area, however the individual
/// markers are rotated.
fn enclosed_area(markers: &[[Point2f; 4]; 4]) -> Quad {
    let count = (markers.len() * 4) as f32;
    let middle_x = markers.iter().flatten().map(|p| p.x).sum::<f32>() / count;
    let middle_y = markers.iter().flatten().map(|p| p.y).sum::<f32>() / count;
    let distance = |p: &Point2f| (p.x - middle_x).powi(2) + (p.y - middle_y).powi(2);
    markers.map(|outline| {
        outline
            .into_iter()
            .min_by(|a, b| distance(a).total_cmp(&distance(b)))
            .expect("a marker outline has four corners")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_names_ignore_case_and_prefix() {
        assert_eq!(parse_dictionary("4x4_50").unwrap(), Dictionary::DICT_4X4_50);
        assert_eq!(
            parse_dictionary("DICT_6X6_250").unwrap(),
            Dictionary::DICT_6X6_250
        );
        assert_eq!(
            parse_dictionary(" dict_apriltag_36H11 ").unwrap(),
            Dictionary::DICT_APRILTAG_36h11
        );
        let error = parse_dictionary("9x9").unwrap_err().to_string();
        assert!(error.contains("APRILTAG_36h11"), "{error}");
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        assert!(Locator::new("4x4_50", [0, 1, 2, 1]).is_err());
    }

    #[test]
    fn ids_are_four_different_numbers_between_commas() {
        assert_eq!(parse_ids("10,11,12,13").unwrap(), [10, 11, 12, 13]);
        assert_eq!(parse_ids(" 3, 2,1 ,0").unwrap(), [3, 2, 1, 0]);
        for (wrong, reason) in [
            ("1,2,3", "got 3"),
            ("1,2,3,4,5", "got 5"),
            ("1,2,3,x", "\"x\" is not a marker id"),
            ("1 2 3 4", "is not a marker id"),
            ("", "is not a marker id"),
            ("1,2,3,1", "1 is given twice"),
        ] {
            let error = parse_ids(wrong).unwrap_err().to_string();
            assert!(error.contains(reason), "{wrong:?}: {error}");
        }
    }

    #[test]
    fn signals_take_the_ids_that_the_corners_leave() -> Result<()> {
        let signals = Signals::new("4x4_50", [0, 1, 2, 3])?;
        assert_eq!(signals.id(Signal::Ready), 4);
        assert_eq!(signals.id(Signal::Done), 5);
        assert_eq!(signals.id(Signal::Page(0)), 6);
        assert_eq!(signals.id(Signal::Page(15)), 21);
        // page numbers start over
        assert_eq!(signals.id(Signal::Page(16)), 6);
        assert_eq!(signals.signal(21), Some(Signal::Page(15)));
        assert_eq!(signals.signal(4), Some(Signal::Ready));
        assert_eq!(signals.signal(5), Some(Signal::Done));
        for no_signal in [0, 3, 22, 49] {
            assert_eq!(signals.signal(no_signal), None);
        }

        let signals = Signals::new("apriltag_16h5", [5, 0, 7, 2])?;
        assert_eq!(signals.id(Signal::Ready), 1);
        assert_eq!(signals.id(Signal::Done), 3);
        assert_eq!(signals.id(Signal::Page(0)), 4);
        assert_eq!(signals.id(Signal::Page(1)), 6);
        assert_eq!(signals.id(Signal::Page(2)), 8);
        Ok(())
    }

    #[test]
    fn pattern_is_the_bits_inside_a_black_border() -> Result<()> {
        let marker = pattern("4x4_50", 3)?;
        assert_eq!(marker.side(), 6);
        for at in 0..6 {
            for (row, column) in [(0, at), (5, at), (at, 0), (at, 5)] {
                assert!(marker.is_dark(row, column), "border module {row},{column}");
            }
        }
        let white = (1..5)
            .flat_map(|row| (1..5).map(move |column| (row, column)))
            .filter(|&(row, column)| !marker.is_dark(row, column))
            .count();
        assert!(
            (1..16).contains(&white),
            "{white} white modules among the 16 bits"
        );
        assert_ne!(marker, pattern("4x4_50", 4)?);
        assert_eq!(pattern("apriltag_36h11", 0)?.side(), 8);
        Ok(())
    }

    #[test]
    fn pattern_of_an_id_outside_the_dictionary_is_an_error() {
        let error = pattern("4x4_50", 50).unwrap_err().to_string();
        assert!(error.contains("0 to 49"), "{error}");
        assert!(pattern("4x4_50", -1).is_err());
    }

    /// A square marker outline, listed clockwise starting from the corner that `start` selects.
    fn outline(x: f32, y: f32, side: f32, start: usize) -> [Point2f; 4] {
        let square = [(x, y), (x + side, y), (x + side, y + side), (x, y + side)];
        std::array::from_fn(|i| {
            let (px, py) = square[(i + start) % 4];
            Point2f::new(px, py)
        })
    }

    #[test]
    fn enclosed_area_uses_the_corners_facing_the_middle() {
        // markers of side 10 around the area (10,10)-(90,60), each listed from a different starting corner
        let markers = [
            outline(0.0, 0.0, 10.0, 0),
            outline(90.0, 0.0, 10.0, 1),
            outline(90.0, 60.0, 10.0, 2),
            outline(0.0, 60.0, 10.0, 3),
        ];
        let expected =
            [(10.0, 10.0), (90.0, 10.0), (90.0, 60.0), (10.0, 60.0)].map(|(x, y)| Point2f::new(x, y));
        assert_eq!(enclosed_area(&markers), expected);
    }
}
