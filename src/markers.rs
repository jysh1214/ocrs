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

/// Detects the four markers and reports the area they enclose.
pub struct Locator {
    detector: ArucoDetector,
    ids: [i32; 4],
}

impl Locator {
    /// `ids` are the markers at the top-left, top-right, bottom-right and bottom-left corner of the area.
    /// Telling them apart by id is what gives the page its orientation, so they have to differ.
    pub fn new(dictionary: &str, ids: [i32; 4]) -> Result<Self> {
        ensure_distinct(&ids)?;
        let dictionary = objdetect::get_predefined_dictionary(parse_dictionary(dictionary)?)?;
        let mut parameters = DetectorParameters::default()?;
        // marker corners end up as page corners, so they are worth the sub-pixel pass
        parameters.set_corner_refinement_method(CornerRefineMethod::CORNER_REFINE_SUBPIX as i32);
        let detector = ArucoDetector::new(&dictionary, &parameters, RefineParameters::new_def()?)?;
        Ok(Self { detector, ids })
    }

    /// The area enclosed by the markers, or `None` unless all four are visible in `frame`.
    pub fn locate(&self, frame: &Mat) -> Result<Option<Quad>> {
        let mut corners = Vector::<Vector<Point2f>>::new();
        let mut found = Vector::<i32>::new();
        self.detector
            .detect_markers_def(frame, &mut corners, &mut found)?;

        let mut markers = [[Point2f::default(); 4]; 4];
        for (marker, id) in markers.iter_mut().zip(self.ids) {
            let Some(index) = found.iter().position(|candidate| candidate == id) else {
                return Ok(None);
            };
            let outline = corners.get(index)?;
            ensure!(
                outline.len() == 4,
                "marker {id} was reported with {} corners",
                outline.len()
            );
            for (corner, point) in marker.iter_mut().zip(&outline) {
                *corner = point;
            }
        }
        Ok(Some(enclosed_area(&markers)))
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
