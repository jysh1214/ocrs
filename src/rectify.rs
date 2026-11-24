//! Warps the area between the markers into an upright page.

use anyhow::Result;
use opencv::core::{self, BORDER_CONSTANT, GpuMat, Mat, Point2f, Scalar, Size, Stream, Vector};
use opencv::prelude::*;
use opencv::{cudawarping, imgproc};

use crate::Device;
use crate::markers::Quad;

/// Page size that keeps the resolution the area has in the frame: its longer horizontal and vertical edge.
pub fn natural_size(quad: &Quad) -> Size {
    let length = |a: Point2f, b: Point2f| (a.x - b.x).hypot(a.y - b.y);
    let [tl, tr, br, bl] = *quad;
    let width = length(tl, tr).max(length(bl, br));
    let height = length(tl, bl).max(length(tr, br));
    Size::new((width.round() as i32).max(1), (height.round() as i32).max(1))
}

pub struct Rectifier {
    size: Size,
    /// Frame and page on the GPU, kept between frames so their memory is reused
    gpu: Option<(GpuMat, GpuMat)>,
}

impl Rectifier {
    pub fn new(device: Device, size: Size) -> Result<Self> {
        let gpu = match device {
            Device::Cuda => Some((GpuMat::new_def()?, GpuMat::new_def()?)),
            Device::Cpu => None,
        };
        Ok(Self { size, gpu })
    }

    /// Fills `page` with the part of `frame` inside `quad`, stretched to the page size.
    pub fn rectify(&mut self, frame: &Mat, quad: &Quad, page: &mut Mat) -> Result<()> {
        let (right, bottom) = ((self.size.width - 1) as f32, (self.size.height - 1) as f32);
        let corners =
            [(0.0, 0.0), (right, 0.0), (right, bottom), (0.0, bottom)].map(|(x, y)| Point2f::new(x, y));
        let transform = imgproc::get_perspective_transform_def(
            &Vector::<Point2f>::from_slice(quad),
            &Vector::<Point2f>::from_slice(&corners),
        )?;
        // anything the frame doesn't cover is paper
        let white = Scalar::all(255.0);
        match &mut self.gpu {
            Some((gpu_frame, gpu_page)) => {
                gpu_frame.upload(frame)?;
                cudawarping::warp_perspective(
                    gpu_frame,
                    gpu_page,
                    &transform,
                    self.size,
                    imgproc::INTER_LINEAR,
                    BORDER_CONSTANT,
                    white,
                    &mut Stream::null()?,
                )?;
                gpu_page.download(page)?;
            }
            None => imgproc::warp_perspective(
                frame,
                page,
                &transform,
                self.size,
                imgproc::INTER_LINEAR,
                core::BORDER_CONSTANT,
                white,
            )?,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_size_takes_the_longer_edges() {
        let quad = [(10.0, 10.0), (110.0, 12.0), (118.0, 70.0), (8.0, 62.0)].map(|(x, y)| Point2f::new(x, y));
        // top edge ~100, bottom edge ~110.3; left edge ~52, right edge ~58.5
        assert_eq!(natural_size(&quad), Size::new(110, 59));
    }

    #[test]
    fn rectified_page_shows_the_quad_upright() -> Result<()> {
        // a frame that is black except for a white rectangle, which is the quad to extract
        let mut frame = Mat::new_rows_cols_with_default(120, 160, core::CV_8UC3, Scalar::all(0.0))?;
        let area = core::Rect::new(40, 30, 80, 50);
        frame
            .roi_mut(area)?
            .set_to(&Scalar::all(255.0), &core::no_array())?;
        let quad =
            [(40.0, 30.0), (119.0, 30.0), (119.0, 79.0), (40.0, 79.0)].map(|(x, y)| Point2f::new(x, y));

        let mut page = Mat::default();
        Rectifier::new(Device::Cpu, Size::new(80, 50))?.rectify(&frame, &quad, &mut page)?;
        assert_eq!(page.size()?, Size::new(80, 50));
        let (mut darkest, mut brightest) = (0.0, 0.0);
        core::min_max_loc(
            &page.reshape(1, 0)?,
            Some(&mut darkest),
            Some(&mut brightest),
            None,
            None,
            &core::no_array(),
        )?;
        assert_eq!((darkest, brightest), (255.0, 255.0));
        Ok(())
    }
}
