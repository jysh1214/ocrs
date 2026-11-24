# ocrs

Reads the text between four ArUco / AprilTag markers in every frame of a video and appends it to a text file.
A frame that shows the same text as the frame before it is skipped. Written in Rust on top of OpenCV, with the
image warping and the neural networks running on CUDA.

Per frame:

1. find the four markers (`objdetect::ArucoDetector`) and take the area they enclose;
2. warp that area into an upright page (`cuda::warpPerspective`);
3. detect the text lines and read each of them (PP-OCRv5 models on OpenCV DNN's CUDA backend);
4. put the lines into reading order and compare the text with the previous frame's;
5. if it is new, append it to the output file and flush.

## Building

OpenCV is not taken from the system. `cargo build` compiles the checkout in `third_party/opencv` (with the
CUDA modules from `third_party/opencv_contrib`) and builds the Rust bindings against it, see
[third_party/README.md](third_party/README.md).

Needed on the machine:

- CUDA toolkit and cuDNN (Arch: `cuda`, `cudnn`)
- FFmpeg development files, for reading video files
- `cmake` on `PATH`, and `ninja` if available
- `clang` on `PATH` plus its libclang, for generating the bindings

```sh
git submodule update --init --depth 1
cargo build --release
```

The first build compiles OpenCV, which takes several minutes; its log is
`third_party/_build/opencv/ocrs-build.log`. Later builds reuse `third_party/_build` and `third_party/_install`
(`cargo clean` leaves them alone, delete them to rebuild OpenCV from scratch).

The build does not notice a system update that replaces CUDA, cuDNN or FFmpeg. If `ocrs` stops starting after
one (`error while loading shared libraries: libavformat.so...`), delete those two directories and build again.

Environment variables read by the OpenCV build:

| Variable | Meaning |
| --- | --- |
| `CMAKE` | cmake executable, default `cmake` |
| `OCRS_CUDA_ARCH` | compute capabilities to generate CUDA code for, e.g. `8.9`; default: what `nvidia-smi` reports |
| `NVCC_CCBIN` | host C++ compiler for nvcc, when CUDA doesn't support the default one yet |
| `CMAKE_PREFIX_PATH` | extra prefixes to search, e.g. a cuDNN outside the system directories |
| `OCRS_OPENCV_CMAKE_ARGS` | extra arguments for OpenCV's CMake configuration |

## Usage

```sh
ocrs recording.mp4 -o recording.txt
ocrs 0 -o live.txt --dict apriltag_36h11 --ids 10,11,12,13    # camera 0
ocrs 'scan_%04d.png' -o scans.txt --device cpu                # image sequence, no GPU
```

| Option | Default | |
| --- | --- | --- |
| `--dict` | `4x4_50` | marker dictionary, any of OpenCV's predefined ones (`6x6_250`, `apriltag_36h11`, ...) |
| `--ids` | `0,1,2,3` | ids of the markers at the top-left, top-right, bottom-right, bottom-left corner |
| `--device` | `cuda` | `cpu` runs everything without a GPU (slow, meant for testing) |
| `--page-size` | from the first frame | size in pixels the text area is rectified to, `WIDTHxHEIGHT` |
| `--similarity` | `0.9` | how alike two consecutive texts must be to count as the same content |
| `--separator` | none | a line to write between the texts of two frames |
| `--dump-pages` | off | directory to save the rectified text area of every recorded frame to |
| `--det-model`, `--rec-model`, `--rec-dict` | built in | other PP-OCR models, see [models/README.md](models/README.md) |

### Markers

Four markers of one dictionary with four different ids sit just outside the corners of the text area:

```
[TL]                    [TR]
    +------------------+
    |  text area       |
    +------------------+
[BL]                    [BR]
```

The text area is spanned by the corner of each marker that faces the middle, so the markers themselves are not
part of it and may be rotated individually. Which marker is which is decided by id alone, so the page comes out
upright however the camera is turned. A frame is skipped unless all four markers are found.

### Output

The text of a recorded frame is written as one line per text line, top to bottom; pieces of text on the same
row are joined by a space. The file is flushed after every recorded frame.

`--similarity 0.9` treats two consecutive frames as the same content when at most a tenth of their characters
differ (edit distance relative to the longer text). With a camera, the reading of an unchanged page flickers by
a character now and then, and comparing for equality would record such a page many times. Use `--similarity 1`
to skip only identical texts.

## Tests

`cargo test` runs on the CPU and does not touch the GPU: unit tests for the pure logic, and
`tests/pipeline.rs` which reads synthetic frames end to end. The one test that sends the same frames through
the CUDA code paths is ignored by default:

```sh
cargo test -- --ignored
```
