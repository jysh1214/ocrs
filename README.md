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
ocrs 0 -o live.txt --camera-size 1920x1080                    # camera 0
ocrs 0 -o live.txt --dict apriltag_36h11 --ids 10,11,12,13    # other markers
ocrs 'scan_%04d.png' -o scans.txt --device cpu                # image sequence, no GPU
```

| Option | Default | |
| --- | --- | --- |
| `--dict` | `4x4_50` | marker dictionary, any of OpenCV's predefined ones (`6x6_250`, `apriltag_36h11`, ...) |
| `--ids` | `0,1,2,3` | ids of the markers at the top-left, top-right, bottom-right, bottom-left corner |
| `--device` | `cuda` | `cpu` runs everything without a GPU (slow, meant for testing) |
| `--camera-size` | the camera's own | frame size to ask a camera for, `WIDTHxHEIGHT`; see below |
| `--page-size` | from the first frame | size in pixels the text area is rectified to, `WIDTHxHEIGHT` |
| `--similarity` | `0.9` | how alike two consecutive texts must be to count as the same content |
| `--separator` | none | a line to write between the texts of two frames |
| `--dump-pages` | off | directory to save the rectified text area of every recorded frame to |
| `--det-model`, `--rec-model`, `--rec-dict` | built in | other PP-OCR models, see [models/README.md](models/README.md) |

### Cameras

A camera delivers the size it starts up with, which is 640x480 for nearly all of them and too little for a
page of text. `--camera-size 1920x1080` asks for more. `ocrs` says which size the frames have when it starts;
if the camera does not have the size asked for, that is the nearest one it has, and
`v4l2-ctl --list-formats-ext` lists them all. USB cameras tend to have their large sizes as Motion JPEG
only, so `ocrs` switches to that where the size asked for is not to be had uncompressed.

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

## Test target: `ocrs-show`

`ocrs-show` puts a text file between four markers on the terminal it runs in. With a camera pointed at that
terminal the whole chain can be tried out: what `ocrs` writes should be the text that is on show.

```sh
ocrs-show tests/sample.txt                    # in a terminal the camera sees
ocrs 0 -o read.txt --camera-size 1920x1080    # in another one
```

`tests/sample.txt` is a text for this: some 1,900 words of prose with tables, figures, every printable ASCII
character and the characters that are easily taken for one another.

The markers are drawn with half block characters in the corners of the terminal, and the text goes into the
area between them, one page at a time. The text has the size of the terminal's font: change it there (in
Ghostty with `ctrl` `+` and `ctrl` `-`) and the layout follows, as it does when the window changes size. The
rest is set with options, and most of it with keys as well while the chart is on show:

| Option | Keys | Default | |
| --- | --- | --- | --- |
| `--module` | `+` `-` | `2` | size of a marker module: this many columns wide and half as many rows high |
| `--margin` | `}` `{` | `1` | blank space between the edges of the terminal and the markers, in modules |
| `--pad` | `]` `[` | `1` | blank space around the text: this many rows and twice as many columns |
| `--line-spacing` | `.` `,` | `0` | empty rows between two lines of text |
| `--invert` | `i` | off | light text on a dark panel |
| `--bold` | `b` | off | bold text |
| `--wrap` | `w` | `word` | `none` cuts lines that are too long off instead of continuing them on the next row |
| `--no-notes` | `h` | shown | the keys and the measurements above and below the text area, which `ocrs` does not read |
| `--ink`, `--paper` | | `000000`, `ffffff` | the dark and the light colour as `RRGGBB`; a grey paper dims a screen that glares |
| `--interval` | | off | seconds after which the next page comes up by itself, the first one after the last |
| `--dict`, `--ids` | | `4x4_50`, `0,1,2,3` | the markers, to be given to `ocrs` the same way |

`space` and `p` turn the pages; a form feed character in the file starts a new one. `q` quits and prints the
options for the layout as it was then, along with what the terminal measured, so that a setup that read well
can be noted down and shown again:

```
ocrs-show notes.txt --module 3 --margin 1 --pad 2 --line-spacing 1
terminal 120x40 cells, cell 10x20 px, marker 18x9 cells = 180x180 px, text 70x14 cells, 7 lines a page
```

How large the text has to be is a matter of the camera: what counts is how many pixels of its frames a
column of the terminal gets, which is the width of the frame, times the share of it that the terminal fills,
divided by the columns of the terminal. Screens rendered as clean images are read nearly without errors
down to 6 pixels a column. A camera adds noise, blur and glare to that, so 10 or more are a better aim: no
more than some 150 columns across a frame of 1920 pixels, and no more than 50 across one of 640.
`--dump-pages` shows what `ocrs` gets to read.

The markers need blank space around them to be found. The margin is that space towards a dark window frame
or screen bezel, the padding towards the text. Without padding the text touches the corners of the markers
and the text area is located a pixel or two off.

## Tests

`cargo test` runs on the CPU and does not touch the GPU: unit tests for the pure logic, `tests/pipeline.rs`
which reads synthetic frames end to end, and the tests of `ocrs-show`, which have its screens found and
read as images. The one test that sends frames through the CUDA code paths is ignored by default:

```sh
cargo test -- --ignored
```
