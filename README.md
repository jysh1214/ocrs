# ocrs

Reads the text between four ArUco / AprilTag markers in every frame of a video and appends it to a text file.
A text is recorded once, when most of the recent frames agree on it. Written in Rust on top of OpenCV, with
the image warping and the neural networks running on CUDA.

Per frame:

1. find the four markers (`objdetect::ArucoDetector`) and take the area they enclose;
2. warp that area into an upright page (`cuda::warpPerspective`);
3. detect the text lines and read each of them (PP-OCRv6 models on OpenCV DNN's CUDA backend);
4. put the lines into reading order and wait until most of the recent frames read that text;
5. if it differs from the text recorded before, append it to the output file and flush.

## Building

OpenCV is not taken from the system. `cargo build` compiles the checkout in `third_party/opencv` (with the
CUDA modules from `third_party/opencv_contrib`) and builds the Rust bindings against it, see
[third_party/README.md](third_party/README.md).

Needed on the machine:

- CUDA toolkit and cuDNN (Arch: `cuda`, `cudnn`), unless CUDA is left out, see [below](#without-cuda)
- FFmpeg development files, for reading video files
- `cmake` on `PATH`, and `ninja` if available
- `clang` on `PATH` plus its libclang, for generating the bindings
- Python with `onnx` and `numpy`, for fetching the models

```sh
git submodule update --init --depth 1
python tools/download_models.py
cargo build --release
```

The models are not in the repository. `tools/download_models.py` fetches them into `model/`, some 31 MB, and
the build compiles them into `ocrs`, see [model/README.md](model/README.md). A build without them succeeds
with a warning: its `ocrs` has to be told its models with `--det-model`, `--rec-model` and `--rec-dict`, and
its `ocrs-show` lacks nothing.

The first build compiles OpenCV, which takes several minutes; its log is
`third_party/_build/opencv/ocrs-build.log`. Later builds reuse `third_party/_build` and `third_party/_install`
(`cargo clean` leaves them alone, delete them to rebuild OpenCV from scratch).

The build does not notice a system update that replaces CUDA, cuDNN or FFmpeg. If `ocrs` stops starting after
one (`error while loading shared libraries: libavformat.so...`), delete those two directories and build again.

### Without CUDA

`ocrs-show` has no use for a GPU, and `ocrs` does without one when run with `--device cpu`. For a machine
without the CUDA toolkit, such as one that only shows the text, there is a second way to build both:

```sh
git submodule update --init --depth 1 third_party/opencv
cargo build --release --no-default-features
```

This compiles `third_party/opencv` without CUDA, into directories of its own
(`third_party/_build/opencv-cpu` and `_install/opencv-cpu`) that the build with CUDA neither uses nor
disturbs. It takes no CUDA toolkit, no cuDNN and no `third_party/opencv_contrib`; cmake, clang and the FFmpeg
development files are needed as before. In the programs built this way `--device` is `cpu` unless told
otherwise, and `cuda` is turned down. A machine that only shows the text can do without the models and the
Python that fetches them, too.

Either build puts its programs into `target/release`. To keep both on one machine, give one of them a place
of its own, for instance `--target-dir target/cpu`.

### Environment variables

Environment variables read by the OpenCV build:

| Variable | Meaning |
| --- | --- |
| `CMAKE` | cmake executable, default `cmake` |
| `OCRS_CUDA_ARCH` | with CUDA: compute capabilities to generate code for, e.g. `8.9`; default: what `nvidia-smi` reports |
| `NVCC_CCBIN` | with CUDA: host C++ compiler for nvcc, when CUDA doesn't support the default one yet |
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
| `--device` | `cuda` | `cpu` runs everything without a GPU (slow, meant for testing); the default in a build without CUDA |
| `--camera-size` | the camera's own | frame size to ask a camera for, `WIDTHxHEIGHT`; see below |
| `--preview` | off | show what the camera sees on the terminal, with the markers that are found; see below |
| `--page-size` | from the first frame | size in pixels the text area is rectified to, `WIDTHxHEIGHT` |
| `--stable` | `3`, pictures `1` | how many frames must read a text before it is recorded, see [Output](#output) |
| `--similarity` | `0.9` | how alike two texts must be to count as the same content |
| `--separator` | none | a line to write between the texts of two frames |
| `--dump-pages` | off | directory to save the rectified text area of every recorded frame to |
| `--det-model`, `--rec-model`, `--rec-dict` | built in | other PP-OCR models, see [model/README.md](model/README.md) |

### Cameras

A camera delivers the size it starts up with, which is 640x480 for nearly all of them and too little for a
page of text. `--camera-size 1920x1080` asks for more. `ocrs` says which size the frames have when it starts;
if the camera does not have the size asked for, that is the nearest one it has, and
`v4l2-ctl --list-formats-ext` lists them all. USB cameras tend to have their large sizes as Motion JPEG
only, so `ocrs` switches to that where the size asked for is not to be had uncompressed.

To point the camera, `--preview` draws what it sees on the terminal that `ocrs` runs in, a frame at a time,
as coarse as the terminal has character cells (a smaller font gives a finer picture). In the picture the
text area and its four markers are outlined in green as far as they are found, the signal marker of
`ocrs-show` in cyan, markers with other ids in magenta, and squares that look like a marker but can't be
read (too small, blurred or outshone) in red. The lines below it name the markers that are found and those
that are missing, and say what is in view. Reading goes on as without the preview; what `ocrs` would
otherwise write to the terminal while it runs is printed when it ends. `q` ends it.

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

A recorded text is written as one line per text line, top to bottom; pieces of text on the same row are
joined by a space. The file is flushed after every text.

A single frame of a camera is not to be trusted: one taken while the camera moves is blurred, and one taken
while the page changes shows two pages at once. So a text is only recorded once `--stable 3` of the last 5
frames agree on it (N of the last 2N-1), and of the readings that agree, the one that is most like the
others is written, which tends to be the one with the fewest misread characters. Pictures are views of their
own with nothing to agree on, so for a picture or a sequence of pictures the default is `--stable 1`, which
records what every frame reads.

`--similarity 0.9` lets two texts count as the same content when at most a tenth of their characters differ
(edit distance relative to the longer text). That decides whether frames agree, and whether a text repeats
the one recorded before it and is skipped. With a camera, the reading of an unchanged page flickers by a
character now and then, so asking for equal texts with `--similarity 1` would record little or far too much.

## Test target: `ocrs-show`

`ocrs-show` puts a text file between four markers, on the terminal it runs in or on a
[web page](#as-a-web-page). With a camera pointed at that the whole chain can be tried out: what `ocrs` writes
should be the text that is on show.

```sh
ocrs-show tests/sample.txt --interval 2                 # in a terminal the camera sees
ocrs 0 -o read.txt --camera-size 1920x1080 --preview    # in another one
```

`ocrs-show` begins on a start screen and waits there. `ocrs` says when it has that screen in view, which is
the moment to press `enter`: the pages follow, by key or every `--interval` seconds, and after the last one
an end screen, at which `ocrs` stops by itself.

Between the top two markers there is a fifth one, which tells `ocrs` what is on show: the start screen, the
end screen, or a page and its number (counted up to 16 and then from 0 again). With that `ocrs` need not
guess from the text where one page ends and the next begins. It records each page once, as soon as enough
of its frames agree, and does not read the rest of them; a page that goes away before that is recorded with
its likeliest reading and a remark. It follows that `ocrs` does not need to be told how long a page stays;
the interval only has to be long enough for `--stable` frames of the camera.

The page numbers also show when the text has holes: `ocrs` says so when a number is left out, which is a
page that no frame showed with all its markers, and when a page went by without a frame that could be
read, and it ends with the count of both (`NOT RECORDED: 2 of the pages ...`). A camera that is held by hand
loses pages that way; one that stands still does not.

`tests/sample.txt` is a text for this: some 1,900 words of prose with tables, figures, every printable ASCII
character and the characters that are easily taken for one another.
`tests/code.txt` is one of code: some 260 lines of C++ that compile (`g++ -std=c++20 -x c++ tests/code.txt`),
none wider than 62 columns, with the operators and brackets of the language, every printable ASCII character
and names that differ by characters such as `l` and `1`.

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
| `--no-notes` | `h` | shown | the keys and the measurements below the text area, which `ocrs` does not read |
| `--ink`, `--paper` | | `000000`, `ffffff` | the dark and the light colour as `RRGGBB`; a grey paper dims a screen that glares |
| `--interval` | | off | seconds after which the next page comes up by itself, up to the end screen |
| `--dict`, `--ids` | | `4x4_50`, `0,1,2,3` | the markers, to be given to `ocrs` the same way |

`space` and `p` turn the pages, `enter` does what `space` does; a form feed character in the file starts a
new page. `q` quits and prints the
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
`ocrs` measures it on every page it reads and says so with the text it records (`frame 5: 19 lines, 12.0 px
a column`) and in the line of figures below the `--preview`; `--dump-pages` shows what it gets to read.

The markers need blank space around them to be found. The margin is that space towards a dark window frame
or screen bezel, the padding towards the text. Without padding the text touches the corners of the markers
and the text area is located a pixel or two off.

### As a web page

On a terminal the markers are made of character cells. They cannot be smaller than 6 columns by 3 rows, and
they grow with the text: the larger the font, the more of the screen goes to the markers. `--html` writes the
chart as a web page instead, where the markers have a size of their own and the text has the size that gets
`--columns` characters into a line:

```sh
ocrs-show tests/sample.txt --html show.html --interval 2 --columns 80
```

```
wrote show.html: 8 pages of 80x28 characters, a new one every 2 s
chart 4:3, marker 5% of its width, a character 1/95: 10.5 px a column where the chart is 1000 px wide in a frame of the camera
```

The page is one file that loads nothing else, so it can be copied to whatever has the screen, or fetched
from there (`python -m http.server` in its directory). A browser shows the chart as large as its window
allows, and `f` gives it the whole screen. It has the start screen, the end screen and the fifth marker of
the chart on the terminal, so `ocrs` is run the same way, and the keys that turn the pages: `enter` or
`space` forward, `p` back, `g` and `G` to the first and the last page, `h` for the notes.

| Option | Default | |
| --- | --- | --- |
| `--html` | | file to write the page to |
| `--aspect` | `4:3` | shape of the chart as `W:H` or `WxH`, best that of the camera's frames: `16:9` for 1920x1080 |
| `--columns` | `80` | characters in a line of text; fewer make the text larger and the pages more |
| `--line-height` | `2` | height of a line of text in widths of a character, which is about what a terminal has |
| `--marker` | `5` | side of a marker in percent of the chart's width |
| `--font` | the system's | font of the text as CSS names it, one with characters of equal width |

`--margin`, `--pad`, `--line-spacing`, `--wrap`, `--invert`, `--bold`, `--ink`, `--paper`, `--interval`,
`--no-notes`, `--dict` and `--ids` do what they do on the terminal; `--module` is for the terminal alone.
The layout is settled when the page is written. For another one, write the page again and load it again in
the browser, which comes back to the page it was on.

A chart in the shape of the camera's frames can fill them, and what `ocrs-show` prints says what that comes
to: the 80 characters above get 10.5 pixels each where the chart is 1000 pixels wide in a frame, and 12.6
where it is 1200. `--pad 0` gives the text the space that the padding takes, about a tenth more.

The markers are best left at 5 %. In frames blurred the way a webcam blurs them, a 4x4 marker is found
reliably from some 45 pixels on, which is 5 % of a chart that fills 900 pixels of a frame; markers of 4 %
get lost unless the chart fills nearly all of a frame of 1280, and markers of 3 % are not found at all.
Dictionaries with more modules to a marker need larger markers.

## Tests

`cargo test` runs on the CPU and does not touch the GPU: unit tests for the pure logic, `tests/pipeline.rs`
which reads synthetic frames end to end, and the tests of `ocrs-show`, which have its screens and the layout
of its web page found and read as images; what a browser makes of that page is not among them. The one test
that sends frames through the CUDA code paths is ignored by default:

```sh
cargo test -- --ignored
```
