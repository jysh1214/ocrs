# model

The networks that ocrs reads with. They are not in the repository; `tools/download_models.py` fetches them
into this directory, which git ignores but for this file and the licence:

```sh
python tools/download_models.py
```

The build compiles the three files of the small tier into `ocrs` if they are here (`build.rs`). Without them
it warns and builds an `ocrs` that has to be told its models when it is run; `ocrs-show` needs none.

They are PaddleOCR's PP-OCRv6 models of the small tier, licensed under the Apache License 2.0 (see
`LICENSE`), in the ONNX conversions published by [RapidOCR](https://github.com/RapidAI/RapidOCR) (v3.10.0).
The recognition model is one model for several scripts: of its 18,708 characters most are Chinese and
Japanese, next to the Latin and Greek letters, digits and signs.

| File | Source |
| --- | --- |
| `PP-OCRv6_det_small.onnx` | `onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx`, unchanged |
| `PP-OCRv6_rec_small.onnx` | `onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx`, modified, see below |
| `ppocrv6_dict.txt` | `paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt`, unchanged |

Source paths are relative to `https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/`. The script
has the SHA-256 of every source file and refuses a download that differs.

## The recognition model is modified

OpenCV's ONNX importer cannot load the recognition model as published, because the model computes shapes at
run time. `tools/freeze_onnx_shapes.py` rewrites those computations so that the model loads and still accepts
lines of any width. It also writes the model's five means over a whole image as the pooling layers they are,
which OpenCV runs on the GPU; as published, each of them would be computed on the CPU. It checks the result
against the original before writing it. The download script runs it, which is why it needs Python with `onnx`
and `numpy`; by hand it is

```sh
python tools/freeze_onnx_shapes.py published.onnx converted.onnx
```

The detection model has five such means too. It is kept as published all the same: in a measurement they
cost about 2 ms of the 72 ms that a frame takes on the GPU.

## Why these models

Measured on screenshots of `ocrs-show --html` in frames of 1280x960, as the share of the text that is read
right, against the PP-OCRv5 mobile models (English recognition) that ocrs used before. "Blurred" is the
chart 1150 pixels wide in the frame under a Gaussian blur of 2.5 pixels, which stands in for a camera; no
camera took part.

| | PP-OCRv5 mobile | PP-OCRv6 small |
| --- | --- | --- |
| `tests/code.txt`, 66 columns, sharp | 99.32 % | 99.61 % |
| `tests/code.txt`, 66 columns, blurred | 96.55 % | 99.32 % |
| `tests/sample.txt`, 80 columns, sharp | 99.64 % | 99.72 % |
| `tests/sample.txt`, 80 columns, blurred | 99.46 % | 99.67 % |
| a frame with some 20 lines of code, RTX 4090 | 54 ms | 72 ms |
| the same frame on the CPU, 32 threads | 590 ms | 490 ms |
| size of the three files | 12.7 MB | 31.2 MB |

The medium tier of PP-OCRv6 (139 MB) reads the sharp screens a little better again (99.76 % and 99.82 %)
and the blurred code worse (98.62 %), as it gets the spaces wrong there. A frame takes it 89 ms on the GPU
and 1.4 s on the CPU.

## Using other models

The script fetches the medium tier as well, and prints the options that make ocrs use it:

```sh
python tools/download_models.py medium
ocrs ... --det-model model/PP-OCRv6_det_medium.onnx --rec-model model/PP-OCRv6_rec_medium.onnx \
    --rec-dict model/ppocrv6_dict.txt
```

Apart from that, `--det-model` takes a PP-OCR detection model (DB) in ONNX format, as it is if OpenCV can
run it: the mobile model of PP-OCRv5 and the small and medium models of PP-OCRv6 can, the server model of
PP-OCRv5 (`ch_PP-OCRv5_det_server.onnx`) as published fails in one of its reshape layers and would need a
conversion of its own first. `--rec-model` takes a PP-OCR recognition model (CTC, input height 48) after the
conversion above, together with its character list in `--rec-dict`.

The models that ocrs used before are `onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx` and
`onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx` with
`paddle/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile/ppocrv5_en_dict.txt`. That recognition model knows 436
characters, none of them Chinese or Japanese. In the four measurements above, which are of ASCII text, it
wrote one character that is not ASCII and the small model of PP-OCRv6 four, a full-width semicolon and a
Greek letter among them.
