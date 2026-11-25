# models

The networks that are compiled into the binary (`src/ocr.rs`). They are PaddleOCR's PP-OCRv6 models of the
small tier, licensed under the Apache License 2.0 (see `LICENSE`), in the ONNX conversions published by
[RapidOCR](https://github.com/RapidAI/RapidOCR) (v3.10.0). The recognition model is one model for several
scripts: of its 18,708 characters most are Chinese and Japanese, next to the Latin and Greek letters, digits
and signs.

| File | Source | SHA-256 of the source file |
| --- | --- | --- |
| `PP-OCRv6_det_small.onnx` | `onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx`, unchanged | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` |
| `PP-OCRv6_rec_small.onnx` | `onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx`, modified, see below | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` |
| `ppocrv6_dict.txt` | `paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt`, unchanged | `b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d` |

Source paths are relative to `https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/`.

## The recognition model is modified

OpenCV's ONNX importer cannot load the recognition model as published, because the model computes shapes at
run time. `tools/freeze_onnx_shapes.py` rewrites those computations so that the model loads and still accepts
lines of any width. It also writes the model's five means over a whole image as the pooling layers they are,
which OpenCV runs on the GPU; as published, each of them would be computed on the CPU. The script checks the
result against the original before writing it:

```sh
python tools/freeze_onnx_shapes.py PP-OCRv6_rec_small.onnx models/PP-OCRv6_rec_small.onnx
```

The script needs Python with `onnx` and `numpy`.

The detection model has five such means too. It is kept as published all the same: in a measurement they
cost about 2 ms of the 72 ms that a frame takes on the GPU.

## Why these models

Measured on screenshots of `ocrs-show --html` in frames of 1280x960, as the share of the text that is read
right, against the PP-OCRv5 mobile models (English recognition) that were built in before. "Blurred" is the
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
and the blurred code worse (98.62 %), as it gets the spaces wrong there.

## Using other models

`--det-model` takes a PP-OCR detection model (DB) in ONNX format, as it is if OpenCV can run it: the mobile
model of PP-OCRv5 and the small and medium models of PP-OCRv6 can, the server model of PP-OCRv5
(`ch_PP-OCRv5_det_server.onnx`) as published fails in one of its reshape layers and would need a conversion
of its own first. `--rec-model` takes a PP-OCR recognition model (CTC, input height 48) after the conversion
above, together with its character list in `--rec-dict`.

The models that were built in before are `onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx` and
`onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx` with
`paddle/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile/ppocrv5_en_dict.txt`. That recognition model knows 436
characters, none of them Chinese or Japanese. In the four measurements above, which are of ASCII text, it
wrote one character that is not ASCII and the built-in model four, a full-width semicolon and a Greek letter
among them.
