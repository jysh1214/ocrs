# models

The networks that are compiled into the binary (`src/ocr.rs`). They are PaddleOCR's PP-OCRv5 models, licensed
under the Apache License 2.0 (see `LICENSE`), in the ONNX conversions published by
[RapidOCR](https://github.com/RapidAI/RapidOCR) (v3.10.0).

| File | Source | SHA-256 of the source file |
| --- | --- | --- |
| `PP-OCRv5_det_mobile.onnx` | `onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx`, unchanged | `4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae` |
| `en_PP-OCRv5_rec_mobile.onnx` | `onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx`, modified, see below | `c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8` |
| `ppocrv5_en_dict.txt` | `paddle/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile/ppocrv5_en_dict.txt`, unchanged | `e025a66d31f327ba0c232e03f407ae8d105e1e709e7ccb3f408aa778c24e70d6` |

Source paths are relative to `https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/`.

## The recognition model is modified

OpenCV's ONNX importer cannot load the recognition model as published, because the model computes shapes at
run time. `tools/freeze_onnx_shapes.py` rewrites those computations so that the model loads and still accepts
lines of any width; it checks the result against the original before writing it:

```sh
python tools/freeze_onnx_shapes.py en_PP-OCRv5_rec_mobile.onnx models/en_PP-OCRv5_rec_mobile.onnx
```

The script needs Python with `onnx` and `numpy`.

## Using other models

`--det-model` takes any PP-OCR detection model (DB) in ONNX format as it is. `--rec-model` takes a PP-OCR
recognition model (CTC, input height 48) after the conversion above, together with its character list in
`--rec-dict`. This is the way to read other scripts, e.g. the Chinese PP-OCRv5 recognition model.
