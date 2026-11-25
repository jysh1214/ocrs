use std::path::Path;

/// The models that are compiled into ocrs, by the name that `src/ocr.rs` asks for them and their file in
/// `model/`.
const MODELS: [(&str, &str); 3] = [
    ("OCRS_DETECTOR", "PP-OCRv6_det_small.onnx"),
    ("OCRS_RECOGNIZER", "PP-OCRv6_rec_small.onnx"),
    ("OCRS_DICTIONARY", "ppocrv6_dict.txt"),
];

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    built_in_models();
    // The OpenCV libraries are in the vendored install, which the dynamic loader doesn't search. Recording
    // that directory in the binaries lets them start without LD_LIBRARY_PATH.
    let lib_dir =
        std::env::var("DEP_OPENCV_VENDORED_LIB").expect("opencv-vendored exports its library directory");
    println!("cargo::rustc-link-arg=-Wl,-rpath,{lib_dir}");
}

/// The models are not in the repository: `tools/download_models.py` fetches them into `model/`. They are
/// compiled in if they are there. A build without them is not an error, as `ocrs-show` has no use for them
/// and `ocrs` can be told its models when it is run.
fn built_in_models() {
    println!("cargo::rustc-check-cfg=cfg(built_in_models)");
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("model");
    let mut missing = Vec::new();
    for (variable, name) in MODELS {
        let file = directory.join(name);
        println!("cargo::rerun-if-changed={}", file.display());
        println!("cargo::rustc-env={variable}={}", file.display());
        if !file.is_file() {
            missing.push(name);
        }
    }
    if missing.is_empty() {
        println!("cargo::rustc-cfg=built_in_models");
    } else {
        println!(
            "cargo::warning=ocrs is built without models, as model/ has no {}: tools/download_models.py \
             fetches them",
            missing.join(", ")
        );
    }
}
