//! Builds `third_party/opencv` (+ `third_party/opencv_contrib`) with CUDA and installs it under
//! `third_party/_install/opencv`.
//!
//! The build tree lives in `third_party/_build/opencv` rather than in `OUT_DIR` on purpose: `OUT_DIR` is
//! per profile and changes with every toolchain update, and recompiling OpenCV's CUDA code takes a long
//! time. CMake keeps the tree incremental, and a stamp file lets an up-to-date tree skip CMake entirely.
//!
//! Environment variables (all optional):
//! * `CMAKE`: cmake executable, default `cmake` from `PATH`
//! * `OCRS_CUDA_ARCH`: CUDA compute capabilities to generate code for, e.g. `8.9` or `8.6 8.9`.
//!   Default: what `nvidia-smi` reports for the installed GPUs, or every architecture nvcc knows.
//! * `NVCC_CCBIN`: host C++ compiler for nvcc (CUDA usually lags behind the newest GCC)
//! * `CMAKE_PREFIX_PATH`: extra prefixes to search, e.g. for a cuDNN that is not installed system-wide
//! * `OCRS_OPENCV_CMAKE_ARGS`: extra arguments appended to the CMake command line (whitespace separated)

use std::env;
use std::fs::{self, File};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// OpenCV modules the application uses. `BUILD_LIST` adds whatever these depend on.
const MODULES: &[&str] = &[
    "core",
    "imgproc",
    "imgcodecs",
    "videoio",
    "objdetect",
    "dnn",
    "cudev",
    "cudawarping",
];

/// Variables that change the result of the build; cargo re-runs this script when one of them changes.
const ENV_VARS: &[&str] = &[
    "CMAKE",
    "OCRS_CUDA_ARCH",
    "OCRS_OPENCV_CMAKE_ARGS",
    "NVCC_CCBIN",
    "CUDA_PATH",
    "CMAKE_PREFIX_PATH",
    "CC",
    "CXX",
];

const STAMP_NAME: &str = "ocrs-build-stamp";

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let third_party = manifest_dir
        .join("../../third_party")
        .canonicalize()
        .expect("third_party/ directory next to crates/");
    let opencv_src = third_party.join("opencv");
    let contrib_src = third_party.join("opencv_contrib");
    let build_dir = third_party.join("_build/opencv");
    let prefix = third_party.join("_install/opencv");
    let stamp = prefix.join(STAMP_NAME);

    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed={}", stamp.display());
    for var in ENV_VARS {
        println!("cargo::rerun-if-env-changed={var}");
    }
    let revisions = [&opencv_src, &contrib_src].map(|dir| source_revision(dir));

    let args = cmake_args(&opencv_src, &contrib_src, &prefix);
    let wanted_stamp = format!("{}\n{}\n", revisions.join("\n"), args.join("\n"));
    let up_to_date = fs::read_to_string(&stamp).is_ok_and(|have| have == wanted_stamp);
    if !up_to_date {
        // a half-finished build must not look current to the next run
        let _ = fs::remove_file(&stamp);
        build(&build_dir, &args);
        fs::write(&stamp, wanted_stamp).expect("can't write the build stamp");
    }

    let lib_dir = prefix.join("lib");
    let libs = installed_libs(&lib_dir);
    assert!(!libs.is_empty(), "no libopencv_*.so in {}", lib_dir.display());
    println!("cargo::metadata=root={}", prefix.display());
    println!(
        "cargo::metadata=include={}",
        prefix.join("include/opencv4").display()
    );
    println!("cargo::metadata=lib={}", lib_dir.display());
    println!("cargo::metadata=libs={}", libs.join(","));
}

/// Commit of a source checkout, and a re-run trigger for when a different commit gets checked out.
fn source_revision(dir: &Path) -> String {
    if !dir.join("CMakeLists.txt").exists() && !dir.join("modules").exists() {
        panic!(
            "{} is empty, fetch the submodules first: git submodule update --init --depth 1",
            dir.display()
        );
    }
    let git = |args: &[&str]| {
        let out = Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        println!("cargo::rerun-if-changed={git_dir}/HEAD");
    }
    git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown revision".to_string())
}

fn cmake_args(opencv_src: &Path, contrib_src: &Path, prefix: &Path) -> Vec<String> {
    let mut args = vec![
        "-S".to_string(),
        opencv_src.display().to_string(),
        "-DCMAKE_BUILD_TYPE=Release".to_string(),
        format!("-DCMAKE_INSTALL_PREFIX={}", prefix.display()),
        "-DCMAKE_INSTALL_LIBDIR=lib".to_string(),
        // the installed libraries find each other, and keep finding CUDA / cuDNN in non-default locations
        "-DCMAKE_INSTALL_RPATH=$ORIGIN".to_string(),
        "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON".to_string(),
        format!(
            "-DOPENCV_EXTRA_MODULES_PATH={}",
            contrib_src.join("modules").display()
        ),
        format!("-DBUILD_LIST={}", MODULES.join(",")),
    ];
    if which("ninja") {
        args.push("-GNinja".to_string());
    }
    let fixed = [
        "-DBUILD_SHARED_LIBS=ON",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_DOCS=OFF",
        "-DBUILD_opencv_apps=OFF",
        "-DBUILD_JAVA=OFF",
        "-DBUILD_opencv_python2=OFF",
        "-DBUILD_opencv_python3=OFF",
        "-DOPENCV_GENERATE_PKGCONFIG=OFF",
        // CUDA: device code through CMake's own CUDA language support (FindCUDA is gone from current CMake),
        // DNN inference through cuDNN. Asking for OPENCV_DNN_CUDA explicitly turns a missing cuDNN into a
        // configure error instead of a silent CPU-only DNN module.
        "-DWITH_CUDA=ON",
        "-DENABLE_CUDA_FIRST_CLASS_LANGUAGE=ON",
        "-DWITH_CUBLAS=ON",
        "-DWITH_CUDNN=ON",
        "-DOPENCV_DNN_CUDA=ON",
        "-DWITH_CUFFT=OFF",
        "-DWITH_NVCUVID=OFF",
        "-DWITH_NVCUVENC=OFF",
        // video files and cameras
        "-DWITH_FFMPEG=ON",
        "-DWITH_V4L=ON",
        "-DWITH_GSTREAMER=OFF",
        "-DWITH_OBSENSOR=OFF",
        // the ONNX importer needs protobuf, the bundled copy avoids chasing the system one's ABI
        "-DWITH_PROTOBUF=ON",
        "-DBUILD_PROTOBUF=ON",
        "-DWITH_FLATBUFFERS=OFF",
        // nothing below is used; IPP and ADE would also download archives at configure time
        "-DWITH_IPP=OFF",
        "-DWITH_ADE=OFF",
        "-DWITH_ITT=OFF",
        "-DWITH_OPENCL=OFF",
        "-DWITH_VA=OFF",
        "-DWITH_EIGEN=OFF",
        "-DWITH_LAPACK=OFF",
        "-DWITH_GTK=OFF",
        "-DWITH_QT=OFF",
        "-DWITH_VTK=OFF",
        "-DWITH_TIFF=OFF",
        "-DWITH_WEBP=OFF",
        "-DWITH_OPENEXR=OFF",
        "-DWITH_JASPER=OFF",
        "-DWITH_OPENJPEG=OFF",
        "-DWITH_AVIF=OFF",
    ];
    args.extend(fixed.iter().map(|s| s.to_string()));

    match cuda_arch() {
        Some(arch) => {
            args.push(format!("-DCUDA_ARCH_BIN={arch}"));
            args.push("-DCUDA_ARCH_PTX=".to_string());
        }
        None => println!(
            "cargo::warning=can't tell the GPU's compute capability, generating CUDA code for every architecture (set OCRS_CUDA_ARCH to narrow it down)"
        ),
    }
    if let Some(ccbin) = env::var_os("NVCC_CCBIN").filter(|v| !v.is_empty()) {
        args.push(format!(
            "-DCMAKE_CUDA_HOST_COMPILER={}",
            Path::new(&ccbin).display()
        ));
    }
    if let Ok(extra) = env::var("OCRS_OPENCV_CMAKE_ARGS") {
        args.extend(extra.split_whitespace().map(str::to_string));
    }
    args
}

/// Compute capabilities to build device code for, in OpenCV's `CUDA_ARCH_BIN` format.
fn cuda_arch() -> Option<String> {
    if let Ok(arch) = env::var("OCRS_CUDA_ARCH") {
        return Some(arch).filter(|a| !a.trim().is_empty());
    }
    // plain driver query, doesn't create a CUDA context
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let mut caps = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && l.chars().all(|c| c.is_ascii_digit() || c == '.'))
        .map(str::to_string)
        .collect::<Vec<_>>();
    caps.sort();
    caps.dedup();
    (!caps.is_empty()).then(|| caps.join(" "))
}

fn build(build_dir: &Path, args: &[String]) {
    fs::create_dir_all(build_dir).expect("can't create the OpenCV build directory");
    let cmake = env::var_os("CMAKE").unwrap_or_else(|| "cmake".into());
    let jobs = env::var("NUM_JOBS").unwrap_or_else(|_| "1".to_string());

    let configure_log = build_dir.join("ocrs-configure.log");
    let mut configure = Command::new(&cmake);
    configure.arg("-B").arg(build_dir).args(args);
    run(configure, &configure_log, "configuring OpenCV");
    let summary = fs::read_to_string(&configure_log).unwrap_or_default();
    // OpenCV only notes a missing FFmpeg in its summary, without it no video file can be opened
    if !summary_says_yes(&summary, "FFMPEG:") {
        panic!(
            "OpenCV was configured without FFmpeg (development files for libavcodec/libavformat/libswscale are needed), see {}",
            configure_log.display()
        );
    }

    let build_log = build_dir.join("ocrs-build.log");
    let mut build = Command::new(&cmake);
    build
        .arg("--build")
        .arg(build_dir)
        .args(["--target", "install", "--parallel", &jobs]);
    run(build, &build_log, "building OpenCV");
}

/// Runs `cmd` with its output going to `log`; a failure panics with the end of that log.
fn run(mut cmd: Command, log: &Path, what: &str) {
    let out = File::create(log).unwrap_or_else(|e| panic!("can't create {}: {e}", log.display()));
    let err = out.try_clone().expect("can't duplicate the log handle");
    let status = cmd
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err)
        .status()
        .unwrap_or_else(|e| match e.kind() {
            ErrorKind::NotFound => panic!(
                "{:?} was not found, put cmake on PATH or point the CMAKE environment variable at it",
                cmd.get_program()
            ),
            _ => panic!("can't run {:?}: {e}", cmd.get_program()),
        });
    if status.success() {
        return;
    }
    let text = fs::read_to_string(log).unwrap_or_default();
    let lines = text.lines().collect::<Vec<_>>();
    let tail = lines[lines.len().saturating_sub(60)..].join("\n");
    let hint = if text.contains("requires cuDNN") {
        "\nhint: cuDNN was not found. Install it (it has to match the CUDA toolkit's major version), or add its prefix to CMAKE_PREFIX_PATH."
    } else {
        ""
    };
    panic!(
        "{what} failed ({status}), full log: {}\n{tail}{hint}",
        log.display()
    );
}

/// Whether OpenCV's configure summary has a `-- <key> YES ...` line.
fn summary_says_yes(summary: &str, key: &str) -> bool {
    summary.lines().any(|line| {
        line.split_once(key)
            .is_some_and(|(before, value)| before.trim() == "--" && value.trim_start().starts_with("YES"))
    })
}

/// Names (without `lib` / `.so`) of the installed OpenCV libraries.
fn installed_libs(lib_dir: &Path) -> Vec<String> {
    let mut libs = fs::read_dir(lib_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            Some(name.strip_prefix("lib")?.strip_suffix(".so")?.to_string())
        })
        .filter(|name| name.starts_with("opencv_"))
        .collect::<Vec<_>>();
    libs.sort();
    libs
}

fn which(program: &str) -> bool {
    env::var_os("PATH").is_some_and(|paths| env::split_paths(&paths).any(|dir| dir.join(program).is_file()))
}
