//! Nothing to call here: this crate exists for its build script, which compiles the OpenCV checkout in
//! `third_party/opencv` (with the feature `cuda`: plus the CUDA modules from `third_party/opencv_contrib`)
//! and exports the install location to dependent build scripts as
//! `DEP_OPENCV_VENDORED_{ROOT,INCLUDE,LIB,LIBS}`.
