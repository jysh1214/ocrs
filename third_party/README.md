# third_party

| Path | What | Version |
| --- | --- | --- |
| `opencv/` | OpenCV, git submodule | branch `4.x` at `a495c9b` (2026-11-23), reports itself as 4.15.0-dev |
| `opencv_contrib/` | OpenCV's extra modules (the CUDA ones live here), git submodule | branch `4.x` at `2f5266f` (2026-11-05) |
| `opencv-rust/` | the [`opencv`](https://crates.io/crates/opencv) crate (Rust bindings), copied in and patched | 0.102.0 |
| `opencv-rust.patch` | the changes made to `opencv-rust/` | |
| `_build/`, `_install/` | OpenCV build trees and install prefixes, generated, not in git: `opencv` for the build with CUDA, `opencv-cpu` for the one without | |

The submodules are not on a release tag because the latest 4.x release, 4.14.0, does not compile against
FFmpeg 9 (`AVCodec::pix_fmts` and `AVCodec::supported_framerates` are gone); the fix is on the `4.x` branch
only. Move both submodules to the 4.15.0 tag once it exists.

## How the build fits together

`crates/opencv-vendored/build.rs` runs CMake on `opencv/` + `opencv_contrib/` and installs the result into
`_install/opencv`. The modules to build and all CMake options are listed in that file. Without the cargo
feature `cuda` (`cargo build --no-default-features`) it builds `opencv/` alone, without CUDA, and installs
that into `_install/opencv-cpu`; the two builds are independent of each other.

The `opencv` crate generates its bindings from the installed headers, so OpenCV has to be built before that
crate's build script runs. Cargo only orders build scripts along dependency edges, and the published crate has
no dependency that could build OpenCV for it. That is why a copy of the crate lives here with a small patch:

- `Cargo.toml`: a dependency on `opencv-vendored`, which gives cargo the ordering and passes the install
  location on as `DEP_OPENCV_VENDORED_*`; and two lints switched off that fire with few modules enabled;
- `build/library.rs`: use that location, and only that one, instead of probing the system for an OpenCV;
- `rustfmt.toml`: keeps `cargo fmt --all` from reformatting the copy.

The `docs/` directory of the published crate (pre-generated bindings for docs.rs, 16 MB) is left out.

## Updating

OpenCV: check out the new tag in both submodules. The next `cargo build` notices the new revision and rebuilds.

`opencv` crate: replace `opencv-rust/` with the new release from crates.io (without `docs/`) and re-apply the
patch from inside it:

```sh
patch -p1 < ../opencv-rust.patch
```
