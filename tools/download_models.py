#!/usr/bin/env python3
"""Download the models that ocrs reads with into `model/`. They are not in the repository.

    download_models.py              the small tier of PP-OCRv6, which the build compiles into ocrs
    download_models.py medium       another tier, for --det-model, --rec-model and --rec-dict

The files come from the ONNX conversions published by RapidOCR. Every one is checked against its SHA-256,
and the recognition model is passed through `freeze_onnx_shapes.py`, without which OpenCV cannot load it.

The download needs Python alone, the conversion Python with `onnx` and `numpy`.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

SOURCE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/"
# path below SOURCE and SHA-256, per tier
DETECTORS = {
    "small": (
        "onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx",
        "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    ),
    "medium": (
        "onnx/PP-OCRv6/det/PP-OCRv6_det_medium.onnx",
        "92078b7355007ccfffcd4c8cd441a3afd4538904d06881b29a155e1e679907c2",
    ),
}
RECOGNISERS = {
    "small": (
        "onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx",
        "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    ),
    "medium": (
        "onnx/PP-OCRv6/rec/PP-OCRv6_rec_medium.onnx",
        "eef444829dbbe18d7fea59a3f6eb75647518d2b3a9568d27c92e42940204894b",
    ),
}
# the tier that build.rs takes from model/
BUILT_IN = "small"
# the recognisers of both tiers have the same characters
DICTIONARY = (
    "paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt",
    "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
)

sys.stdout.reconfigure(line_buffering=True)
REPO = Path(__file__).resolve().parent.parent
CONVERTER = Path(__file__).resolve().with_name("freeze_onnx_shapes.py")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(source, checksum, target):
    """Download `source`, a path below SOURCE, to `target`, unless that file is there already and intact."""
    if target.exists() and sha256(target) == checksum:
        print(f"{target.name}: already here")
        return
    part = target.with_name(target.name + ".part")
    digest, done = hashlib.sha256(), 0
    with urllib.request.urlopen(SOURCE + source, timeout=60) as response, open(part, "wb") as file:
        size = int(response.headers.get("Content-Length") or 0)
        while block := response.read(1 << 20):
            file.write(block)
            digest.update(block)
            done += len(block)
            if sys.stderr.isatty():
                of = f" of {size / 1e6:.1f}" if size else ""
                print(f"\r{target.name}: {done / 1e6:.1f}{of} MB", end="", file=sys.stderr)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    if digest.hexdigest() != checksum:
        part.unlink()
        sys.exit(f"{target.name}: the download has SHA-256 {digest.hexdigest()}, expected {checksum}")
    os.replace(part, target)
    print(f"{target.name}: downloaded, {done / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tier", nargs="*", default=["small"], help="which models to get (default: small)")
    parser.add_argument("--dir", type=Path, default=REPO / "model", help="where to put them (default: %(default)s)")
    parser.add_argument("--force", action="store_true", help="convert a recognition model that is there already again")
    args = parser.parse_args()
    for tier in args.tier:
        if tier not in DETECTORS:
            parser.error(f"there is no tier {tier!r}, only {' and '.join(sorted(DETECTORS))}")
    args.dir.mkdir(parents=True, exist_ok=True)

    dictionary = args.dir / Path(DICTIONARY[0]).name
    fetch(*DICTIONARY, dictionary)
    for tier in dict.fromkeys(args.tier):
        detector = args.dir / Path(DETECTORS[tier][0]).name
        fetch(*DETECTORS[tier], detector)

        # The recognition model is kept as converted. The file as published is of no use to ocrs.
        recogniser = args.dir / Path(RECOGNISERS[tier][0]).name
        if recogniser.exists() and not args.force:
            print(f"{recogniser.name}: already here (--force converts it again)")
        else:
            with tempfile.TemporaryDirectory(dir=args.dir) as scratch:
                published = Path(scratch) / recogniser.name
                fetch(*RECOGNISERS[tier], published)
                converted = Path(scratch) / "converted.onnx"
                done = subprocess.run([sys.executable, str(CONVERTER), str(published), str(converted)])
                if done.returncode != 0:
                    sys.exit(f"{recogniser.name}: the conversion failed; it needs Python with onnx and numpy")
                os.replace(converted, recogniser)

        def shown(path):
            return os.path.relpath(path) if path.is_relative_to(Path.cwd()) else str(path)

        if tier == BUILT_IN and args.dir == REPO / "model":
            print(f"{tier}: the next `cargo build` compiles these into ocrs")
        else:
            print(
                f"{tier}: ocrs ... --det-model {shown(detector)} --rec-model {shown(recogniser)}"
                f" --rec-dict {shown(dictionary)}"
            )


if __name__ == "__main__":
    main()
