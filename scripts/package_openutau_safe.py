#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gyu_singer.openutau_packager import PackageError, build_openutau_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-package", action="store_true")
    args = parser.parse_args()
    try:
        result = build_openutau_package(Path(__file__).resolve().parents[1], args.output, args.backend, args.diagnostic_package)
    except PackageError as error:
        parser.error(str(error))
    print(f"{result.status}: {result.path}")


if __name__ == "__main__":
    main()
