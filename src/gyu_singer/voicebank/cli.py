from __future__ import annotations

import argparse
import json
from pathlib import Path

from gyu_singer.paths import project_roots

from .factory import FactoryError, VoicebankFactory


def main() -> None:
    parser = argparse.ArgumentParser(prog="gyu-voicebank")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--input", type=Path, required=True)
    init.add_argument("--name", required=True)
    init.add_argument("--languages", required=True)
    init.add_argument("--workspace", type=Path, required=True)
    init.add_argument("--rights-manifest", type=Path, required=True)
    init.add_argument("--dry-run", action="store_true")
    for name in ("inspect", "train", "evaluate", "review-pack", "build", "status"):
        command = sub.add_parser(name)
        command.add_argument("--workspace", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--workspace", type=Path, required=True)
    prepare.add_argument("--review-manifest", type=Path)
    package = sub.add_parser("package")
    package.add_argument("--workspace", type=Path, required=True)
    mode = package.add_mutually_exclusive_group(required=True)
    mode.add_argument("--diagnostic", action="store_true")
    mode.add_argument("--release", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace
    factory = VoicebankFactory(project_roots().project, workspace)
    try:
        if args.command == "init":
            result = factory.init(args.input, args.name, [item.strip() for item in args.languages.split(",") if item.strip()], args.rights_manifest, args.dry_run)
        elif args.command == "prepare":
            result = factory.prepare(args.review_manifest)
        elif args.command == "package":
            result = factory.package(release=args.release, diagnostic=args.diagnostic)
        else:
            result = getattr(factory, args.command.replace("-", "_"))()
    except (FactoryError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
