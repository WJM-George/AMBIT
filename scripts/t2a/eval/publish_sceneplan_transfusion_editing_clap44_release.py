#!/usr/bin/env python3
"""Seal a native Editing delivery only after the complete independent audio test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))

from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_release import publish_release, validate_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--joint-selection",type=Path)
    parser.add_argument("--joint-selection-sha256")
    parser.add_argument("--independent-test",type=Path)
    parser.add_argument("--independent-test-sha256")
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--release-sha256")
    args = parser.parse_args()
    if args.verify_only:
        if not args.release_sha256: parser.error("--verify-only needs --release-sha256")
        value = validate_release(args.output_dir/"RELEASE.json",expected_sha256=args.release_sha256)
        print(json.dumps({"event":"clap44_release_verified","checkpoint":value["checkpoint"],"status":"PASS"}))
        return 0
    if not all((args.joint_selection,args.joint_selection_sha256,args.independent_test,args.independent_test_sha256)):
        parser.error("publication needs the pinned joint selection and independent-test result")
    record = publish_release(joint_selection=args.joint_selection,selection_sha256=args.joint_selection_sha256,
        independent_test=args.independent_test,test_sha256=args.independent_test_sha256,output_dir=args.output_dir)
    print(json.dumps({"event":"clap44_release_published","release":record,"status":"PASS"}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
