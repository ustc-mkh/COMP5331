"""Fetch/verify the pinned BM3 source; never downloads training data or runs code."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "ef82a333823ae01ac5964039720f697532b40ea3"
REPOSITORY = "https://github.com/enoche/BM3"
FILES = [
    "LICENSE", "README.md", "src/models/bm3.py",
    "src/common/abstract_recommender.py", "src/common/loss.py",
    "src/common/trainer.py", "src/common/init.py",
    "src/utils/dataset.py", "src/utils/dataloader.py", "src/utils/metrics.py",
    "src/utils/topk_evaluator.py", "src/utils/utils.py", "src/utils/quick_start.py",
    "src/utils/configurator.py", "src/utils/logger.py", "src/utils/misc.py",
    "src/configs/overall.yaml", "src/configs/model/BM3.yaml",
    "src/configs/dataset/baby.yaml", "src/main.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="offline integrity check")
    args = parser.parse_args()
    destination = ROOT / "upstream" / "BM3"
    manifest_path = ROOT / "upstream" / "manifest.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["commit"] != COMMIT or set(manifest["files"]) != set(FILES):
            raise RuntimeError("Unexpected upstream manifest")
        for name, meta in manifest["files"].items():
            digest = hashlib.sha256((destination / name).read_bytes()).hexdigest()
            if digest != meta["sha256"]:
                raise RuntimeError("Upstream file has changed: " + name)
        print("Verified {} unmodified upstream files at {}".format(len(FILES), COMMIT))
        return
    manifest = {"repository": REPOSITORY, "commit": COMMIT, "files": {}}
    for name in FILES:
        url = "https://raw.githubusercontent.com/enoche/BM3/{}/{}".format(COMMIT, name)
        request = urllib.request.Request(url, headers={"User-Agent": "BM3-course-reproduction"})
        with urllib.request.urlopen(request, timeout=45) as response:
            content = response.read()
        target = destination / name
        if target.exists() and target.read_bytes() != content:
            raise RuntimeError("Refusing to overwrite modified source: " + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest["files"][name] = {"sha256": hashlib.sha256(content).hexdigest(),
                                   "bytes": len(content), "url": url}
        print(name, len(content), flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    license_target = ROOT / "LICENSE"
    if not license_target.exists():
        license_target.write_bytes((destination / "LICENSE").read_bytes())


if __name__ == "__main__":
    main()
