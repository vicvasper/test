"""Pre-flight check for a Trace the Ace submission.zip.

Every check here mirrors something the official runtime actually does, so a
clean run means the platform will at least *execute* your code. Run it before
every upload:

    python training/preflight.py --zip submission/submission.zip --data-dir data-demo

Why this exists
---------------
The platform's entrypoint validates the archive with a *substring* grep:

    submission_files=$(zip -sf ./submission/submission.zip)
    grep -q main.py <<<$submission_files

An archive containing ``submission_src/main.py`` passes that check, and then
dies seconds later at ``python main.py`` because the entrypoint has already
``cd``-ed to ``/code_execution`` where no ``main.py`` exists. The local
``just check-submission`` recipe uses ``grep -F -x`` (exact line) and does catch
it -- which is precisely why skipping that step is expensive. This script makes
the exact-match check non-optional and adds the ones the platform cannot do for
you.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

# Anything that reaches the network. The runtime has no internet: these do not
# fail slowly, they fail the whole submission.
NETWORK_CALLS = re.compile(
    r"\b(requests\.(get|post|put)|urlopen|urlretrieve|httpx\.(get|post)|"
    r"wget|curl|pip\s+install|subprocess\.(run|call|Popen).*pip|"
    r"snapshot_download|hf_hub_download|load_dataset\()",
    re.I,
)
# from_pretrained on a bare hub id (no local path, no local_files_only) is a
# silent download attempt.
HUB_ID = re.compile(r"from_pretrained\(\s*[\"']([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)[\"']")

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def record(status: str, check: str, detail: str = "") -> None:
    results.append((status, check, detail))


def check_archive(zip_path: Path) -> list[str]:
    if not zip_path.exists():
        record(FAIL, "archive exists", f"{zip_path} not found")
        return []

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        bad = [n for n in names if zf.getinfo(n).file_size > 0 and n.startswith("/")]
    if bad:
        record(FAIL, "no absolute paths in archive", ", ".join(bad[:3]))

    if "main.py" in names:
        record(PASS, "main.py at archive root")
    else:
        nested = [n for n in names if n.endswith("/main.py") or n.endswith("main.py")]
        if nested:
            record(
                FAIL,
                "main.py at archive root",
                f"found only {nested[0]!r}. The platform entrypoint's substring grep "
                f"ACCEPTS this and then crashes at 'python main.py'. Repack from "
                f"INSIDE the source dir: cd submission_src && rpzip -r ../submission/submission.zip ./*",
            )
        else:
            record(FAIL, "main.py at archive root", "no main.py anywhere in the archive")

    total = sum(zipfile.ZipFile(zip_path).getinfo(n).file_size for n in names)
    record(
        PASS if total < 6 * 2**30 else WARN,
        "archive uncompressed size",
        f"{total / 2**20:.1f} MiB",
    )
    return names


def check_source(zip_path: Path) -> None:
    """Static scan of every .py in the archive for runtime-fatal patterns."""
    with zipfile.ZipFile(zip_path) as zf:
        py_files = [n for n in zf.namelist() if n.endswith(".py")]
        if not py_files:
            record(FAIL, "archive contains python", "no .py files")
            return
        net_hits, hub_hits, syntax_bad = [], [], []
        for name in py_files:
            text = zf.read(name).decode("utf-8", errors="replace")
            try:
                ast.parse(text)
            except SyntaxError as exc:
                syntax_bad.append(f"{name}:{exc.lineno}")
            for line_no, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                if NETWORK_CALLS.search(line):
                    net_hits.append(f"{name}:{line_no}")
                for match in HUB_ID.finditer(line):
                    if "local_files_only" not in line and "/code_execution" not in line:
                        hub_hits.append(f"{name}:{line_no} -> {match.group(1)}")

    record(PASS if not syntax_bad else FAIL, "all python parses", ", ".join(syntax_bad[:3]))
    record(
        PASS if not net_hits else FAIL,
        "no network calls at inference",
        ", ".join(net_hits[:4]) + "  (runtime has no internet)",
    )
    record(
        PASS if not hub_hits else FAIL,
        "no Hugging Face hub downloads",
        ", ".join(hub_hits[:3])
        + "  (load from /code_execution/huggingface_models/<repo_id> with local_files_only=True, "
        "and the model must already be listed in runtime/huggingface_models.txt)",
    )


def run_submission(zip_path: Path, data_dir: Path, timeout_s: int) -> Path | None:
    """Execute exactly as the entrypoint does: unzip, cd, python main.py."""
    workdir = Path(tempfile.mkdtemp(prefix="preflight_"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(workdir)
    shutil.copytree(data_dir, workdir / "data")

    # Inherit the real environment. Deliberately do NOT rewrite PATH or HOME:
    # trimming them drops interpreter search paths (user site-packages, for one)
    # and produces import errors the actual runtime would never raise -- a false
    # alarm here is as costly as a missed one.
    child_env = dict(os.environ)
    child_env["IS_SMOKE_TEST"] = "1"
    # Offline is the runtime default; make an accidental hub call fail loudly here
    # rather than silently on the platform.
    child_env["HF_HUB_OFFLINE"] = "1"
    child_env["TRANSFORMERS_OFFLINE"] = "1"

    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "main.py"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        record(FAIL, "main.py completes", f"exceeded {timeout_s}s")
        return None
    elapsed = time.time() - started

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-600:]
        record(FAIL, "main.py exits 0", f"exit={proc.returncode}\n{tail}")
        return None
    record(PASS, "main.py exits 0", f"{elapsed:.1f}s on this data")

    out = workdir / "submission.csv"
    if not out.exists():
        record(FAIL, "submission.csv produced", "entrypoint would report an error and score nothing")
        return None
    record(PASS, "submission.csv produced")
    return out


def check_output(out_csv: Path, data_dir: Path) -> None:
    import pandas as pd

    sub = pd.read_csv(out_csv)
    fmt = pd.read_csv(data_dir / "submission_format.csv")

    record(
        PASS if list(sub.columns) == list(fmt.columns) else FAIL,
        "columns match submission_format",
        f"{list(sub.columns)} vs {list(fmt.columns)}",
    )
    record(
        PASS if len(sub) == len(fmt) else FAIL,
        "row count matches",
        f"{len(sub)} vs {len(fmt)}",
    )
    same_order = sub["response_id"].tolist() == fmt["response_id"].tolist()
    record(
        PASS if same_order else FAIL,
        "response_id order identical",
        "" if same_order else "rows are misaligned -- this scores like noise while looking healthy",
    )

    p = pd.to_numeric(sub["probability"], errors="coerce")
    record(PASS if p.notna().all() else FAIL, "no NaN / non-numeric probabilities")
    record(
        PASS if ((p > 0) & (p < 1)).all() else FAIL,
        "probabilities strictly inside (0, 1)",
        "log loss is unbounded at exactly 0 or 1 -- clip to [1e-4, 1-1e-4]",
    )
    if p.nunique() <= 1:
        record(
            WARN,
            "predictions vary",
            f"every row is {p.iloc[0]:.4f}. A constant 0.5 scores ln(2)=0.6931 -- this is what "
            "the unmodified example submission produces. Confirm your model actually ran.",
        )
    else:
        record(PASS, "predictions vary", f"{p.nunique()} distinct values")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--skip-run", action="store_true", help="static checks only")
    args = ap.parse_args()

    print(f"Pre-flight: {args.zip}  (data: {args.data_dir})\n")
    names = check_archive(args.zip)
    if names:
        check_source(args.zip)
        if not args.skip_run and not any(s == FAIL for s, _, _ in results):
            out = run_submission(args.zip, args.data_dir, args.timeout)
            if out:
                check_output(out, args.data_dir)

    width = max(len(c) for _, c, _ in results)
    for status, check, detail in results:
        mark = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL"}[status]
        print(f"  [{mark}] {check.ljust(width)}  {detail}".rstrip())

    failed = sum(s == FAIL for s, _, _ in results)
    warned = sum(s == WARN for s, _, _ in results)
    print(f"\n{len(results) - failed - warned} passed, {warned} warnings, {failed} failures")
    if failed:
        print("DO NOT UPLOAD. Fix the failures above first.")
        return 1
    print("Archive is safe to upload. Submit a smoke test first if you have one left.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
