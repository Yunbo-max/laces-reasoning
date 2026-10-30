#!/usr/bin/env python3
"""Calibrated footprint guard for the read-only measurement trees.

WHY THIS FILE EXISTS
--------------------
The footprint guard used in rounds 1-14 was an inline shell one-liner, and in round 15 it was
found to be broken on TWO independent axes at the same time:

  * wrong ROOT  -- it globbed `scripts/eval/latentuse/latentuse_g*`, but the trees live at the
                   repo root `laces-reasoning/latentuse_g*`.  A wrong root matches nothing, and
                   "0 files modified" out of a 0-file candidate set reads exactly like a clean
                   footprint.
  * wrong DAY   -- it used a cutoff of 2026-09-22 22:58 while every mtime is 2026-09-23, so the
                   honest answer was 110/110 ("everything changed").

The two errors push in opposite directions, so each one alone would have produced a plausible
but wrong verdict.  A guard that cannot fail is not a guard, so this one ships with controls:

  POSITIVE CONTROLS (the check must FIRE):
    P1  wrong root            -> candidate set empty            -> VACUOUS, must not report PASS
    P2  impossible-future cutoff used as the real cutoff, on a tree we then declare dirty
    P3  a synthetic dirty file inside a temp tree               -> must be reported as modified
  NEGATIVE CONTROL (the check must PASS):
    N1  a temp tree whose files are all older than the cutoff   -> 0 modified

Run:  python footprint_guard.py            # audit the real trees
      python footprint_guard.py --calibrate # run controls, then audit
Exit: 0 audit clean, 1 files modified, 3 instrument broken (vacuous / control failed).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# The trees live at the REPO ROOT, not under scripts/eval/latentuse.  This constant is the exact
# thing that was wrong before, so it is asserted below rather than assumed.
TREES = [REPO / "latentuse_g2_out", REPO / "latentuse_g3"]
MEASUREMENT_SUFFIXES = (".json", ".jsonl")
# Round 15b: the suffix filter was itself a blind spot. G2 flagged that my report carried two
# denominators, 110 and 124; it guessed "stale". It is not stale -- it is a PREDICATE difference:
# 110 = .json/.jsonl only, 124 = every file, and the 14-file delta is 6 .pt latent caches, 2 .txt,
# 2 .py, 2 .pyc, 1 .sh, 1 .log. Two of those 14 changed after the cutoff (z_cache_full_seed44.pt at
# 23:08 and g3_job_id.txt at 23:14) and my guard could not see either, because a `.pt` latent cache
# is an evidence file in every sense that matters even though it is not named `*.json`.
# So the audit now runs at BOTH scopes and prints both, and a scope's verdict is never reported
# without its own denominator. Quoting "4/110" as if it covered the tree was the error.
ALL_FILES_SCOPE = None  # sentinel: no suffix filter


# Session start: nothing under TREES should be written by THIS agent after this instant.  Note
# well: this guard proves "no file changed", which is NOT the same as "I did not change it" --
# mtime and uid cannot attribute authorship (all agents here share one uid).
DEFAULT_CUTOFF = "2026-09-23 22:58"


def enumerate_measurements(roots: list[Path], suffixes=MEASUREMENT_SUFFIXES) -> list[Path]:
    """suffixes=None means EVERY file (the wider scope). A suffix filter is a scope choice, and a
    verdict at one scope says nothing about the other -- see the note on MEASUREMENT_SUFFIXES."""
    out: list[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for p in r.rglob("*"):
            if p.is_file() and (suffixes is None or p.suffix in suffixes):
                out.append(p)
    return sorted(out)


def parse_cutoff(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


def audit(roots: list[Path], cutoff_str: str, label: str,
          suffixes=MEASUREMENT_SUFFIXES) -> tuple[bool, list[Path]]:
    """Returns (ok, modified).  Raises RuntimeError if the instrument is not trustworthy."""
    cutoff = parse_cutoff(cutoff_str + (":00" if cutoff_str.count(":") == 1 else ""))
    files = enumerate_measurements(roots, suffixes)
    scope = "/".join(suffixes) if suffixes else "ALL FILES (no suffix filter)"

    # --- DENOMINATOR FIRST.  A zero candidate set cannot fail, so it is an error, not a pass. ---
    print(f"[{label}] roots        : {[str(r) for r in roots]}")
    print(f"[{label}] scope        : {scope}")
    print(f"[{label}] denominator  : {len(files)} files")
    if not files:
        raise RuntimeError(
            f"[{label}] VACUOUS: 0 candidate files. A wrong root looks identical to a clean "
            f"footprint. Refusing to report PASS."
        )

    mtimes = [p.stat().st_mtime for p in files]

    # --- must-match control: a permissive cutoff has to select EVERY file. ---
    permissive = parse_cutoff("2000-01-01 00:00:00")
    n_perm = sum(1 for m in mtimes if m > permissive)
    print(f"[{label}] control MUST-MATCH  (> 2000-01-01): {n_perm}/{len(files)}  (expect {len(files)})")
    if n_perm != len(files):
        raise RuntimeError(f"[{label}] mtime comparison is broken: permissive cutoff selected {n_perm}")

    # --- must-reject control: an impossible cutoff has to select NONE. ---
    impossible = parse_cutoff("2099-01-01 00:00:00")
    n_imp = sum(1 for m in mtimes if m > impossible)
    print(f"[{label}] control MUST-REJECT (> 2099-01-01): {n_imp}/{len(files)}  (expect 0)")
    if n_imp != 0:
        raise RuntimeError(f"[{label}] mtime comparison is inert: impossible cutoff selected {n_imp}")

    # --- discrimination: the cutoff must sit inside the observed mtime range, otherwise the
    #     verdict is determined by the cutoff rather than by the data. ---
    lo, hi = min(mtimes), max(mtimes)
    print(f"[{label}] observed mtime range: {datetime.fromtimestamp(lo)} .. {datetime.fromtimestamp(hi)}")
    print(f"[{label}] cutoff             : {datetime.fromtimestamp(cutoff)}")
    if cutoff < lo:
        print(f"[{label}] NOTE: cutoff is BEFORE every file -> a 'dirty' verdict here is forced, "
              f"not evidence. This is failure mode 'wrong DAY' from round 15.")
    if cutoff > hi:
        print(f"[{label}] NOTE: cutoff is AFTER every file -> a 'clean' verdict here is forced, "
              f"not evidence.")

    modified = [p for p, m in zip(files, mtimes) if m > cutoff]
    print(f"[{label}] modified since cutoff: {len(modified)}/{len(files)}")
    for p in modified:
        st = p.stat()
        try:
            shown = p.relative_to(REPO)
        except ValueError:
            # Calibration trees live in /tmp, outside REPO. Print the absolute path rather than
            # crashing -- and never silently drop the row, because a guard that skips the files it
            # cannot pretty-print would under-report exactly the thing it exists to find.
            shown = p
        print(f"[{label}]    {datetime.fromtimestamp(st.st_mtime)}  {st.st_size:>8}B  {shown}")
    return (len(modified) == 0), modified


def calibrate() -> bool:
    ok = True
    print("=" * 78)
    print("CALIBRATION")
    print("=" * 78)

    # P1: wrong root -> must raise VACUOUS.
    bad_root = [Path(__file__).parent / "latentuse_g2_out", Path(__file__).parent / "latentuse_g3"]
    try:
        audit(bad_root, DEFAULT_CUTOFF, "P1-wrong-root")
        print("P1 FAIL: wrong root did NOT raise -- THE INSTRUMENT IS BROKEN, a wrong root passes.")
        ok = False
    except RuntimeError as e:
        print(f"P1 PASS: fired as required -> {str(e).splitlines()[0]}")

    tmp = Path(tempfile.mkdtemp(prefix="footprint_cal_"))
    try:
        # N1: all files older than cutoff -> clean.
        clean = tmp / "clean"
        clean.mkdir()
        for i in range(3):
            f = clean / f"rows_{i}.jsonl"
            f.write_text('{"a":1}\n')
            old = time.time() - 86400 * 3
            os.utime(f, (old, old))
        cut = datetime.fromtimestamp(time.time() - 3600).strftime("%Y-%m-%d %H:%M:%S")
        good, mod = audit([clean], cut, "N1-known-good")
        if good and not mod:
            print("N1 PASS: known-good tree reported clean (0 modified out of 3).")
        else:
            print(f"N1 FAIL: known-good tree reported {len(mod)} modified -- false positive.")
            ok = False

        # P3: one fresh file among old ones -> must be flagged, and named.
        dirty = tmp / "dirty"
        dirty.mkdir()
        for i in range(3):
            f = dirty / f"rows_{i}.jsonl"
            f.write_text('{"a":1}\n')
            old = time.time() - 86400 * 3
            os.utime(f, (old, old))
        fresh = dirty / "tampered.json"
        fresh.write_text("{}\n")  # mtime = now, i.e. after the cutoff
        good, mod = audit([dirty], cut, "P3-injected-dirty")
        if (not good) and [p.name for p in mod] == ["tampered.json"]:
            print("P3 PASS: injected dirty file detected, and it is the only one named.")
        else:
            print(f"P3 FAIL: expected exactly ['tampered.json'], got {[p.name for p in mod]}")
            ok = False

        # P2: impossible cutoff on a tree that IS dirty -> forced-clean must be announced.
        good, mod = audit([dirty], "2099-06-01 00:00:00", "P2-forced-clean")
        if good:
            print("P2 PASS: a future cutoff yields a forced 'clean'; the NOTE above marks it as "
                  "not-evidence. (This is why the audit prints its cutoff vs the mtime range.)")
        else:
            print("P2 FAIL: future cutoff still found modifications -- comparison is wrong.")
            ok = False

        # P4: THE SCOPE BLIND SPOT (round 15b). Inject a fresh file whose suffix is OUTSIDE the
        # measurement filter -- a .pt latent cache, exactly the real case. The narrow scope MUST
        # miss it and the wide scope MUST catch it. If both catch it, or both miss it, the
        # two-scope design is pointless and this control says so.
        sneaky = dirty / "z_cache_tampered.pt"
        sneaky.write_text("x")  # mtime = now, after the cutoff
        narrow_ok, narrow_mod = audit([dirty], cut, "P4-narrow-scope")
        wide_ok, wide_mod = audit([dirty], cut, "P4-wide-scope", suffixes=None)
        narrow_names = sorted(p.name for p in narrow_mod)
        wide_names = sorted(p.name for p in wide_mod)
        if "z_cache_tampered.pt" not in narrow_names and "z_cache_tampered.pt" in wide_names:
            print(f"P4 PASS: the .pt cache is INVISIBLE to the narrow scope {narrow_names} and "
                  f"VISIBLE to the wide scope {wide_names}. This is the round-15b blind spot, "
                  f"reproduced on demand -- which is why both scopes must be reported.")
        else:
            print(f"P4 FAIL: narrow={narrow_names} wide={wide_names}. Expected the .pt file to be "
                  f"absent from narrow and present in wide; the two-scope design is not doing "
                  f"what it claims.")
            ok = False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("-" * 78)
    print(f"CALIBRATION: {'PASSED (5/5 controls behaved)' if ok else 'FAILED -- DO NOT TRUST THE AUDIT'}")
    print("-" * 78)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()

    if a.calibrate and not calibrate():
        return 3

    print("=" * 78)
    print("AUDIT OF REAL TREES (read-only evidence) -- BOTH SCOPES")
    print("=" * 78)
    try:
        narrow_clean, narrow_mod = audit(TREES, a.cutoff, "narrow")
        print()
        wide_clean, wide_mod = audit(TREES, a.cutoff, "wide", suffixes=None)
    except RuntimeError as e:
        print(str(e))
        return 3

    print("-" * 78)
    n_narrow = len(enumerate_measurements(TREES))
    n_wide = len(enumerate_measurements(TREES, None))
    print(f"SCOPE RECONCILIATION: narrow (.json/.jsonl) = {n_narrow} files, "
          f"wide (all files) = {n_wide} files, delta = {n_wide - n_narrow}")
    print(f"  changed after cutoff: narrow {len(narrow_mod)}/{n_narrow}, wide {len(wide_mod)}/{n_wide}")
    hidden = [p for p in wide_mod if p not in narrow_mod]
    if hidden:
        print(f"  *** {len(hidden)} changed file(s) are INVISIBLE to the narrow scope: ***")
        for p in hidden:
            print(f"      {p.name}  ({p.suffix or 'no suffix'})")
        print("  A verdict quoted at the narrow scope would have called these clean.")
    if n_narrow == n_wide:
        print("  NOTE: the two scopes coincide here, so this reconciliation has no resolving power.")

    print("-" * 78)
    if wide_clean:
        print("FOOTPRINT: CLEAN at BOTH scopes -- no file changed after the cutoff.")
        return 0
    print(f"FOOTPRINT: {len(wide_mod)} file(s) changed after the cutoff (wide scope).")
    print("This guard does NOT say who changed them. mtime and uid cannot attribute authorship;")
    print("every agent in this session shares one uid. Re-derive any number you quoted from the")
    print("NEW bytes -- a pin tells you that something changed, not whether your claim survived.")
    print("And quote a denominator WITH its scope: '4/110' and '6/124' are the same tree.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
