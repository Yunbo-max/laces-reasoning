#!/usr/bin/env python3
"""Gate a staged loop cell and copy it to canonical. Refuses on any failed check.

Usage: publish_gated_loop_cell.py [--stage DIR] NAME [NAME ...]
`--stage` selects the staging directory (default longctx_decodefix); the EOS-fix rebuild of
2026-09-21 stages into longctx_eosfix, so it needs `--stage results/benchmarks/longctx_eosfix`.
"""
import json, shutil, sys
from pathlib import Path

# `eos_fires`: does this model DEMONSTRABLY stop early once the real eos ids are in the stop set?
# Measured, not assumed, and it is not a property of the suite but of the model+tokenizer:
#   rwkv7-world3 / global-dit-50k -- yes. Both decode with the RWKV-7 tokenizer, whose three eos
#     declarations disagree (tokenizer 65530 / generation_config 0 / config 2); with the union in
#     the stop set, 141/450 and 79/450 rows end below their own dataset budget.
#   gla-1.3b -- NO, and legitimately so. gla declares eos=2 in config.json, generation_config.json
#     AND tokenizer_config.json, all three agreeing, so its pre-resolver runs already had the
#     correct stop id; it simply never emits it (0 of 2950 rows contain id 2 AND 0 of 2950 end
#     below budget -- BOTH halves are needed, because greedy_loop breaks before appending so a
#     stop that fired leaves no trace in the ids). Requiring early stops here would be wrong.
EXPECT = {
    "rwkv7-world3__ruler":     dict(rows=1800, cells=9, n=200, max_new=128,  cap=16384, eos_fires=True),
    "rwkv7-world3__longbench": dict(rows=450,  cells=9, n=50,  max_new=512,  cap=32768, eos_fires=True),
    "gla-1.3b__longbench":     dict(rows=450,  cells=9, n=50,  max_new=512,  cap=32768, eos_fires=False),
    # EOS-fix rebuild. global-dit is a RELAY arm, so `decode`/`decode_path` are None by design --
    # it never touches greedy_loop. The relay exemption below is keyed off `ckpt_dir`, not off the
    # cell's name, so it cannot be used to wave through a baseline that merely looks relay-ish.
    "global-dit-50k__longbench": dict(rows=450, cells=9, n=50, max_new=512, cap=32768, eos_fires=True),
}

# Staging directory, overridable so one gate serves every rebuild wave.
STAGE = Path("results/benchmarks/longctx_decodefix")

def sc(v): return float(v) if isinstance(v, (int, float)) else float(v["score"])

def assert_independent(pa: Path, pb: Path) -> str:
    """Two files are a REPLICATE only if md5 AND elapsed_seconds both differ.

    2026-09-21: I cited three 'same-protocol 400/400' pairs as a determinism floor. All three were
    ONE file copied between directories -- md5 identical and `elapsed_seconds` equal to 13 significant
    figures -- so the agreement was tautological. A peer caught it. Equal elapsed_seconds is the
    tell-tale: two real runs never match past ~2 digits.
    Genuine replicates found after this check: rwkv7 longbench (4543.94 vs 4523.49) and gla longbench
    (2495.84 vs 2597.12), both 450/450 identical -- recurrent loop decode IS bit-reproducible, but that
    is a per-model measurement, never assume it for fp16 attention models.
    """
    import hashlib, json as _j
    ha, hb = (hashlib.md5(p.read_bytes()).hexdigest() for p in (pa, pb))
    ea, eb = (_j.loads(p.read_text()).get("elapsed_seconds") for p in (pa, pb))
    if ha == hb:
        return f"NOT INDEPENDENT: md5 identical ({ha[:8]}) -- same file copied, agreement is tautological"
    if ea == eb:
        return f"NOT INDEPENDENT: elapsed_seconds identical ({ea}) -- same run, agreement is tautological"
    return f"independent (md5 {ha[:8]}/{hb[:8]}, elapsed {ea:.2f}/{eb:.2f})"

def gate(name: str, dry: bool = False) -> bool:
    src = STAGE / f"{name}.json"
    if not src.exists():
        print(f"{name}: NOT STAGED YET"); return False
    d = json.loads(src.read_text()); cfg = d.get("config") or {}
    e = EXPECT[name]
    ags = {k: sc(v) for k, v in (d.get("aggregates") or {}).items()}
    ns = {(v.get("n") if isinstance(v, dict) else None) for v in (d.get("aggregates") or {}).values()}
    fails = []
    # Relay arms (ours) never call greedy_loop -- they decode inside LACESRuntime/GlobalRuntime, so
    # `decode_path` is legitimately None for them and the loop checks below do not apply. Keyed off
    # `ckpt_dir` (a relay cell HAS one and has no `model_dir`), never off the cell's name, so this
    # cannot be used to wave through a baseline. Their decode correctness is covered instead by the
    # stop-token checks further down, which apply to every cell regardless of path.
    relay = bool(cfg.get("ckpt_dir")) and not cfg.get("model_dir")
    if relay:
        print(f"  [relay arm: ckpt_dir set, no model_dir -> loop-path checks N/A, "
              f"stop-token checks still enforced]")
    else:
        # decode provenance -- explicit key presence, never a default (see ledger section 17 rule)
        if "decode_path" not in d: fails.append("decode_path key ABSENT")
        elif d["decode_path"] != "loop": fails.append(f"decode_path={d['decode_path']!r}")
        if d.get("decode_is_trustworthy") is not True:
            fails.append(f"decode_is_trustworthy={d.get('decode_is_trustworthy')!r}")
        if cfg.get("decode") != "loop": fails.append(f"config.decode={cfg.get('decode')!r}")
    # completeness
    if len(d.get("rows") or []) != e["rows"]: fails.append(f"rows={len(d.get('rows') or [])} != {e['rows']}")
    if len(ags) != e["cells"]: fails.append(f"cells={len(ags)} != {e['cells']}")
    if ns != {e["n"]}: fails.append(f"n per cell={ns} != {{{e['n']}}}")
    if cfg.get("max_new_tokens") != e["max_new"]: fails.append(f"max_new={cfg.get('max_new_tokens')}")
    if cfg.get("max_input_tokens") != e["cap"]: fails.append(f"max_input={cfg.get('max_input_tokens')}")
    # scorer provenance -- a cell must declare ONE scorer, never carry two comparable fields.
    # (2026-09-21: I differenced an official `score` against a legacy `score` and inflated a
    # patch delta 2.4x. A file with row-level `score_legacy` has been rescored; a RULER file with
    # NO metric_note is natively official, which reads backwards. Record it, don't infer it.)
    rows0 = (d.get("rows") or [{}])[0]
    if "score_legacy" in rows0 and not d.get("metric_note"):
        fails.append("rows carry score_legacy but metric_note is absent (ambiguous scorer)")
    # Scorer CONTROL. 2026-09-21: rescoring a cell's own UNMODIFIED text must return the score
    # already stored in it. I skipped this and reported a phantom "RULER changed 1 row" result that
    # was really two different scorer branches differenced against each other: RULER stores
    # `dataset` as `4096_qa_1` (`{length}_{task}`, no `/`), so a rescore that parses the task with
    # `split("/")[-1]` misses RULER_MATCH_PART_TASKS and silently uses string_match_all. The
    # control fails 26/1800 under that parse and 0/1800 under `split("_", 1)[1]`. This is the third
    # time the "differenced two scorers" family has bitten, so it is checked here rather than
    # trusted to discipline.
    try:
        import importlib.util as _il
        _s = _il.spec_from_file_location("_rocb", "scripts/eval/run_open_context_bench.py")
        _m = _il.module_from_spec(_s)
        sys.modules["_rocb"] = _m
        _s.loader.exec_module(_m)
    except Exception as exc:
        print(f"  [scorer control SKIPPED: cannot import runner ({exc})]")
    else:
        suite = (cfg.get("suite") or "").lower()
        bad = tot = 0
        for r in d.get("rows") or []:
            ds, text, want = r.get("dataset", ""), r.get("prediction", ""), r.get("score")
            if want is None:
                continue
            try:
                if suite == "ruler":
                    got = _m.ruler_score(ds.split("_", 1)[1], text, r["answers"])
                elif suite == "longbench":
                    got = max((_m.metric_for("longbench", ds)(text, a)
                               for a in r["answers"]), default=0.0)
                else:
                    continue  # babilong/infinitebench/memeval need extra fields; not covered here
            except Exception:
                continue
            tot += 1
            if abs(float(got) - float(want)) > 1e-9:
                bad += 1
        if tot:
            if bad:
                fails.append(f"scorer control FAILED {bad}/{tot}: rescoring the cell's own "
                             f"unmodified predictions does not reproduce its stored scores -- the "
                             f"stored `score` was not produced by this scorer, so any delta taken "
                             f"against it compares two different metrics")
            else:
                print(f"  [scorer control {tot}/{tot}: stored scores reproduce exactly]")

    if "ruler" in name:
        # official string_match_all is fractional on multi-gold tasks; legacy is binary everywhere
        scores = [float(r.get("score", 0)) for r in d.get("rows") or []]
        if not any(0.0 < s < 1.0 for s in scores):
            fails.append("RULER cell has no fractional score -> looks like the LEGACY scorer")
    # stop-token reachability. 2026-09-21: every check above passed on cells whose stop condition
    # could never fire. `greedy_loop` and the relay runtimes were handed only
    # `tokenizer.eos_token_id`, which for RWKV-7 is 65530 -- an id that appears in 0 rows of any
    # cell -- while the real end-of-text is 0 (generation_config) and config.json says 2. So every
    # row ran to its full budget, and on LongBench (the only F1/ROUGE-F1 suite) that cost macro
    # 0.0242 on the baseline and 0.0243 on our own global-dit arm. The containment suites changed
    # by exactly 0 rows, which is precisely why nothing here noticed.
    #
    # Two independent symptoms, both cheap, because either alone has honest exceptions:
    #   (a) `stop_ids` absent   -> the run predates the resolver; provenance unknown, not "fine".
    #   (b) every dataset has exactly ONE generation length -> nothing ever stopped early. A model
    #       that genuinely never emits EOS looks identical (gla is real: 0/2950 rows), so this is a
    #       warning unless (c) also holds.
    #   (c) a declared stop id appears INSIDE the generations -> proof the stop check did not fire.
    #       This is the only one that is conclusive on its own, so it is the hard failure.
    #
    # `short_rows` = rows ending BELOW their own dataset's budget, i.e. rows where something stopped
    # early. Computed here, before the branches, because BOTH (a) and (b) need it. 2026-09-21: the
    # first version of this counted distinct lengths across the whole cell, which mostly measures how
    # many per-dataset budgets the suite has (LongBench uses 32/64/128/512), not whether anything
    # stopped. champ-dynamic looked healthy at "33 distinct lengths" while only 32/450 of its rows
    # ever stopped.
    tok_key = next((k for k in ("prediction_token_ids", "generated_token_ids") if k in rows0), None)
    by_ds: dict = {}
    for r in d.get("rows") or []:
        by_ds.setdefault(r.get("dataset"), set()).add(len(r.get(tok_key) or []))
    short_rows = (sum(1 for r in d.get("rows") or []
                      if len(r.get(tok_key) or []) < max(by_ds[r.get("dataset")]))
                  if tok_key else None)
    if "stop_ids" not in d:
        # 2026-09-21: this used to be a WARNING, and that made the gate publish-unsafe. Run
        # without `--stage`, STAGE defaults to longctx_decodefix, whose rwkv7 longbench file is the
        # PRE-EOS-FIX 0.0955 run -- and the gate PASSED it, so `publish_gated_loop_cell.py
        # rwkv7-world3__longbench` would have silently overwritten the published 0.1197 with the
        # defective 0.0955. Every other check passes on that file precisely because the stop
        # condition was the thing that was broken. A missing `stop_ids` key means the run predates
        # the resolver, i.e. its stop condition is UNVERIFIABLE, and an unverifiable stop condition
        # is exactly the defect this gate exists to catch -- so it is now a hard failure. Rebuild
        # the cell, or point `--stage` at the directory holding the rebuilt one.
        fails.append("no `stop_ids` key -- this run predates the stop-id resolver, so its stop "
                     "condition cannot be verified and it may be a pre-EOS-fix file. Every other "
                     "check passes on such a file BECAUSE the stop condition is what was broken. "
                     "Refusing to publish; re-run the cell or fix --stage.")
    elif e.get("eos_fires") and not short_rows:
        # A cell that declares real stop ids but never stops is the original defect wearing the
        # fix's provenance fields. Only asserted for models measured to stop (see EXPECT).
        fails.append("declares stop_ids but 0 rows end below their own dataset budget -- the stop "
                     "set is present yet never fires, which is the pre-fix behaviour with post-fix "
                     "provenance")
    if "stop_ids" in d:
        declared = set(d.get("stop_ids") or [])
        if declared and tok_key:
            # An eos id INSIDE a generation is not itself a defect: on the `model.generate` path the
            # runtime stops at eos and then PADS the rest of the tensor (pad_token_id falls back to
            # eos_token_id), so the id is present by construction. 2026-09-21: auditing every cell
            # for "eos present" flagged 29 cells across 9 models; the pad/real-text test below
            # cleared all of the hf-path ones (3698 rows with eos, 3698 pure-pad tails, 0 with real
            # text) and kept exactly the 8 known-bad cells (1352 rows, 0 pure, 1352 real text).
            # The defect signature is CONTENT AFTER THE STOP, not the stop's presence.
            inside = {i: sum(1 for r in d.get("rows") or [] if i in (r.get(tok_key) or []))
                      for i in sorted(declared)}
            real_after = 0
            for r in d.get("rows") or []:
                t = r.get(tok_key) or []
                at = next((k for k, x in enumerate(t) if x in declared), None)
                if at is not None and not set(t[at:]) <= declared:
                    real_after += 1
            hit = {i: c for i, c in inside.items() if c}
            if real_after:
                fails.append(f"{real_after} row(s) contain REAL TEXT after a declared stop id "
                             f"{sorted(hit)} -> the stop check did not fire; these rows ran past "
                             f"their own EOS and the scorer saw the junk")
            elif hit:
                print(f"  [note: declared stop id(s) {sorted(hit)} appear inside generations, but "
                      f"every tail is pure eos/pad -> the stop fired and the rest is padding]")

            by_ds = {}
            for r in d.get("rows") or []:
                by_ds.setdefault(r.get("dataset"), set()).add(len(r.get(tok_key) or []))
            short = short_rows
            pinned = [k for k, v in by_ds.items() if len(v) == 1]
            if pinned and len(pinned) == len(by_ds):
                print(f"  ** WARNING: every one of {len(by_ds)} datasets has exactly ONE "
                      f"generation length and {short}/{len(d.get('rows') or [])} rows end below "
                      f"their dataset budget -> nothing stopped early. Legitimate only if this "
                      f"model truly never emits any of {sorted(declared)}. NOTE: you cannot "
                      f"verify that by checking whether a stop id appears in the stored ids -- "
                      f"generate() breaks BEFORE appending, so a stop that FIRED is invisible "
                      f"there. Use this row count instead.")

    # output health
    preds = [r.get("prediction", "") for r in d.get("rows") or []]
    empty = sum(1 for p in preds if not p.strip())
    macro = sum(ags.values()) / len(ags) if ags else float("nan")
    print(f"{name}: macro={macro:.4f} rows={len(preds)} cells={len(ags)} n={ns} "
          f"empty={empty} distinct={len(set(preds))} elapsed_h={(d.get('elapsed_seconds') or 0)/3600:.3f} "
          f"bos={cfg.get('prepend_bos')}")
    for k in sorted(ags): print(f"    {k:<24} {ags[k]:.4f}")
    if fails:
        print("  ** GATE FAILED: " + "; ".join(fails)); return False
    dst = Path(f"results/benchmarks/longctx/{name}.json")
    if dry:
        # 2026-09-21: running this script to *check* a cell silently re-published one. It was a
        # byte-identical no-op that time, but a gate you cannot run without mutating canonical is
        # a gate nobody will run. --dry-run reports the verdict and writes nothing.
        print(f"  GATE PASSED (--dry-run: would publish to {dst}, nothing written)")
        return True
    # Preserve the predecessor before overwriting. `results/` is gitignored, so an overwrite is
    # otherwise unrecoverable -- and this wave overwrites cells that were themselves gated, which
    # is exactly when the old file is worth keeping for a before/after audit.
    if dst.exists():
        keep = Path("results/benchmarks/superseded_eosfix") / f"{name}.json"
        keep.parent.mkdir(parents=True, exist_ok=True)
        if not keep.exists():
            shutil.copy2(dst, keep)
            print(f"  predecessor preserved -> {keep}")
        else:
            print(f"  predecessor already preserved at {keep} (not overwritten)")
    shutil.copy2(src, dst)
    import hashlib
    h = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
    assert h(src) == h(dst), "copy mismatch"
    print(f"  GATE PASSED -> published to {dst} (md5 verified)")
    return True

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--stage" in args:
        i = args.index("--stage")
        STAGE = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    # 2026-09-21: `gate()` grew a `dry` parameter and a --dry-run docstring, but nothing here ever
    # parsed the flag -- so `--dry-run` fell through as a CELL NAME, printed "NOT STAGED YET" for
    # it, and then published every other named cell FOR REAL. The only thing that stopped a live
    # overwrite was that src and dst happened to be the same path (SameFileError). A flag that
    # silently means its own opposite is worse than no flag, so it is parsed here and any
    # unrecognised `--flag` is now a hard error rather than a cell name.
    dry = False
    for flag in ("--dry-run", "--dry"):
        if flag in args:
            dry = True
            args = [a for a in args if a != flag]
    unknown = [a for a in args if a.startswith("-")]
    if unknown:
        sys.exit(f"unknown option(s) {unknown}; supported: --stage DIR, --dry-run")
    bad = [a for a in args if a not in EXPECT]
    if bad:
        sys.exit(f"unknown cell(s) {bad}; known: {sorted(EXPECT)}")
    for n in (args or list(EXPECT)):
        gate(n, dry=dry); print()

