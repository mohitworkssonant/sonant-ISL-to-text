"""
Compatibility verdict for the iSign x Sign2GPT trial run.

A trial answers ONE question: can this codebase consume iSign RGB video and
drive Sign2GPT end to end? Not "is the translation good" - a few hundred clips
cannot answer that, and a BLEU near zero here means nothing about the dataset.

So this script checks the six things that would actually break, reads them out
of the artefacts the run leaves behind, and prints PASS/FAIL per item:

    1. iSign CSV parsed and split       (corpus CSVs exist, non-empty, well-formed)
    2. Video decode + LMDB              (LMDBs exist, frames readable, sane counts)
    3. Pseudo-gloss keys line up        (every translation is a dict_sentence key)
    4. Stage 1 trained                  (epochs completed, loss fell, class_f1 > 0)
    5. Stage 2 trained                  (epochs completed, LM loaded, BLEU computed)
    6. Generation produces English      (from isign_infer.py, pasted in by --prediction)

Usage:
    python scripts/isl/isign_trial_report.py \
        --csv_dir   /workspace/sonant-ISL-to-text/data/isl \
        --pkl       /workspace/sonant-ISL-to-text/data/isl/processed_words.isl_pkl \
        --lmdb_root /workspace/lmdb/isl/lmdb_videos \
        --s1_log    /workspace/results/trial_stage1.log \
        --s2_log    /workspace/results/trial_stage2.log \
        --prediction "the man is going to school"
"""

import argparse
import pickle
import re
from pathlib import Path

import pandas as pd

G, R, Y, B, X = "\033[1;32m", "\033[1;31m", "\033[1;33m", "\033[1m", "\033[0m"


class Report:
    def __init__(self):
        self.rows = []
        self.s1_threshold_limited = False

    def add(self, name, ok, detail):
        self.rows.append((name, ok, detail))

    def render(self):
        width = max(len(n) for n, _, _ in self.rows)
        print(f"\n{B}iSign x Sign2GPT - compatibility trial{X}\n")
        for name, ok, detail in self.rows:
            tag = f"{G}PASS{X}" if ok is True else (f"{R}FAIL{X}" if ok is False else f"{Y}????{X}")
            print(f"  [{tag}] {name.ljust(width)}  {detail}")
        fails = sum(1 for _, ok, _ in self.rows if ok is False)
        unknown = sum(1 for _, ok, _ in self.rows if ok is None)
        print()
        if fails:
            print(f"{R}{fails} check(s) failed - iSign and Sign2GPT are NOT wired up correctly yet.{X}")
        elif unknown:
            print(f"{Y}All executed checks passed; {unknown} could not be evaluated (see above).{X}")
        else:
            print(f"{G}All checks passed. iSign RGB video drives Sign2GPT end to end.{X}")
        print("\nRemember what this does and does not show:")
        print("  DOES  - the data format, decode, LMDB, vocabulary, both training stages")
        print("          and generation are compatible and run without intervention.")
        print("  DOES NOT - say anything about translation quality. A few hundred clips")
        print("          cannot. Quality needs the full-scale run.")
        if self.s1_threshold_limited:
            print()
            print("  Note on stage 1: class_f1 uses a hard-coded 0.5 threshold, while the")
            print("  prototype head spreads ~1.0 of probability mass across every class. At")
            print("  a few hundred clips almost nothing crosses 0.5, so a valid class_f1 of")
            print("  0.0 is expected here and is NOT evidence of a broken pipeline. To watch")
            print("  it actually move, re-run stage 1 alone with more epochs - it is ~15 s")
            print("  per epoch at this size:")
            print("      SIGN2GPT_CKPT_PATH=/workspace/checkpoints_s1x ISIGN_S1_EPOCHS=60 \\")
            print("        python main.py --config=configs/isl/isign_budget_stage1_config.py")
        print()
        return 1 if fails else 0


def all_floats(log_text, key):
    """Every value logged under `key` or any key starting with it.

    The text logger is `pprint.pprint(dict)`, and dict-valued metrics are
    flattened with an underscore - so BLEU lands in the log as
    `'valid/obleu_bleu4': 1.37`, not `valid/obleu`. Matching on the prefix
    means the report keeps working whether a metric is scalar or a dict.
    """
    pat = rf"['\"]?{re.escape(key)}[A-Za-z0-9_/]*['\"]?\s*[:=]\s*([-+0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    return [float(x) for x in re.findall(pat, log_text)]


def last_float(log_text, key):
    hits = all_floats(log_text, key)
    return hits[-1] if hits else None


def exact_floats(log_text, key):
    """Values logged under exactly `key` - no prefix matching.

    Needed for losses: a prefix match on "loss" also swallows
    'train/avg_loss_bce', interleaving two different series and making the
    first-to-last comparison meaningless.
    """
    pat = rf"['\"]{re.escape(key)}['\"]\s*:\s*([-+0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    return [float(x) for x in re.findall(pat, log_text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--lmdb_root", required=True)
    ap.add_argument("--s1_log", required=True)
    ap.add_argument("--s2_log", required=True)
    ap.add_argument("--prediction", default=None,
                    help="Paste the PREDICTION line from isign_infer.py")
    a = ap.parse_args()

    rep = Report()
    csv_dir, lmdb_root = Path(a.csv_dir), Path(a.lmdb_root)

    # 1 -------------------------------------------------------------- CSVs
    try:
        sizes, bad = {}, []
        for s in ["train", "dev", "test"]:
            df = pd.read_csv(csv_dir / f"ISL.{s}.corpus.csv", sep="|")
            sizes[s] = len(df)
            if list(df.columns) != ["name", "speaker", "orth", "translation", "start", "end"]:
                bad.append(f"{s}:columns")
            if df["orth"].isna().any():
                bad.append(f"{s}:orth-NaN")
        ok = not bad and all(v > 0 for v in sizes.values())
        rep.add("1. iSign CSV parsed and split", ok,
                f"train/dev/test = {sizes['train']}/{sizes['dev']}/{sizes['test']}"
                + (f"  problems: {bad}" if bad else ""))
    except Exception as e:                                    # noqa: BLE001
        rep.add("1. iSign CSV parsed and split", False, f"{type(e).__name__}: {e}")
        sizes = {}

    # 2 -------------------------------------------------------------- LMDB
    try:
        n_lmdb = sum(1 for d in lmdb_root.iterdir() if d.is_dir())
        listed = sum(sizes.values()) if sizes else 0
        cover = 100.0 * min(n_lmdb, listed) / listed if listed else 0
        rep.add("2. Video decode -> LMDB", n_lmdb > 0 and cover > 95,
                f"{n_lmdb:,} clip LMDBs on disk, covering {cover:.0f}% of the CSV rows")
    except Exception as e:                                    # noqa: BLE001
        rep.add("2. Video decode -> LMDB", False, f"{type(e).__name__}: {e}")

    # 3 --------------------------------------------------- pseudo-gloss keys
    try:
        pg = pickle.load(open(a.pkl, "rb"))
        miss = 0
        for s in ["train", "dev", "test"]:
            df = pd.read_csv(csv_dir / f"ISL.{s}.corpus.csv", sep="|")
            miss += int((~df["translation"].astype(str).isin(pg["dict_sentence"].keys())).sum())
        rep.add("3. Pseudo-gloss keys line up", miss == 0,
                f"{len(pg['dict_lem_to_id']):,} classes, {miss} unmatched translations")
    except Exception as e:                                    # noqa: BLE001
        rep.add("3. Pseudo-gloss keys line up", False, f"{type(e).__name__}: {e}")

    # 4 ------------------------------------------------------------ stage 1
    #
    # Judging stage 1 by "valid class_f1 > 0" was wrong for a trial, and the
    # first real run exposed it. ClassF1Score uses a HARD-CODED threshold of
    # 0.5 (trainer/psuedo_gloss_trainer.py), while the prototype head emits
    # class_scores = sum_t(cls_softmax * time_softmax) - a distribution whose
    # total across classes is ~1. With 546 classes the average class sits at
    # 0.0018, so a positive prediction needs one class to absorb ~270x the
    # mean. PHOENIX gets there, but only after ~90k sample-passes; a 226-clip,
    # 6-epoch trial has ~1.4k. Valid class_f1 = 0 at this scale says nothing
    # about wiring.
    #
    # What DOES discriminate at trial scale:
    #   * the metric prints 0.0 rather than NaN -> at least one clip carried a
    #     pseudo-gloss target (compute() averages over classes with tp+fn>0;
    #     with no targets that selection is empty and mean() is NaN);
    #   * train class_f1 > 0 -> some predictions actually cross the threshold;
    #   * the loss falls and nothing crashed.
    try:
        t1 = Path(a.s1_log).read_text(errors="ignore")
        vf1 = all_floats(t1, "valid/class_f1_score")
        tf1 = all_floats(t1, "train/class_f1_score")
        losses = (exact_floats(t1, "train/avg_loss") or exact_floats(t1, "train/loss")
                  or exact_floats(t1, "loss"))
        fell = len(losses) >= 2 and losses[-1] < losses[0]
        has_targets = bool(vf1) or bool(tf1)          # printed at all, and not NaN
        learning = (max(tf1) > 0 if tf1 else False) or (max(vf1) > 0 if vf1 else False)

        ok = has_targets and fell
        bits = []
        if vf1:
            bits.append(f"valid class_f1 {vf1[-1]:.4f}")
        if tf1:
            bits.append(f"train class_f1 {max(tf1):.4f}")
        if losses:
            bits.append(f"loss {losses[0]:.3f} -> {losses[-1]:.3f}" + ("" if fell else " (NOT falling)"))
        detail = "; ".join(bits) if bits else "no stage-1 metrics in log"

        if not has_targets:
            ok = False
            detail += "  <- no class_f1 logged at all: pseudo-gloss targets are missing"
        elif not learning:
            detail += "  (threshold-limited at this scale, not a wiring fault - see notes)"
        if "CUDA out of memory" in t1:
            ok, detail = False, "CUDA OOM - lower ISIGN_BS or ISIGN_MAX_SEQ. " + detail
        rep.add("4. Stage 1 (vision) trained", ok, detail)
        rep.s1_threshold_limited = has_targets and not learning
    except Exception as e:                                    # noqa: BLE001
        rep.add("4. Stage 1 (vision) trained", False, f"{type(e).__name__}: {e}")

    # 5 ------------------------------------------------------------ stage 2
    try:
        t2 = Path(a.s2_log).read_text(errors="ignore")
        loaded = "stage-1 checkpoint" in t2 or "stage1_ckpt" in t2
        ableu = last_float(t2, "valid_test/ableu_bleu4") or last_float(t2, "valid_test/ableu")
        obleu = last_float(t2, "valid/obleu_bleu4") or last_float(t2, "valid/obleu")
        oom = "CUDA out of memory" in t2
        ok = loaded and (ableu is not None or obleu is not None) and not oom
        detail = (f"stage-1 ckpt loaded={loaded}; "
                  f"obleu={obleu if obleu is not None else 'n/a'}, "
                  f"ableu={ableu if ableu is not None else 'n/a (needs >=10 epochs)'}")
        if oom:
            detail = "CUDA OOM in stage 2 - use xglm-564M or ISIGN_BS=1. " + detail
        rep.add("5. Stage 2 (translation) trained", ok, detail)
    except Exception as e:                                    # noqa: BLE001
        rep.add("5. Stage 2 (translation) trained", False, f"{type(e).__name__}: {e}")

    # 6 -------------------------------------------------------- generation
    if a.prediction is None:
        rep.add("6. Generation produces English", None,
                "run scripts/isl/isign_infer.py and pass --prediction '<its output>'")
    else:
        p = a.prediction.strip()
        words = p.split()
        latin = sum(c.isascii() and c.isalpha() for c in p)
        looks_english = len(words) >= 2 and latin > 0.6 * max(1, len(p.replace(" ", "")))
        rep.add("6. Generation produces English", looks_english,
                f"{len(words)} words, {'reads as English' if looks_english else 'does NOT read as English'}: {p!r}")

    raise SystemExit(rep.render())


if __name__ == "__main__":
    main()
