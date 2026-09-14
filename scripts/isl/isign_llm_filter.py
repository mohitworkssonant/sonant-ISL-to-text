"""
LLM-assisted text filtering, in two halves so any LLM can do the middle bit.

    export  ->  batched prompts you paste/pipe into whatever model you have
    apply   ->  turn the model's verdicts into a keeplist for the subset builder

WHAT AN LLM CAN AND CANNOT JUDGE HERE
-------------------------------------
It only sees the English text. So it can spot rows that are not transcriptions
of a signed utterance at all - on-screen headings, captions referring to
diagrams, instructions to the reader, fragments - which heuristics miss because
they are grammatical.

It CANNOT see whether the video actually contains the signing that matches the
text. iSign is auto-segmented (ISLRTC by audio pauses, ISH by caption
timestamps), so misalignment is the dominant noise source, and no text-only
filter touches it. Do not let a clean-looking filtered corpus imply clean
alignment.

ORDER OF OPERATIONS
-------------------
Run isign_corpus_stats.py FIRST. Heuristics and the vocabulary cap are free,
deterministic, and remove far more rows than an LLM will. Send the LLM only
what survives - typically a 20-30k candidate pool rather than all 127k, which
is the difference between a few hundred calls and a few thousand.

    python scripts/isl/isign_corpus_stats.py --csv iSign_v1.1.csv \
        --write_keeplist coherent.txt --vocab_cap 3000
    python scripts/isl/isign_llm_filter.py export --csv iSign_v1.1.csv \
        --keeplist coherent.txt --out prompts.jsonl --batch 40
    # ... run prompts.jsonl through your LLM, collect replies into replies.jsonl
    python scripts/isl/isign_llm_filter.py apply --replies replies.jsonl \
        --keeplist coherent.txt --out keep_final.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isign_build_subset import clean_text                     # noqa: E402

INSTRUCTION = """You are cleaning a sign-language translation dataset. Each line is an
English sentence that is supposed to be the translation of one short signed video clip.

Mark a line KEEP if it is a self-contained utterance a person could plausibly sign:
a statement, question, instruction or exclamation that stands on its own.

Mark it DROP if it is:
 - a heading, title, page/chapter/section marker, or list item
 - text referring to something visual the signer cannot sign ("see the diagram above")
 - a sentence fragment that does not stand alone
 - metadata, a URL, a file name, or transcription noise
 - so generic it carries no content ("yes", "okay then", "thank you very much")

Reply with one line per item, exactly: <id> KEEP  or  <id> DROP
No explanations."""


def cmd_export(a):
    df = pd.read_csv(a.csv)
    uid = next(c for c in df.columns if c.lower() in ("uid", "id", "clip_id", "name"))
    txt = next(c for c in df.columns if c.lower() in ("text", "translation", "sentence"))
    df = df.dropna(subset=[uid, txt])
    df["name"] = df[uid].astype(str).str.strip()

    if a.keeplist:
        keep = {ln.strip() for ln in Path(a.keeplist).read_text().splitlines() if ln.strip()}
        before = len(df)
        df = df[df["name"].isin(keep)]
        print(f"[llm] keeplist: {before:,} -> {len(df):,} rows to judge")
    if a.limit:
        df = df.head(a.limit)

    # One prompt per batch of rows; short ids keep the token cost down.
    n = 0
    with open(a.out, "w") as f:
        for start in range(0, len(df), a.batch):
            chunk = df.iloc[start:start + a.batch]
            lines = [f"{i+1}. {clean_text(t)}" for i, t in enumerate(chunk[txt].values)]
            f.write(json.dumps({
                "batch": n,
                "ids": chunk["name"].tolist(),
                "prompt": INSTRUCTION + "\n\n" + "\n".join(lines),
            }) + "\n")
            n += 1
    est_tok = len(df) * 18 + n * 170
    print(f"[llm] wrote {n:,} prompts covering {len(df):,} rows -> {a.out}")
    print(f"[llm] roughly {est_tok/1000:.0f}k input tokens in total")
    print("[llm] Each line's 'prompt' is one request; keep 'ids' with the reply so")
    print("      'apply' can line them back up. Reply format expected:")
    print('      {"batch": 0, "reply": "1 KEEP\\n2 DROP\\n..."}')


def cmd_apply(a):
    keep_ids, drop_ids, seen = [], [], 0
    for line in Path(a.replies).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ids = rec.get("ids")
        if ids is None:
            sys.exit("[err] replies must carry the same 'ids' list as the export")
        verdicts = {}
        for m in re.finditer(r"^\s*(\d+)\s*[.\):]?\s*(KEEP|DROP)\s*$",
                             rec["reply"], re.MULTILINE | re.IGNORECASE):
            verdicts[int(m.group(1))] = m.group(2).upper()
        for i, cid in enumerate(ids, start=1):
            seen += 1
            v = verdicts.get(i)
            # An unparsed line is kept: a filter that silently discards data
            # when the model rambles is worse than one that lets a little noise
            # through. Count them so the ratio is visible.
            (drop_ids if v == "DROP" else keep_ids).append(cid)

    unparsed = seen - sum(1 for _ in keep_ids) - len(drop_ids)
    base = None
    if a.keeplist:
        base = {ln.strip() for ln in Path(a.keeplist).read_text().splitlines() if ln.strip()}
        keep_ids = [c for c in keep_ids if c in base]

    Path(a.out).write_text("\n".join(keep_ids) + "\n")
    print(f"[llm] judged {seen:,} rows: KEEP {len(keep_ids):,} | DROP {len(drop_ids):,}"
          f"  ({100*len(drop_ids)/max(1,seen):.1f}% dropped)")
    if unparsed:
        print(f"[llm] {unparsed:,} verdicts could not be parsed and were kept")
    print(f"[llm] wrote {a.out}")
    print(f"[llm] next:  python scripts/isl/isign_build_subset.py --csv <csv> "
          f"--keeplist {a.out} --output_dir data/isl --max_clips 8000 --max_clips_per_video 5")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export")
    e.add_argument("--csv", required=True)
    e.add_argument("--keeplist", default=None)
    e.add_argument("--out", default="prompts.jsonl")
    e.add_argument("--batch", type=int, default=40)
    e.add_argument("--limit", type=int, default=0)
    e.set_defaults(fn=cmd_export)

    p = sub.add_parser("apply")
    p.add_argument("--replies", required=True)
    p.add_argument("--keeplist", default=None, help="Intersect with this (the heuristic keeplist)")
    p.add_argument("--out", default="keep_final.txt")
    p.set_defaults(fn=cmd_apply)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
