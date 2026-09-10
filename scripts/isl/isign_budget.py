"""
Decide the epoch budget from MEASURED speed, not from anyone's estimate.

Throughput predictions for this pipeline are unreliable - they depend on the
card, the clip lengths that survived filtering, dataloader workers and disk.
Guessing wrong on a $7 ceiling means either running out of money mid-stage-2
(a wasted stage 1) or under-spending.

So: launch stage 1, let it run ~100 iterations, read the real seconds/iteration
off the log, kill it, and run this. It converts measured speed into the epoch
counts you can actually afford, then you set ISIGN_S1_EPOCHS / ISIGN_S2_EPOCHS
and start for real. Because the cosine LR schedule is defined over max_epochs,
this has to be decided BEFORE the run you keep - which is exactly why this
calibration step exists.

Usage:
    # after ~100 iterations of stage 1
    python scripts/isl/isign_budget.py \
        --budget_usd 6.0 --gpu_usd_per_hour 0.34 \
        --train_clips 2400 --dev_clips 300 \
        --bs 2 --s1_sec_per_iter 1.9

    # once you also know stage 2's speed
    python scripts/isl/isign_budget.py ... --s2_sec_per_iter 3.4
"""

import argparse


def fmt_h(h):
    return f"{h:5.2f} h" if h >= 1 else f"{h * 60:5.1f} m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_usd", type=float, required=True,
                    help="Money you are willing to spend on GPU time (keep a reserve)")
    ap.add_argument("--gpu_usd_per_hour", type=float, required=True)
    ap.add_argument("--train_clips", type=int, required=True)
    ap.add_argument("--dev_clips", type=int, default=0,
                    help="Dev split size - per-epoch validation is not free")
    ap.add_argument("--bs", type=int, required=True, help="cfg.bs actually used")
    ap.add_argument("--s1_sec_per_iter", type=float, required=True)
    ap.add_argument("--s2_sec_per_iter", type=float, default=None,
                    help="If unknown, assumed 1.8x stage 1 (LM forward+backward)")
    ap.add_argument("--s1_share", type=float, default=0.4,
                    help="Fraction of the budget for stage 1 (default 0.4)")
    ap.add_argument("--beam_eval_factor", type=float, default=6.0,
                    help="Beam-search test pass cost relative to a training iteration")
    args = ap.parse_args()

    s2_spi = args.s2_sec_per_iter or args.s1_sec_per_iter * 1.8
    assumed = args.s2_sec_per_iter is None

    train_iters = max(1, args.train_clips // args.bs)
    dev_iters = max(0, args.dev_clips // args.bs)

    # per-epoch wall clock
    s1_epoch_h = (train_iters + dev_iters) * args.s1_sec_per_iter / 3600
    s2_epoch_h = (train_iters + dev_iters) * s2_spi / 3600
    # the beam-search tester fires every 10 epochs on the dev split
    beam_h = dev_iters * s2_spi * args.beam_eval_factor / 3600

    budget_h = args.budget_usd / args.gpu_usd_per_hour
    s1_h = budget_h * args.s1_share
    s2_h = budget_h - s1_h

    e1 = int(s1_h / s1_epoch_h)
    # stage 2 must reach at least 10 epochs or the tester never runs
    e2 = int((s2_h - beam_h) / s2_epoch_h)

    print()
    print(f"  GPU                {args.gpu_usd_per_hour:.2f} $/h -> {budget_h:.1f} h for ${args.budget_usd:.2f}")
    print(f"  iterations/epoch   {train_iters} train + {dev_iters} dev  (bs={args.bs})")
    print(f"  stage 1 epoch      {fmt_h(s1_epoch_h)}   @ {args.s1_sec_per_iter:.2f} s/iter")
    print(f"  stage 2 epoch      {fmt_h(s2_epoch_h)}   @ {s2_spi:.2f} s/iter"
          f"{'  (ASSUMED 1.8x - re-run once measured)' if assumed else ''}")
    print(f"  beam test pass     {fmt_h(beam_h)}   (fires every 10 epochs)")
    print()
    print(f"  AFFORDABLE:  ISIGN_S1_EPOCHS={max(1, e1)}   ISIGN_S2_EPOCHS={max(1, e2)}")
    total_h = e1 * s1_epoch_h + e2 * s2_epoch_h + beam_h * max(1, e2 // 10)
    print(f"  projected total    {fmt_h(total_h)}  =  ${total_h * args.gpu_usd_per_hour:.2f}")
    print()

    if e2 < 10:
        print("  WARNING: stage 2 cannot reach 10 epochs, so the beam-search tester")
        print("           never fires and you get NO autoregressive BLEU. Fix by")
        print("           shrinking the dataset (fewer clips, or a tighter")
        print("           --max_src_frames), not by accepting 9 epochs.")
        need = (10 * s2_epoch_h + beam_h + e1 * s1_epoch_h) * args.gpu_usd_per_hour
        print(f"           A 10-epoch stage 2 with this data needs ~${need:.2f} total.")
    if e1 < 6:
        print("  WARNING: fewer than ~6 stage-1 epochs rarely moves class_f1 off the")
        print("           floor. Prefer fewer clips over fewer epochs.")
    print("  Reserve ~20% of your real balance: pods bill while you debug.")
    print()


if __name__ == "__main__":
    main()
