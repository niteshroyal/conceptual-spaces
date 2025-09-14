#!/usr/bin/env python3
import argparse, re, math
from collections import defaultdict


# # Example usage:
# python3 mcnemar_taste.py \
#   --classification /path/to/hits_classification.txt \
#   --rank_class /path/to/hits_ranking_and_classification.txt



def parse_hits(path):
    """
    Parses files like:
        'sweet taste food item : 1'
    Returns a list of (label, int{0,1}), preserving order.
    """
    rows = []
    pat = re.compile(r"^(.*?):\s*([01])\s*$")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # tolerate both "sweet taste food item : 1" and "sweet taste food item: 1"
            if " : " in line:
                left, right = line.rsplit(":", 1)
            else:
                m = pat.match(line)
                if not m:
                    continue
                left, right = m.group(1), m.group(2)
            label = left.strip().lower()
            val = int(right.strip())
            rows.append((label, val))
    return rows

def group_pairs_by_label(rows_a, rows_b):
    """
    Pairs predictions by position within each label (category).
    Assumes both files enumerate the SAME dataset in the SAME order.
    Returns dict[label] -> list of (a, b) where
        a = classification-only (baseline)
        b = ranking+classification (augmented)
    """
    # split each file into per-label sequences preserving order
    seq_a = defaultdict(list)
    seq_b = defaultdict(list)
    for lab, v in rows_a:
        seq_a[lab].append(v)
    for lab, v in rows_b:
        seq_b[lab].append(v)

    labels = sorted(set(seq_a.keys()) & set(seq_b.keys()))
    pairs = {}
    for lab in labels:
        la, lb = seq_a[lab], seq_b[lab]
        if len(la) != len(lb):
            print(f"[WARN] Label '{lab}' has different lengths: A={len(la)} vs B={len(lb)}. "
                  f"Using min length to pair.")
        n = min(len(la), len(lb))
        pairs[lab] = list(zip(la[:n], lb[:n]))
    return pairs

def binom_two_sided_p(b, c):
    """
    Exact McNemar two-sided p-value via binomial test with p=0.5 on n=b+c and x=min(b,c).
    """
    from math import comb
    n = b + c
    x = min(b, c)
    # two-sided: sum of tail probabilities up to x
    tail = sum(comb(n, k) for k in range(0, x+1)) / (2 ** n)
    p = 2 * tail
    return min(1.0, p)

def mcnemar_chi2_cc(b, c):
    """
    McNemar's chi-square with continuity correction (Edwards, 1948):
        X2 = (|b - c| - 1)^2 / (b + c)
    Returns (statistic, p_value) with chi-square df=1.
    """
    num = abs(b - c) - 1.0
    stat = (num * num) / (b + c) if (b + c) > 0 else 0.0
    # chi-square survival function with 1 df: sf(x) = exp(-x/2)
    # (because for 1 df, chi-square is exp with rate 1/2)
    p = math.exp(-stat / 2.0)
    return stat, p

def evaluate_category(pairs, use_exact=True, exact_threshold=25):
    """
    For a list of (a, b) pairs (a=classification-only, b=ranking+classification),
    build contingency:
        b = A correct & B wrong  -> n10
        c = A wrong  & B correct -> n01
    Also compute per-model accuracy.
    """
    a_correct = sum(1 for a, _ in pairs if a == 1)
    b_correct = sum(1 for _, b in pairs if b == 1)
    n = len(pairs)
    n10 = sum(1 for a, b in pairs if a == 1 and b == 0)  # baseline correct, augmented wrong
    n01 = sum(1 for a, b in pairs if a == 0 and b == 1)  # baseline wrong, augmented correct

    # choose exact binomial when small discordant counts (recommended), else chi-square CC
    b_plus_c = n10 + n01
    if use_exact or b_plus_c < exact_threshold:
        p = binom_two_sided_p(n10, n01)
        stat = None
        test_name = "McNemar exact (binomial)"
    else:
        stat, p = mcnemar_chi2_cc(n10, n01)
        test_name = "McNemar chi-square (CC)"

    return {
        "N": n,
        "acc_class_only": a_correct / n if n else float('nan'),
        "acc_rank_class": b_correct / n if n else float('nan'),
        "n10_baseline_only": n10,
        "n01_augmented_only": n01,
        "discordant": b_plus_c,
        "test": test_name,
        "stat": stat,
        "p": p,
    }

def main():
    ap = argparse.ArgumentParser(description="McNemar test per Taste subset")
    ap.add_argument("--classification", required=True,
                    help="Path to hits file for classification-only (baseline) (e.g., hits_classification.txt)")
    ap.add_argument("--rank_class", required=True,
                    help="Path to hits file for ranking+classification (augmented) (e.g., hits_ranking_and_classification.txt)")
    ap.add_argument("--use-chi2", action="store_true",
                    help="Force chi-square with continuity correction instead of exact binomial")
    ap.add_argument("--exact-threshold", type=int, default=25,
                    help="If not forcing chi2, use exact binomial when (n10+n01) < threshold (default: 25)")
    args = ap.parse_args()

    rows_cls = parse_hits(args.classification)
    rows_aug = parse_hits(args.rank_class)
    pairs_by_label = group_pairs_by_label(rows_cls, rows_aug)

    # Only keep the six taste categories (case-insensitive startswith is fine)
    wanted = ["sweet", "salty", "sour", "bitter", "umami", "fatty"]
    labels = [lab for lab in pairs_by_label.keys() if any(lab.startswith(w) for w in wanted)]

    print("\nMcNemar significance testing (classification-only vs ranking+classification)\n")
    header = ("Label", "N", "Acc_cls", "Acc_rank+cls", "n10 A✓B✗", "n01 A✗B✓", "Disc", "Test", "Stat", "p-value")
    print("{:<10s} {:>5s} {:>8s} {:>12s} {:>11s} {:>11s} {:>6s} {:>26s} {:>8s} {:>10s}".format(*header))
    for lab in sorted(labels):
        res = evaluate_category(
            pairs_by_label[lab],
            use_exact=not args.use_chi2,
            exact_threshold=args.exact_threshold
        )
        stat_str = "-" if res["stat"] is None else f"{res['stat']:.4f}"
        print("{:<10s} {:>5d} {:>8.3f} {:>12.3f} {:>11d} {:>11d} {:>6d} {:>26s} {:>8s} {:>10.4g}".format(
            lab.split()[0],  # print just 'sweet', 'salty', ...
            res["N"],
            res["acc_class_only"] * 100,
            res["acc_rank_class"] * 100,
            res["n10_baseline_only"],
            res["n01_augmented_only"],
            res["discordant"],
            res["test"],
            stat_str,
            res["p"]
        ))
    print()

if __name__ == "__main__":
    main()
