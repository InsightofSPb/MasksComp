#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# выделяем суффикс ..._tau0.10_k5 в конце строки
RE_TAU_K = re.compile(r"^(?P<base>.*)_tau(?P<tau>[0-9.]+)_k(?P<k>[0-9]+)$")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--only-parsed", action="store_true")
    p.add_argument("--paper-k", type=str, default="5,50")
    p.add_argument("--paper-tau", type=str, default="0.10,0.30")
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    df["method"] = df["method"].astype(str).str.strip()
    if "tau" in df.columns:
        df["tau"] = pd.to_numeric(df["tau"], errors="coerce")

    tau_in_method = []
    k_in_method = []
    method_base = []

    for m in df["method"].tolist():
        mm = RE_TAU_K.match(m)
        if mm:
            method_base.append(mm.group("base"))
            tau_in_method.append(float(mm.group("tau")))
            k_in_method.append(int(mm.group("k")))
        else:
            method_base.append(m)
            tau_in_method.append(None)
            k_in_method.append(None)

    df["tau_in_method"] = pd.to_numeric(tau_in_method, errors="coerce")
    df["k_in_method"] = pd.to_numeric(k_in_method, errors="coerce")
    df["method_base"] = method_base

    # эффективные tau/k: сначала берём из method, иначе из колонок (если есть)
    df["tau_eff"] = df["tau_in_method"]
    if "tau" in df.columns:
        df["tau_eff"] = df["tau_eff"].fillna(df["tau"])

    df["k_eff"] = df["k_in_method"]
    if "k" in df.columns:
        df["k_eff"] = df["k_eff"].fillna(pd.to_numeric(df["k"], errors="coerce"))

    if args.only_parsed:
        df = df[df["k_eff"].notna()].copy()

    show_cols = ["ROC-AUC", "PR-AUC", "Hit@K", "Recall@K", "n_pairs"]
    for c in show_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    key_cols = ["dataset", "split", "label_from", "method_base", "tau_eff", "k_eff", "n_pairs"]
    key_cols = [c for c in key_cols if c in df.columns]  # на случай если чего-то нет
    df = df.sort_index().drop_duplicates(subset=key_cols, keep="last").copy()

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)

    cols = ["method_base", "tau_eff", "k_eff"] + [c for c in show_cols if c in df.columns]

    print("\nAVAILABLE method_base values:")
    print(sorted(df["method_base"].unique().tolist()))

    print("\nFULL parsed table (first 30 rows):")
    print(df[cols].sort_values(["method_base", "tau_eff", "k_eff"]).head(30).to_string(index=False))

    paper_k = [int(x) for x in args.paper_k.split(",") if x.strip()]
    paper_tau = [float(x) for x in args.paper_tau.split(",") if x.strip()]

    d2 = df[df["tau_eff"].isin(paper_tau) & df["k_eff"].isin(paper_k)].copy()
    if len(d2) == 0:
        print("\nPAPER VIEW: no rows match filters.")
        return

    d2 = d2.sort_values(["method_base", "tau_eff", "k_eff"])

    print("\nPAPER VIEW (method_base x tau x k):")
    print(d2[cols].to_string(index=False))

    for k in paper_k:
        dk = d2[d2["k_eff"] == k].copy()
        if dk.empty:
            continue
        pv = dk.pivot_table(index="method_base", columns="tau_eff",
                            values=[c for c in ["ROC-AUC", "PR-AUC"] if c in dk.columns],
                            aggfunc="first")
        pv.columns = [f"{a}@tau={b:g}" for a, b in pv.columns]
        print(f"\nCOMPACT ROC/PR for k={k}:")
        print(pv.to_string())

if __name__ == "__main__":
    main()
