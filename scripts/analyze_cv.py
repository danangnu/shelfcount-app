"""Analyse cross-validation outputs (paper Sections VI.F-K and VIII).

  python scripts/analyze_cv.py --cv experiments/cv [--detection]

Writes experiments/cv/results.json and per_photo.csv, and prints the paper's tables.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

GRID = [round(0.25 + 0.05 * i, 2) for i in range(10)]  # 0.25 ... 0.70
METHODS = {"finetuned": "Fine-tuned YOLO11n", "yoloe_text": "YOLOE-11s, text prompts", "yoloe_free": "YOLOE-11s, prompt-free"}
RNG = np.random.default_rng(0)
B = 10_000


def count(p, tau, region=None, margin=0.03):
    if region is None:
        return sum(1 for s, _, _ in p["boxes"] if s >= tau)
    x1, y1, x2, y2 = region
    mx, my = margin * (x2 - x1), margin * (y2 - y1)
    return sum(1 for s, cx, cy in p["boxes"] if s >= tau and x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my)


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def metrics(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    e = pred - true
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "MAPE": (np.abs(e) / true).mean() * 100,
            "W10": (np.abs(e) / true <= 0.10).mean() * 100, "Bias": e.mean()}


def boot_ci(pred, true, groups=None):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    n = len(pred)
    stats = {k: [] for k in ("MAE", "RMSE", "MAPE", "W10", "Bias")}
    if groups is None:
        idx_sets = (RNG.integers(0, n, n) for _ in range(B))
    else:
        groups = np.asarray(groups)
        ug = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in ug}
        idx_sets = (np.concatenate([members[g] for g in RNG.choice(ug, len(ug))]) for _ in range(B))
    for idx in idx_sets:
        m = metrics(pred[idx], true[idx])
        for k in stats:
            stats[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in stats.items()}


def wider(a, b):
    return {k: (min(a[k][0], b[k][0]), max(a[k][1], b[k][1])) for k in a}


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True)
    ap.add_argument("--src", default=None, help="dataset folder with annotations.csv, for the duplicate-label sensitivity")
    ap.add_argument("--posthoc-exclude", nargs="*", default=[], help="planograms left out in a post-hoc sensitivity analysis")
    ap.add_argument("--detection", action="store_true", help="also run detection metrics on each test fold (needs ultralytics)")
    a = ap.parse_args()
    cv = Path(a.cv).resolve()
    folds = json.load(open(cv / "folds.json"))["folds"]
    regions = json.load(open(cv / "labelled_regions.json"))
    zs = {m: json.load(open(cv / f"preds_{m}.json")) for m in ("yoloe_text", "yoloe_free") if (cv / f"preds_{m}.json").exists()}
    fold_preds = {k: json.load(open(cv / f"preds_{k}.json")) for k in folds if (cv / f"preds_{k}.json").exists()}

    dedup = {}
    if a.src:
        import collections
        bb = collections.defaultdict(list)
        for r in csv.reader(open(Path(a.src) / "annotations.csv")):
            bb[r[0]].append(tuple(map(float, r[1:5])))
        for n, boxes in bb.items():
            kept = []
            for b in boxes:
                if all(iou(b, k) <= 0.9 for k in kept):
                    kept.append(b)
            dedup[n] = len(kept)

    rows, chosen, curves = [], {}, {}
    for k, fp in fold_preds.items():
        for m in ["finetuned"] + list(zs):
            val = fp["inner_val"] if m == "finetuned" else {n: zs[m][n] for n in fp["inner_val"]}
            test = fp["test"] if m == "finetuned" else {n: zs[m][n] for n in fp["test"]}
            mae_whole = {t: np.mean([abs(count(p, t) - p["true"]) for p in val.values()]) for t in GRID}
            mae_reg = {t: np.mean([abs(count(p, t, regions[n]) - p["true"]) for n, p in val.items()]) for t in GRID}
            tw, tr = min(GRID, key=lambda t: (mae_whole[t], t)), min(GRID, key=lambda t: (mae_reg[t], t))
            chosen[(m, k)] = {"tau_whole": tw, "tau_region": tr}
            for n, p in test.items():
                rows.append({"method": m, "fold": k, "image": n, "planogram": n.split("_")[1], "true": p["true"],
                             "tau": tw, "pred_whole": count(p, tw), "pred_region": count(p, tw, regions[n]),
                             "pred_region_tau_region": count(p, tr, regions[n]),
                             "pred_region_m0": count(p, tw, regions[n], 0.0), "pred_region_m6": count(p, tw, regions[n], 0.06),
                             "true_dedup": dedup.get(n, p["true"]), "ms": p["ms"]})
            if m == "finetuned":
                for t in GRID:
                    curves.setdefault(t, {"whole": [], "region": []})
                    curves[t]["whole"] += [count(p, t) - p["true"] for p in test.values()]
                    curves[t]["region"] += [count(p, t, regions[n]) - p["true"] for n, p in test.items()]

    with open(cv / "per_photo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    res = {"chosen_tau": {f"{m}|{k}": v for (m, k), v in chosen.items()}, "methods": {}}
    print("\nThresholds chosen on inner layouts:", {f"{m}|{k}": v["tau_whole"] for (m, k), v in chosen.items()})
    for m in METHODS:
        r = [x for x in rows if x["method"] == m]
        if not r:
            continue
        true = [x["true"] for x in r]
        out = {}
        for scope, key in (("whole", "pred_whole"), ("region", "pred_region"), ("region_tau_region", "pred_region_tau_region"),
                           ("region_m0", "pred_region_m0"), ("region_m6", "pred_region_m6")):
            pred = [x[key] for x in r]
            point = metrics(pred, true)
            ci = wider(boot_ci(pred, true), boot_ci(pred, true, [x["planogram"] for x in r])) if scope in ("whole", "region") else None
            per_fold = {k: metrics([x[key] for x in r if x["fold"] == k], [x["true"] for x in r if x["fold"] == k]) for k in fold_preds}
            out[scope] = {"point": point, "ci95": ci, "per_fold": per_fold,
                          "fold_mean_sd": {s: (float(np.mean([v[s] for v in per_fold.values()])),
                                               float(np.std([v[s] for v in per_fold.values()], ddof=1)))
                                           for s in point}}
        by_p = {}
        for pg in sorted({x["planogram"] for x in r}):
            rr = [x for x in r if x["planogram"] == pg]
            by_p[pg] = {"n": len(rr), **metrics([x["pred_whole"] for x in rr], [x["true"] for x in rr])}
        out["by_planogram"] = by_p
        if a.posthoc_exclude:
            rr = [x for x in r if x["planogram"] not in a.posthoc_exclude]
            for scope, key in (("whole", "pred_whole"), ("region", "pred_region")):
                pr, tr = [x[key] for x in rr], [x["true"] for x in rr]
                out[f"posthoc_{scope}"] = {"n": len(rr), "excluded": a.posthoc_exclude, "point": metrics(pr, tr),
                                           "ci95": wider(boot_ci(pr, tr), boot_ci(pr, tr, [x["planogram"] for x in rr]))}
        out["whole_dedup"] = {"point": metrics([x["pred_whole"] for x in r], [x["true_dedup"] for x in r])}
        out["region_dedup"] = {"point": metrics([x["pred_region"] for x in r], [x["true_dedup"] for x in r])}
        e = np.array([x["pred_whole"] - x["true"] for x in r], float)
        avg = (np.array([x["pred_whole"] for x in r]) + np.array(true)) / 2
        out["bland_altman"] = {"mean_diff": float(e.mean()), "loa": (float(e.mean() - 1.96 * e.std(ddof=1)),
                                                                      float(e.mean() + 1.96 * e.std(ddof=1))),
                               "corr_err_vs_size": float(np.corrcoef(np.abs(e), avg)[0, 1])}
        out["n"] = len(r)
        res["methods"][m] = out

    # Paired comparisons: fine-tuned vs each zero-shot baseline, |e| whole photo
    ft = {x["image"]: abs(x["pred_whole"] - x["true"]) for x in rows if x["method"] == "finetuned"}
    comps, pv = [], []
    for m in zs:
        other = {x["image"]: abs(x["pred_whole"] - x["true"]) for x in rows if x["method"] == m}
        common = sorted(set(ft) & set(other))
        d = np.array([other[n] - ft[n] for n in common], float)  # positive = fine-tuned better
        stat = wilcoxon(d)
        boots = [np.median(d[RNG.integers(0, len(d), len(d))]) for _ in range(B)]
        comps.append({"vs": m, "n": len(d), "median_diff": float(np.median(d)),
                      "ci95": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
                      "mae_diff": float(d.mean()), "p": float(stat.pvalue)})
        pv.append(stat.pvalue)
    for c, p in zip(comps, holm(np.array(pv)) if pv else []):
        c["p_holm"] = float(p)
    res["comparisons"] = comps

    res["threshold_curve"] = {str(t): {"mae_whole": float(np.mean(np.abs(v["whole"]))), "mae_region": float(np.mean(np.abs(v["region"]))),
                                       "bias_whole": float(np.mean(v["whole"])), "bias_region": float(np.mean(v["region"]))}
                              for t, v in curves.items()}
    taus = [v["tau_whole"] for (m, k), v in chosen.items() if m == "finetuned"]
    res["deployed_tau"] = float(np.median(taus)) if taus else None

    # Error analysis: photos off by more than 10% (fine-tuned, whole photo)
    errs = []
    for x in rows:
        if x["method"] != "finetuned":
            continue
        e_w, e_r = x["pred_whole"] - x["true"], x["pred_region"] - x["true"]
        if abs(e_w) / x["true"] > 0.10:
            outside = x["pred_whole"] - x["pred_region"]
            cause = ("unlabelled products outside labelled region" if abs(e_r) / x["true"] <= 0.10 and outside > 0
                     else "overcount inside labelled region" if e_r > 0 else "undercount inside labelled region")
            errs.append({**x, "err_whole": e_w, "err_region": e_r, "outside": outside, "auto_cause": cause})
    res["error_analysis"] = {"n_over_10pct": len(errs), "by_cause": {c: sum(1 for e in errs if e["auto_cause"] == c)
                                                                     for c in sorted({e["auto_cause"] for e in errs})},
                             "photos": errs}
    if (cv / "timing.json").exists():
        t = json.load(open(cv / "timing.json"))
        res["speed"] = {"model_ms": t["model_ms_median"],
                        "end_to_end_ms_median": float(np.median([r["end_to_end_ms"] for r in t["photos"]])),
                        "end_to_end_ms_p95": float(np.percentile([r["end_to_end_ms"] for r in t["photos"]], 95)),
                        "engine_ms_median": float(np.median([r["engine_ms"] for r in t["photos"]]))}

    if a.detection:
        from ultralytics import YOLO
        res["detection"] = {}
        for k in fold_preds:
            imgs = sorted((cv / "data" / "images").glob("*.jpg"))
            test = [str(p) for p in imgs if p.stem.split("_")[1] in folds[k]["test"]]
            (cv / k / "test.txt").write_text("\n".join(test) + "\n")
            (cv / k / "test.yaml").write_text(f"train: {cv / k / 'test.txt'}\nval: {cv / k / 'test.txt'}\nnames:\n  0: product\n")
            mt = YOLO(str(cv / k / "train" / "weights" / "best.pt")).val(data=str(cv / k / "test.yaml"), imgsz=640,
                                                                        batch=8, device="cpu", max_det=1500, plots=False,
                                                                        verbose=False, project=str(cv / k), name="test_val",
                                                                        exist_ok=True)
            res["detection"][k] = {"precision": float(mt.box.mp), "recall": float(mt.box.mr), "mAP50": float(mt.box.map50),
                                   "mAP50_95": float(mt.box.map)}

    json.dump(res, open(cv / "results.json", "w"), indent=1, default=float)
    for m, o in res["methods"].items():
        for scope in ("whole", "region"):
            p, c = o[scope]["point"], o[scope]["ci95"]
            print(f"{METHODS[m]:<28} {scope:<7} n={o['n']} " + " ".join(f"{s}={p[s]:.2f} [{c[s][0]:.2f},{c[s][1]:.2f}]" for s in p))
    print("comparisons", comps)
    print("deployed tau", res["deployed_tau"], "| errors >10%:", res["error_analysis"]["by_cause"])
    print("speed", res.get("speed"), "| detection", res.get("detection"))


if __name__ == "__main__":
    main()
