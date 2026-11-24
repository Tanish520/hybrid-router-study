"""Experiment 1 report: NMT-vs-LLM structural defect tables, figures and examples.

Reads catalog_summary.csv and failure_labels.csv.gz (written by run_audit.py).
Writes, in results/01_structural_audit/:
  system_defect_rates.csv / .md           per-system rates, v1 vs v2, all languages and Indic-6
  placeholder_accelerator_subsets.csv     failure rates on placeholder / accelerator sources
  examples.md                             side-by-side examples per check
  figures/defect_rates_v1_v2.png          v1 vs v2 defect rate per system
  figures/defect_heatmap_v2.png           v2 defect rate, system x language
"""
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router.data import INDIC, LANGS, ROOT, catalogs, load_source, read_lines  # noqa: E402
from router.paths import AUDIT, ensure  # noqa: E402

RESULTS, FIGURES = AUDIT, AUDIT / "figures"
CHECKS = ["empty_output", "script_contamination", "broken_placeholder", "broken_accelerator",
          "encoding_corruption", "conversational_chatter", "copy_fallback"]
SYSTEM_ORDER = (["nmt:google", "nmt:microsoft", "nmt:nllb", "nmt:madlad", "nmt:indictrans2"]
                + [f"llm:{s}:{t}" for t in ("baseline", "repaired")
                   for s in ("gpt4o", "llama_8b", "gemma_9b", "mistral_nemo", "qwen_7b")])
LABEL = {"nmt:google": "Google (NMT)", "nmt:microsoft": "Microsoft (NMT)", "nmt:nllb": "NLLB-600M (NMT)",
         "nmt:madlad": "MADLAD-3B (NMT)", "nmt:indictrans2": "IndicTrans2* (NMT)"}
for t in ("baseline", "repaired"):
    for s, name in [("gpt4o", "GPT-4o-mini"), ("llama_8b", "Llama-3.1-8B"), ("gemma_9b", "Gemma-2-9B"),
                    ("mistral_nemo", "Mistral-Nemo"), ("qwen_7b", "Qwen2.5-7B")]:
        LABEL[f"llm:{s}:{t}"] = f"{name} ({'LLM' if t == 'baseline' else 'LLM+IGSC'})"

# reference palette (dataviz skill): categorical slots 1-2, sequential blue ramp
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def key(row):
    return f"nmt:{row['system']}" if row["paradigm"] == "nmt" else f"llm:{row['system']}:{row['tier']}"


def system_table(df):
    df = df.assign(key=df.apply(key, axis=1))
    rows = []
    for k in SYSTEM_ORDER:
        r = {"system": LABEL[k]}
        for v in ("v1", "v2"):
            sub = df[(df.key == k) & (df.version == v)]
            r[f"langs"] = len(sub)
            r[f"{v}_defect_pct"] = round(100 * sub.defective.sum() / sub.lines.sum(), 2)
            ind = sub[sub.lang.isin(INDIC)]
            r[f"{v}_indic6_pct"] = round(100 * ind.defective.sum() / ind.lines.sum(), 2)
            if v == "v2":
                for c in CHECKS:
                    r[f"v2_{c}"] = int(sub[c].sum())
        rows.append(r)
    return pd.DataFrame(rows)


def artifact_subsets(rows_df, failures):
    ph = set(rows_df.index[rows_df.has_placeholder == 1])
    acc = set(rows_df.index[rows_df.has_accelerator_v2 == 1])
    n_langs = defaultdict(set)
    for c in catalogs():
        n_langs[f"nmt:{c.system}" if c.paradigm == "nmt" else f"llm:{c.system}:{c.tier}"].add(c.lang)
    out = []
    for k in SYSTEM_ORDER:
        f = failures.get(k, {})
        ph_fail = sum(1 for (lang, i), iss in f.items() if i in ph and "broken_placeholder" in iss)
        acc_fail = sum(1 for (lang, i), iss in f.items() if i in acc and "broken_accelerator" in iss)
        n = len(n_langs[k])
        out.append({"system": LABEL[k], "langs": n,
                    "placeholder_sources": len(ph), "placeholder_fail_pct": round(100 * ph_fail / (len(ph) * n), 1),
                    "accelerator_sources": len(acc), "accelerator_fail_pct": round(100 * acc_fail / (len(acc) * n), 1)})
    return pd.DataFrame(out)


def load_failures(version="v2"):
    out = defaultdict(dict)
    with gzip.open(RESULTS / "failure_labels.csv.gz", "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["version"] == version:
                out[key(r)][(r["lang"], int(r["line_idx"]))] = r["issues"].split(";")
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def fig_rates(table):
    t = table.iloc[::-1]
    y = range(len(t))
    fig, ax = plt.subplots(figsize=(8, 6.4))
    h = 0.38
    ax.barh([i + h / 2 + 0.01 for i in y], t.v1_defect_pct, height=h, color=BLUE, label="v1: original verifier as published")
    ax.barh([i - h / 2 - 0.01 for i in y], t.v2_defect_pct, height=h, color=ORANGE, label="v2: accelerator + empty-output fixes")
    for i, (a, b) in enumerate(zip(t.v1_defect_pct, t.v2_defect_pct)):
        ax.text(b + 0.2, i - h / 2, f"{b:.2f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(t.system, color=INK)
    ax.set_xlabel("Structurally defective translations (% of segments)", color=INK2)
    ax.axhline(9.5, color=GRID, lw=1)
    ax.axhline(4.5, color=GRID, lw=1)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Structural defect rate per system, 10,228 KDE4 strings x 12 languages\n"
                 "*IndicTrans2 covers the 6 Indic languages only", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "defect_rates_v1_v2.png", dpi=200, facecolor="#fcfcfb")


def fig_heatmap(df):
    d = df[df.version == "v2"].assign(key=df.apply(key, axis=1))
    grid = d.pivot_table(index="key", columns="lang", values="defect_rate_pct").reindex(SYSTEM_ORDER)[list(LANGS)]
    fig, ax = plt.subplots(figsize=(9, 6.4))
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    cmap.set_bad("#f1f0ec")
    im = ax.imshow(grid.values, cmap=cmap, vmin=0, vmax=min(40, float(grid.max().max())), aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="#fcfcfb" if v > 18 else INK)
    ax.set_xticks(range(len(LANGS)))
    ax.set_xticklabels(LANGS, color=INK)
    ax.set_yticks(range(len(SYSTEM_ORDER)))
    ax.set_yticklabels([LABEL[k] for k in SYSTEM_ORDER], color=INK, fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("Defective translations (%), verifier v2", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Structural defect rate by system and language (verifier v2); grey = not covered",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "defect_heatmap_v2.png", dpi=200, facecolor="#fcfcfb")


def examples(failures, src):
    """For each check, a few rows where NLLB fails it, shown next to Google, Llama and repaired Llama."""
    hyp = {}
    for c in catalogs():
        if c.lang in ("hi", "de") and (c.system in ("nllb", "google") or c.system == "llama_8b"):
            hyp[(f"nmt:{c.system}" if c.paradigm == "nmt" else f"llm:{c.system}:{c.tier}", c.lang)] = read_lines(c.path)
    cols = ["nmt:google", "nmt:nllb", "llm:llama_8b:baseline", "llm:llama_8b:repaired"]
    lines = ["# Experiment 1: examples (verifier v2)", "",
             "Rows where NLLB-600M fails a check, shown next to Google, Llama-3.1-8B and IGSC-repaired Llama. "
             "Cells marked ✗ fail the same check.", ""]
    for check in ("broken_placeholder", "broken_accelerator", "copy_fallback"):
        lines += [f"## {check}", "", "| lang | line | source | " + " | ".join(LABEL[c] for c in cols) + " |",
                  "|---|---|---|" + "---|" * len(cols)]
        shown = 0
        for (lang, i), iss in sorted(failures["nmt:nllb"].items(), key=lambda x: (x[0][0] != "hi", x[0][1])):
            if check not in iss or lang not in ("hi", "de"):
                continue
            cells = []
            for c in cols:
                mark = " ✗" if check in failures.get(c, {}).get((lang, i), []) else ""
                cells.append(f"`{hyp[(c, lang)][i]}`{mark}".replace("|", "\\|"))
            lines.append(f"| {lang} | {i} | `{src[i]}` | " + " | ".join(cells) + " |")
            shown += 1
            if shown == 6:
                break
        lines.append("")
    (RESULTS / "examples.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure(FIGURES)
    df = pd.read_csv(RESULTS / "catalog_summary.csv")
    table = system_table(df)
    table.to_csv(RESULTS / "system_defect_rates.csv", index=False)
    cols = ["system", "langs", "v1_defect_pct", "v2_defect_pct", "v1_indic6_pct", "v2_indic6_pct"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in table.iterrows()]
    (RESULTS / "system_defect_rates.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    rows_df = pd.read_csv(ROOT / "data" / "manifest" / "rows.csv", keep_default_na=False).set_index("line_idx")
    failures = load_failures("v2")
    subsets = artifact_subsets(rows_df, failures)
    subsets.to_csv(RESULTS / "placeholder_accelerator_subsets.csv", index=False)
    fig_rates(table)
    fig_heatmap(df)
    examples(failures, load_source())
    print("\n".join(md))
    print()
    print(subsets.to_string(index=False))


if __name__ == "__main__":
    main()
