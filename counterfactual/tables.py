"""The lookup tables behind q, next to the observed first counterfactual look.

Rows per model: ``q_M`` is M's choice-conditioned belief (`counterfactual.beliefs`),
``q_M uncond`` its unconditioned choice probability renormalised over the
unchosen exits (the Figure Five / v1 q, commit `09e6fea`,
`counterfactual.compare.unconditioned_q`). ``observed`` is the share of first
counterfactual looks.

1. **Aggregate by relation to the chosen exit** (all mazes pooled): the
   sibling (same side, the other vertical arm), the horizontal neighbour
   (same up/down level, other side) and the diagonal exit. Per label x
   outcome group, over the labelled trials with a counterfactual look, so
   model and observed rows describe the same trials.
2. **Per maze**: one table per maze. Under each chosen exit, the H and S
   rows (conditioned only) and the observed first-look shares (all
   labelled, H-labelled, S-labelled), so they can be read against each
   other.

Output: ``counterfactual/out/<Monkey>/belief_tables.md``.

    uv run python -m counterfactual.tables [--monkey ...]
"""

from __future__ import annotations

import argparse

import numpy as np

from counterfactual import beliefs, extract
from counterfactual.compare import unconditioned_q
from counterfactual.config import EXIT_NAMES, H_LABEL, MODELS, MONKEYS, S_LABEL, out_dir
from counterfactual.task import relation, unchosen

RELATIONS = ("vertical", "horizontal", "diagonal")
RELATION_NAMES = ("sibling", "horizontal", "diagonal")


def model_q(tr, q):
    """``{row name: (n, 4)}``: conditioned and unconditioned q per model."""
    out = {}
    for m in MODELS:
        out[f"q_{m}"] = np.asarray(q[f"q_{m}"], dtype=float)
        out[f"q_{m} uncond"] = np.array([unconditioned_q(p, int(c)) for p, c in zip(q[f"choice_{m}"], tr["chosen"])])
    return out


def first_looks(tr):
    """First counterfactual exit per trial (0 when none or unlabelled)."""
    first = np.zeros(len(tr["session"]), dtype=int)
    for r in range(len(first)):
        items = extract.counterfactual_items(tr, r)
        if items and np.isfinite(tr["label"][r]):
            first[r] = items[0]
    return first


def _row(name, v, chosen=None):
    return f"| {name} | " + " | ".join("–" if e == chosen else f"{x:.2f}" for e, x in zip((1, 2, 3, 4), v)) + " |"


def maze_table(tr, q, first, maze):
    """One table per maze: for each chosen exit, the H and S rows of the
    model tables, then the observed share of first counterfactual looks
    (all labelled, H-labelled, S-labelled; correct and error trials)."""
    L = ["| chosen | | " + " | ".join(EXIT_NAMES) + " |", "|---|---|---|---|---|---|"]
    fmt = lambda v, c: ["–" if e == c else f"{v[e - 1]:.2f}" for e in (1, 2, 3, 4)]
    for c in (1, 2, 3, 4):
        k = (tr["maze"] == maze) & (tr["chosen"] == c)
        if not k.any():
            continue
        name = EXIT_NAMES[c - 1]
        for m in MODELS:
            L.append(f"| {name} | q_{m} | " + " | ".join(fmt(q[f"q_{m}"][k][0], c)) + " |")
            name = ""
        for title, lab in (("**observed**", None), ("observed, H-labelled", H_LABEL), ("observed, S-labelled", S_LABEL)):
            j = k & (first > 0) & (np.isfinite(tr["label"]) if lab is None else tr["label"] == lab)
            cells = fmt(np.bincount(first[j], minlength=5)[1:] / j.sum(), c) if j.any() else \
                ["–" if e == c else "" for e in (1, 2, 3, 4)]
            L.append(f"| | {title} (n {j.sum()}) | " + " | ".join(cells) + " |")
    return L + [""]


def relation_aggregate(tr, Q, first):
    """Per label x outcome: q and observed share on sibling / horizontal / diagonal."""
    rel = np.array([[RELATIONS.index(relation(int(c), e)) for e in unchosen(int(c))] for c in tr["chosen"]])
    slots = np.array([[e - 1 for e in unchosen(int(c))] for c in tr["chosen"]])
    head = "| | " + " | ".join(RELATION_NAMES) + " |\n|---|---|---|---|"
    error = tr["chosen"] != tr["correct"]
    L = []
    groups = [("All labelled trials", np.isfinite(tr["label"]))]
    for lab, lname in ((H_LABEL, "H-labelled"), (S_LABEL, "S-labelled")):
        for oname, o in (("correct", ~error), ("error", error)):
            groups.append((f"{lname}, {oname}", (tr["label"] == lab) & o))
    for name, k in groups:
        k = k & (first > 0)
        if not k.any():
            continue
        L += [f"**{name}** (n {k.sum()})", "", head]
        idx = np.flatnonzero(k)
        for qname, v in Q.items():
            by_rel = np.zeros(3)
            for r in idx:
                by_rel[rel[r]] += v[r, slots[r]]
            L.append(_row(qname, by_rel / len(idx)))
        obs = np.zeros(3)
        for r in idx:
            obs[RELATIONS.index(relation(int(tr["chosen"][r]), int(first[r])))] += 1
        L += [_row("observed", obs / len(idx)), ""]
    return L


def write(monkey):
    tr, q = extract.load(monkey), beliefs.load(monkey)
    Q, first = model_q(tr, q), first_looks(tr)
    head = "| | " + " | ".join(EXIT_NAMES) + " |\n|---|---|---|---|---|"
    L = [f"# {monkey}: belief tables", "",
         "q_M: model M's belief over the unchosen exits given that M's rule chose the chosen exit. "
         "q_M uncond: M's probability of choosing each exit, renormalised over the unchosen exits (Figure Five). "
         "Observed: share of first counterfactual looks. See counterfactual/README.md and counterfactual/tables.py.", "",
         "## 1. All mazes, by relation to the chosen exit", "",
         "Mean probability on each unchosen exit by its relation to the chosen one (rows sum to 1), "
         "over the labelled trials with a counterfactual look.", ""]
    L += relation_aggregate(tr, Q, first)
    L += ["## 2. Per maze", "",
          "Each model stores one row per chosen exit: its probability that the monkey looks next at each of the "
          "other three exits (the row sums to 1). The row is built from the path that ends at the chosen exit "
          "(draws kept when the model's rule picks it) and applied to every trial that chose it, error trials "
          "included. Under each chosen exit: the H and S rows, then the observed share of first counterfactual "
          "looks (correct and error trials).", ""]
    for maze in range(1, 7):
        h = tr["h"][tr["maze"] == maze][0]
        L += [f"### Maze {maze}: h1..h6 = {', '.join(f'{v:g}' for v in h)}"
              + ("  (h1 = h4: the first interval cannot tell the sides apart)" if h[0] == h[3] else ""), ""]
        L += maze_table(tr, q, first, maze)
    path = out_dir(monkey) / "belief_tables.md"
    path.write_text("\n".join(L))
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description="Belief tables vs observed first looks.")
    p.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=MONKEYS)
    for m in p.parse_args().monkey:
        write(m)


if __name__ == "__main__":
    main()
