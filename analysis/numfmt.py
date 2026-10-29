"""Shared helpers of the macro generators: reading analysis outputs by leaf, and the rounding rules."""
import json
import os
from decimal import ROUND_HALF_UP, Decimal

AN = None
J = {}


def set_dir(path):
    global AN
    AN = path
    J.clear()


def load(rel):
    if rel not in J:
        J[rel] = json.load(open(os.path.join(AN, rel)))
    return J[rel]


def leaf(rel, path):
    x = load(rel)
    for p in path:
        x = x[p]
    return x


def pct(v):
    """Whole percent, rounded half up."""
    return str(Decimal(str(v * 100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def signed_pp(v):
    s = pct(abs(v))
    return ("+" if v > 0 else "\\textminus" if v < 0 else "") + s


def signed_pp1(v):
    s = str(Decimal(str(abs(v) * 100)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    return ("+" if v > 0 else "\\textminus" if v < 0 else "") + s


def holm(pvals):
    """Holm step-down adjustment; returns adjusted p in the input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, run = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = run
    return adj


def pval(v):
    return "<0.001" if v < 0.001 else (f"{v:.3f}" if v < 0.01 else f"{v:.2f}")
