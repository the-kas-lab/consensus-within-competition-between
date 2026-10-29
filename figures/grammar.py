"""The visual grammar shared by every figure, drawn one way each time.

Three building blocks:
* seed_dots + mean_ci: one dot per seed in the team's colour and marker, with the mean and its 95% CI in
  black beside the dots (never over them);
* diff_row: a paired difference, per-seed differences as faint grey dots, the mean and 95% CI in black,
  on an axis with a grey zero line (zero_line);
* direction labels at the two ends of a difference axis (ends).
Dots of discrete values are spread as a deterministic beeswarm, so equal values never hide each other.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from figures import style

DOT = 2.4            # pt, marker size of one seed
DOT_DIFF = 2.0       # pt, a seed's difference (context)
MEAN = 4.2           # pt, the mean marker
CI_LW = style.LW_EMPH


def _swarm(values, step, width):
    """Offsets that spread equal values side by side, centred, within +-width/2."""
    values = np.asarray(values, float)
    out = np.zeros(len(values))
    for v, n in Counter(np.round(values, 9)).items():
        idx = np.where(np.isclose(values, v))[0]
        s = min(step, width / max(n - 1, 1))
        out[idx] = (np.arange(n) - (n - 1) / 2) * s
    return out


def beeswarm(ax, x, values, dot_pt=DOT, gap=0.25, vertical=True, half_width=0.3):
    """Category-axis offsets that keep dots of continuous values from overlapping, computed in printed
    points from the axes' current limits (set limits first). Each dot takes the free slot nearest the
    centre line; returns positions along the category axis."""
    vals = np.asarray(values, float)
    pts = ax.transData.transform(np.column_stack([np.full(len(vals), x), vals]) if vertical
                                 else np.column_stack([vals, np.full(len(vals), x)]))
    scale = ax.figure.dpi / 72.0
    d = (dot_pt + gap) * scale
    along = pts[:, 1] if vertical else pts[:, 0]
    placed = []
    off = np.zeros(len(vals))
    for i in np.argsort(along):
        for k in range(200):
            cand = (k + 1) // 2 * (1 if k % 2 else -1) * d * 0.5 if k else 0.0
            if all((along[i] - a) ** 2 + (cand - o) ** 2 >= d * d for a, o in placed):
                break
        off[i] = cand
        placed.append((along[i], cand))
    # cap the spread at half_width (data units of the category axis): many equal values then pack densely
    ref = ax.transData.transform([(x, 0), (x + half_width, 0)] if vertical else [(0, x), (0, x + half_width)])
    cap = abs(ref[1, 0] - ref[0, 0]) if vertical else abs(ref[1, 1] - ref[0, 1])
    if np.abs(off).max(initial=0) > cap:
        off *= cap / np.abs(off).max()
    if vertical:
        inv = ax.transData.inverted().transform(np.column_stack([pts[:, 0] + off, pts[:, 1]]))[:, 0]
    else:
        inv = ax.transData.inverted().transform(np.column_stack([pts[:, 0], pts[:, 1] + off]))[:, 1]
    return inv


def swarm_dots(ax, x, values, arm, open_marker=False, vertical=True, half_width=0.3):
    """One dot per seed, beeswarmed around category position x (call after setting the limits)."""
    st = style.ARM[arm]
    pos = beeswarm(ax, x, values, vertical=vertical, half_width=half_width)
    kw = dict(marker=st["marker"], ls="none", markersize=DOT, zorder=3, clip_on=False)
    if open_marker:
        kw.update(markerfacecolor=style.WHITE, markeredgecolor=st["color"], markeredgewidth=style.LW_RULE)
    else:
        kw.update(color=st["color"], markeredgewidth=0)
    if vertical:
        ax.plot(pos, values, **kw)
    else:
        ax.plot(values, pos, **kw)


def seed_dots(ax, x, values, arm, open_marker=False, step=0.055, width=0.34, vertical=True):
    """One dot per seed at category position x (vertical) or y (horizontal)."""
    st = style.ARM[arm]
    off = _swarm(values, step, width)
    kw = dict(marker=st["marker"], ls="none", markersize=DOT, zorder=3, clip_on=False)
    if open_marker:
        kw.update(markerfacecolor=style.WHITE, markeredgecolor=st["color"], markeredgewidth=style.LW_RULE)
    else:
        kw.update(color=st["color"], markeredgewidth=0)
    if vertical:
        ax.plot(x + off, values, **kw)
    else:
        ax.plot(values, x + off, **kw)


def mean_ci(ax, x, mean, lo, hi, vertical=True):
    """The mean (black dot) and its 95% CI (black bar)."""
    if vertical:
        ax.plot([x, x], [lo, hi], color=style.TEXT, lw=CI_LW, solid_capstyle="butt", zorder=4)
        ax.plot([x], [mean], marker="o", color=style.TEXT, markersize=MEAN, markeredgewidth=0, zorder=5)
    else:
        ax.plot([lo, hi], [x, x], color=style.TEXT, lw=CI_LW, solid_capstyle="butt", zorder=4)
        ax.plot([mean], [x], marker="o", color=style.TEXT, markersize=MEAN, markeredgewidth=0, zorder=5)


BUBBLE = 1.15        # pt of marker diameter per sqrt(seed)


def bubble_size(n):
    """Marker size (pt diameter) of a circle that stands for n seeds: area proportional to n."""
    return BUBBLE * np.sqrt(n) * 2.0


def diff_row(ax, y, per_seed, mean, lo, hi, lift=0.27):
    """A paired difference on a horizontal axis. Above the row, one grey circle per distinct per-seed
    value, its area proportional to the number of seeds with that value; on the row, the mean and 95% CI."""
    vals = np.round(np.asarray(per_seed, float), 6)
    for v, n in sorted(Counter(vals).items()):
        ax.plot([v], [y - lift], marker="o", ls="none", color=style.ZERO, markersize=bubble_size(n),
                markeredgewidth=0, zorder=2, clip_on=False)
    mean_ci(ax, y, mean, lo, hi, vertical=False)


def bubble_key(ax, x, y, counts=(1, 5, 10), label="Seeds"):
    """A small key for the circle areas, in axes-fraction x and data y."""
    tr = ax.get_yaxis_transform()
    ax.text(x, y, label, transform=tr, ha="left", va="center")
    xx = x + 0.16
    for n in counts:
        ax.plot([xx], [y], marker="o", ls="none", color=style.ZERO, markersize=bubble_size(n),
                markeredgewidth=0, transform=tr, clip_on=False)
        ax.text(xx + 0.035, y, str(n), transform=tr, ha="left", va="center")
        xx += 0.13


def mean_ci_key(ax, x, y, label="Mean and 95% CI", length=0.07):
    """Key entry for the black mean-and-CI glyph, in axes-fraction x and data y; returns the x after it."""
    tr = ax.get_yaxis_transform()
    ax.plot([x, x + length], [y, y], color=style.TEXT, lw=CI_LW, solid_capstyle="butt", transform=tr,
            clip_on=False)
    ax.plot([x + length / 2], [y], marker="o", color=style.TEXT, markersize=MEAN, markeredgewidth=0,
            transform=tr, clip_on=False)
    ax.text(x + length + 0.02, y, label, transform=tr, ha="left", va="center")


def ci_handle():
    """Key entry (a legend handle) for a black vertical 95% CI bar."""
    from matplotlib.lines import Line2D
    return Line2D([], [], color=style.TEXT, marker="|", ls="none", markersize=7, markeredgewidth=CI_LW)


def zero_line(ax, vertical=True):
    if vertical:
        ax.axvline(0, color=style.ZERO, lw=style.LW_RULE, zorder=1)
    else:
        ax.axhline(0, color=style.ZERO, lw=style.LW_RULE, zorder=1)


def ends(ax, left, right, y):
    """Direction words at the two ends of a horizontal difference axis, in axes-fraction x and data y."""
    ax.text(0.0, y, "← " + left, transform=ax.get_yaxis_transform(), ha="left", va="bottom")
    ax.text(1.0, y, right + " →", transform=ax.get_yaxis_transform(), ha="right", va="bottom")


def team_handles(arms, near_frontier=()):
    """Key entries (marker + team name) in the fixed team order."""
    from matplotlib.lines import Line2D
    hs, ls = [], []
    for arm in arms:
        st = style.ARM[arm]
        hs.append(Line2D([], [], marker=st["marker"], ls="none", color=st["color"], markersize=MEAN,
                         markeredgewidth=0))
        ls.append(st["label"])
    return hs, ls


def fmt_est(v):
    """A difference as the text prints it: whole percentage points with a sign (v is a fraction)."""
    from figures import nums
    s = nums.pct(abs(v))
    return s if s == "0" else (("+" if v > 0 else style.MINUS) + s)


def fmt_ci(lo, hi):
    """A 95% CI as the text prints it: signs only when the interval includes zero (fractions in)."""
    if lo > 0 or hi < 0:
        return f"{fmt_est(lo).lstrip('+')} to {fmt_est(hi).lstrip('+')}"
    return f"{fmt_est(lo)} to {fmt_est(hi)}"


def forest(ax, blocks, xlim, xticks, left, right, value_header="Mean (95% CI)", ends_every_block=True):
    """A difference forest in the house grammar. blocks: [(title, [row, ...])], each row a dict with
    label, d (per-seed differences, percentage points), m, lo, hi (percentage points) and text.
    Direction words sit beside the zero line on every block's title row; returns the last y used."""
    y = 0.0
    yt, yl = [], []
    tr = ax.get_yaxis_transform()
    for b, blk in enumerate(blocks):
        title, rows = blk[0], blk[1]
        lft, rgt = (blk[2], blk[3]) if len(blk) == 4 else (left, right)      # a block may name its own sides
        ax.text(-0.02, y, title, transform=tr, ha="right", va="center", fontweight="bold")
        if ends_every_block or b == 0 or len(blk) == 4:
            ax.text(-0.03 * (xlim[1] - xlim[0]), y, f"← {lft}", ha="right", va="center")
            ax.text(0.03 * (xlim[1] - xlim[0]), y, f"{rgt} →", ha="left", va="center")
        y += 1
        for rw in rows:
            diff_row(ax, y, rw["d"], rw["m"], rw["lo"], rw["hi"])
            yt.append(y)
            yl.append(rw["label"])
            ax.text(1.03, y, rw["text"], transform=tr, ha="left", va="center")
            y += 1
    ax.set_yticks(yt, yl)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    top = -1.0
    ax.set_ylim(y - 0.4, top)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks, [("+" if v > 0 else style.MINUS if v < 0 else "") + f"{abs(v):g}" for v in xticks])
    ax.plot([0, 0], [-0.45, y - 0.4], color=style.ZERO, lw=style.LW_RULE, zorder=1)
    ax.text(1.03, top + 0.1, value_header, transform=tr, ha="left", va="center", fontweight="bold")
    return y
