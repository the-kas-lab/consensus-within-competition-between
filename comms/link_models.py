"""Whether a link exists, and whether a packet survives it.

 Two things live here and they are deliberately separate:

* **Connectivity** -- is (i,j) an edge at all? The disk model (Sec 10.1) and the
  smooth logistic model (Sec 10.2). The disk is the zero-width limiting case of
  the logistic, which is the point: no headline finding should depend on an
  artificial hard cutoff sitting at exactly one formation distance.
* **Loss** -- given an edge, does this packet arrive? iid Bernoulli, and
  Gilbert-Elliott burst loss (Sec 10.3).

## Counter-based randomness is the non-negotiable

Every draw is keyed by ``(seed, sender, receiver, comm_tick)`` and computed by
hashing that tuple -- **never** by consuming from a sequential stream. Plan
Sec 10.1 requires this and the reason is the whole statistical design:

A sequential stream couples every later draw to every earlier one. Add one
message, change one planner, reorder two agents, and every subsequent packet
outcome reshuffles -- so a clean/attack pair that should differ *only* in the
attack diverges for unrelated reasons. With counter-based draws, link (3,5) at
tick 40 gets the same value regardless of what any other link did, what order
agents were iterated in, or whether an unrelated code path drew at all. Paired
conditions stay paired by construction rather than by luck.

This is also why ``RngStreams.comm`` exists but is **not** used for link draws:
it is a Generator, and Generators are sequential. The stream is available for
anything genuinely per-episode (schedules, one-off draws); link outcomes use the
hash.

Symmetry: a link is one physical thing, so the key is canonicalized on the
unordered pair. Without that, i->j and j->i would fade independently, which is
not what a symmetric disk graph means.
"""

import hashlib
import struct

import numpy as np

# Two-state Gilbert-Elliott channel labels. GOOD/BAD is the standard naming and
# is kept rather than something project-specific so the model is recognizable.
GOOD = "good"
BAD = "bad"


def _uniform(seed, sender, receiver, comm_tick, salt=""):
    """A reproducible uniform in [0, 1) for one link at one tick.

    Deterministic in its arguments alone -- no hidden state, no ordering
    dependence. BLAKE2b at 8 bytes is used rather than Python's ``hash`` because
    ``hash`` is randomized per process by PYTHONHASHSEED, which would make
    episodes irreproducible across runs.

    The pair is canonicalized so the link is symmetric: draws for (i,j) and
    (j,i) at the same tick are identical.
    """
    lo, hi = (int(sender), int(receiver))
    if lo > hi:
        lo, hi = hi, lo
    key = f"{int(seed)}|{lo}|{hi}|{int(comm_tick)}|{salt}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    # 2**53 keeps the result exactly representable as a float64.
    return (struct.unpack("<Q", digest)[0] >> 11) / float(1 << 53)


class DiskLinkModel:
    """Symmetric disk graph: an edge exists iff distance <= R_comm.

    ``R_comm = inf`` is a complete graph and is the default everywhere, so the
    communication layer is a no-op until a range is configured -- the
    default-off rule from the roadmap.
    """

    def __init__(self, r_comm=float("inf")):
        self.r_comm = float(r_comm)

    def connected(self, distance, seed=None, sender=None, receiver=None, comm_tick=None):
        return float(distance) <= self.r_comm

    def describe(self):
        return {"model": "disk", "r_comm": self.r_comm}


class LogisticLinkModel:
    """Smooth distance-dependent success probability.

        p(d) = 1 / (1 + exp((d - r_50) / sigma))

    The disk model is the ``sigma -> 0`` limit. Used for the robustness pass at
    M12: if a headline finding only survives a hard cutoff, that finding is
    about the cutoff.
    """

    def __init__(self, r_50, sigma):
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}; use DiskLinkModel for the hard cutoff")
        self.r_50 = float(r_50)
        self.sigma = float(sigma)

    def probability(self, distance):
        # Clipped because exp overflows to inf well before the probability
        # stops being 0 or 1 to double precision.
        z = np.clip((float(distance) - self.r_50) / self.sigma, -700.0, 700.0)
        return float(1.0 / (1.0 + np.exp(z)))

    def connected(self, distance, seed, sender, receiver, comm_tick):
        return _uniform(seed, sender, receiver, comm_tick, salt="link") < self.probability(distance)

    def describe(self):
        return {"model": "logistic", "r_50": self.r_50, "sigma": self.sigma}


class NoLoss:
    """The default. Keeps M4 a pure connectivity layer until loss is switched
    on at M12, per's "add only after the primary model works"."""

    def delivers(self, seed, sender, receiver, comm_tick):
        return True

    def describe(self):
        return {"model": "none"}


class IidLossModel:
    """Independent Bernoulli packet loss."""

    def __init__(self, loss_probability):
        if not 0.0 <= loss_probability <= 1.0:
            raise ValueError(f"loss_probability must be in [0,1], got {loss_probability}")
        self.loss_probability = float(loss_probability)

    def delivers(self, seed, sender, receiver, comm_tick):
        return _uniform(seed, sender, receiver, comm_tick, salt="loss") >= self.loss_probability

    def describe(self):
        return {"model": "iid", "loss_probability": self.loss_probability}


class GilbertElliottLossModel:
    """Two-state burst loss.

    The channel sits in GOOD or BAD and transitions with ``p_gb`` / ``p_bg``;
    each state has its own loss probability. Bursts matter because iid loss at
    the same mean rate is much easier to ride out -- a belief survives one
    dropped packet, not forty consecutive ones.

    **State is derived, not stored.** The chain is replayed from tick 0 for the
    requested link, so the result depends only on ``(seed, link, tick)`` and the
    counter-based guarantee survives. Replay is memoized per link; the cost is
    linear in tick and irrelevant at these horizons, and it buys immunity to
    call ordering -- querying link A then B gives the same answers as B then A,
    which a stored-state implementation would not.
    """

    def __init__(self, p_gb=0.05, p_bg=0.3, loss_good=0.0, loss_bad=0.8):
        self.p_gb = float(p_gb)
        self.p_bg = float(p_bg)
        self.loss_good = float(loss_good)
        self.loss_bad = float(loss_bad)
        self._cache = {}

    def _state_at(self, seed, sender, receiver, comm_tick):
        lo, hi = sorted((int(sender), int(receiver)))
        key = (int(seed), lo, hi)
        history = self._cache.setdefault(key, [GOOD])
        for tick in range(len(history), int(comm_tick) + 1):
            previous = history[tick - 1]
            draw = _uniform(seed, lo, hi, tick, salt="ge")
            if previous == GOOD:
                history.append(BAD if draw < self.p_gb else GOOD)
            else:
                history.append(GOOD if draw < self.p_bg else BAD)
        return history[int(comm_tick)]

    def state_at(self, seed, sender, receiver, comm_tick):
        return self._state_at(seed, sender, receiver, comm_tick)

    def delivers(self, seed, sender, receiver, comm_tick):
        state = self._state_at(seed, sender, receiver, comm_tick)
        loss = self.loss_good if state == GOOD else self.loss_bad
        return _uniform(seed, sender, receiver, comm_tick, salt="geloss") >= loss

    def describe(self):
        return {
            "model": "gilbert_elliott",
            "p_gb": self.p_gb,
            "p_bg": self.p_bg,
            "loss_good": self.loss_good,
            "loss_bad": self.loss_bad,
        }


LOSS_MODELS = ("iid", "gilbert_elliott")


def loss_model_from_spec(spec):
    """A NEW loss model from a plain cfg spec, or None for no loss.

    ``{"model": "iid", "p": 0.3}`` or ``{"model": "gilbert_elliott", "p_gb":
    ..., "p_bg": ..., "loss_good": ..., "loss_bad": ...}``. A spec rather than
    an object, so a runner's cfg stays plain data and **every episode gets its
    own instance**. That matters for Gilbert-Elliott, which memoizes each
    link's state history: the history is a pure function of (seed, link), so a
    shared instance would not change any draw, but it would carry one
    episode's memo into the next. Constructing fresh excludes that by
    construction.

    Both models draw once per LINK per communication tick (``_uniform`` keys
    the canonical pair and the tick), so every envelope on one link in one
    round shares the draw, in both directions and whatever its message type.
    """
    if spec is None:
        return None
    spec = dict(spec)
    kind = spec.pop("model", None)
    if kind == "iid":
        model = IidLossModel(spec.pop("p"))
    elif kind == "gilbert_elliott":
        names = ("p_gb", "p_bg", "loss_good", "loss_bad")
        model = GilbertElliottLossModel(**{k: spec.pop(k) for k in names if k in spec})
    else:
        raise ValueError(f"unknown loss model {kind!r}; expected one of {LOSS_MODELS}")
    if spec:
        raise ValueError(f"unknown loss-model parameters {sorted(spec)}")
    return model
