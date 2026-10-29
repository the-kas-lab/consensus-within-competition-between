"""Named, independent random-number streams, one per purpose.

Why this exists
---------------
The whole statistical design of the decentralized/semantic-sabotage work rests
on *paired* episodes: requires matched scenario seeds across
conditions, and every clean/attack comparison is a pair of episodes that must
differ only in the attack. That is only meaningful if an episode is a pure
function of (seed, config).

Before this module there was exactly one stream -- the process-global
``np.random`` -- seeded once in ``Simulation.__init__`` and drawn from by two
unrelated consumers at two different times:

* agent/contact spawn positions, drawn during construction;
* ``ReconController`` patrol waypoints, drawn during the run.

That arrangement has two consequences, both bad:

1. **Adding any new randomness reshuffles everything downstream of it.** A
   packet-loss draw, an attack schedule, a randomized red spawn bearing -- each
   would consume values from the same stream the patrol waypoints come from, so
   a "clean vs. attack" pair with the same seed would diverge for reasons that
   have nothing to do with the attack.'s counter-based *link* RNG
   fixes links only; it does not fix this.
2. **Construction order leaks between episodes.** Building two Simulations
   before running either means the second construction re-seeds the global
   stream, so the first episode's patrol waypoints depend on whether another
   episode happened to be constructed first. See
   ``tests/test_reproducibility.py::test_construction_order_does_not_matter``.

Legacy mode
-----------
``legacy_global=True`` (the default) keeps both existing consumers on the
process-global ``np.random``, so episodes stay **byte-identical** to
pre-refactor runs --'s Milestone 1 exit criterion is "existing
results reproduce through the new interfaces", not "results improve". Newly
added purposes (``comm``, ``attack``, ``scenario``) are *always* independent
Generators regardless of the flag, so consequence (1) above is fixed
immediately and unconditionally.

``legacy_global=False`` puts every purpose on its own ``np.random.Generator``,
which additionally fixes consequence (2). It renumbers episodes -- the same seed
yields a different (equally valid) trajectory -- so it is opt-in until the
project is ready to re-baseline. See
"""

import numpy as np

# Purposes that existed before this module, and therefore have legacy behavior
# to preserve. Anything not listed here is new and is always an independent
# Generator.
_LEGACY_PURPOSES = ("spawn", "patrol", "inert")

PURPOSES = (
    "spawn",      # agent + contact spawn states
    "patrol",     # ReconController waypoints
    "inert",      # bird / civilian random-walk accelerations
    "comm",       # link loss / delivery draws
    "attack",     # adversary scheduling and content
    "scenario",   # randomized scenario families
)


class _GlobalStreamProxy:
    """Forwards every call to the process-global ``np.random``.

    Exists so call sites can be written against the stream API unconditionally
    (``self.rng.spawn.uniform(...)``) while legacy mode still draws from the
    exact same global sequence, in the exact same order, as before the refactor.
    """

    __slots__ = ()

    def __getattr__(self, name):
        return getattr(np.random, name)

    def __repr__(self):
        return "<_GlobalStreamProxy -> np.random>"


class RngStreams:
    """Per-purpose RNG streams derived from one episode seed.

    Streams are independent by construction: ``SeedSequence.spawn`` produces
    statistically independent child sequences, so consuming from one never
    advances another. Access a stream as an attribute::

        rng = RngStreams(seed=42)
        rng.spawn.uniform(0.0, 100.0, size=2)
        rng.comm.random()

    Note ``Generator`` has ``.random()`` but not ``.rand()``; call sites use the
    ``Generator`` API, which the legacy proxy also satisfies since
    ``np.random.random`` is an alias of ``np.random.random_sample`` (the same
    function ``np.random.rand`` draws from).
    """

    def __init__(self, seed, legacy_global=True):
        self.seed = int(seed)
        self.legacy_global = bool(legacy_global)

        root = np.random.SeedSequence(self.seed)
        children = root.spawn(len(PURPOSES))
        self._streams = {}
        for purpose, child in zip(PURPOSES, children):
            if self.legacy_global and purpose in _LEGACY_PURPOSES:
                self._streams[purpose] = _GlobalStreamProxy()
            else:
                self._streams[purpose] = np.random.default_rng(child)

    def __getattr__(self, name):
        try:
            return self.__dict__["_streams"][name]
        except KeyError:
            raise AttributeError(
                f"No RNG stream named {name!r}. Known purposes: {', '.join(PURPOSES)}. "
                "Add new purposes to rng.PURPOSES rather than drawing from a "
                "shared stream -- see this module's docstring."
            ) from None

    def __repr__(self):
        mode = "legacy_global" if self.legacy_global else "isolated"
        return f"RngStreams(seed={self.seed}, mode={mode})"
