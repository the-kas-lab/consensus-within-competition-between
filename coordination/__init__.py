"""Decentralized task allocation: task generation, deterministic bids, CBAA.

The auction. See

Nothing in this package reads ground truth. Every function takes an agent's
frozen ``LocalWorldView`` plus static scenario geometry, and nothing else --
the same discipline the prompt builder is held to, and for the same reason.
"""
