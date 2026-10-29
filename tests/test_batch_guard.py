"""scripts.batch_guard: the safeguards a paid multi-batch run relies on."""
import json

from scripts import batch_guard as bg
from scripts import run_competing_teams as rct

PRICE_IN, PRICE_OUT = 2.5e-7, 1.5e-6


def _manifest(condition="REF", seeds=(1,)):
    out = []
    for seed in seeds:
        for arm in ("cbaa", "commander"):
            e = {"arm": arm, "reds": 2, "family": "multi_bearing", "condition": condition,
                 "attacked": False, "seed": seed, "rep": "r2", "design": "v2.2"}
            e["episode_id"] = rct.episode_id(e)
            out.append(e)
    return out


def _record(e, ident="I", cost=0.1, **kw):
    rec = {k: e.get(k) for k in rct.RECORD_MANIFEST_KEYS}
    rec.update(episode_id=e["episode_id"], identity_hash=ident, cost_usd_written=cost,
               outcome={"touchdown": True, "red_outcomes": ["active", "active"]}, steps_run=50,
               horizon_steps=400)
    rec.update(kw)
    return rec


def _cache_record(tmp_path, name, content="x" * 100, prompt=50, completion=10, max_tokens=1000,
                  cost=None):
    body = {"messages": [{"role": "user", "content": content}], "response_format": {"t": 1},
            "max_tokens": max_tokens}
    nbytes = len(content.encode()) + len(json.dumps({"t": 1}).encode())
    worst = (nbytes + 256) * PRICE_IN + max_tokens * PRICE_OUT
    rec = {"provenance": {"request_body": body},
           "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                     "cost_usd": worst / 2 if cost is None else cost}}
    (tmp_path / name).write_text(json.dumps(rec))
    return worst


def test_audit_accepts_calls_within_the_reservation_and_flags_each_violation(tmp_path):
    _cache_record(tmp_path, "a.json")
    assert bg.audit(str(tmp_path), PRICE_IN, PRICE_OUT) == {"calls_checked": 1, "violations": []}
    _cache_record(tmp_path, "b.json", prompt=10_000)
    _cache_record(tmp_path, "c.json", completion=2000)
    w = _cache_record(tmp_path, "d.json")
    _cache_record(tmp_path, "d.json", cost=2 * w)
    (tmp_path / "e.json").write_text("{not json")
    a = bg.audit(str(tmp_path), PRICE_IN, PRICE_OUT)
    probs = {v["file"].rsplit("/", 1)[1]: v["problems"][0].split()[0] for v in a["violations"]}
    assert probs == {"b.json": "prompt_tokens", "c.json": "completion", "d.json": "cost",
                     "e.json": "unreadable"}


def test_reconcile_stops_on_unaccounted_spend_and_never_releases_budget_on_lag():
    base = {"account_usage": 100.0, "ledger_committed": 10.0}
    ok = bg.reconcile(base, 101.0, 11.0, tolerance=0.02)
    assert ok["status"] == "ok" and ok["committed"] == 11.0
    lag = bg.reconcile(base, 100.5, 11.0, tolerance=0.02)  # account behind the ledger
    assert lag["status"] == "account_lags" and lag["discrepancy"] == -0.5
    assert lag["committed"] == 11.0  # the ledger's liability is kept, not lowered to 10.5
    extra = bg.reconcile(base, 102.0, 11.0, tolerance=0.02)  # account ahead of the ledger
    assert extra["status"] == "unaccounted_spend" and extra["committed"] == 12.0


def test_projection_never_falls_below_the_pre_run_estimate():
    cheap = bg.projection(10.0, 4, [0.5], pre_run_batch_estimate=1.88)
    assert cheap["per_batch"] == 1.88 and abs(cheap["projected_committed"] - 17.52) < 1e-9
    dear = bg.projection(10.0, 4, [0.5, 2.5], pre_run_batch_estimate=1.88)
    assert dear["per_batch"] == 2.5 and dear["projected_committed"] == 20.0
    assert bg.projection(10.0, 0, [], 1.88)["projected_committed"] == 10.0


def test_batch_costs_are_reconstructed_from_saved_records_for_complete_batches_only():
    ref, mh = _manifest("REF"), _manifest("MH")
    recs = [_record(e, cost=0.25) for e in ref] + [_record(mh[0], cost=0.4)]
    costs = bg.batch_costs({"REF": ref, "MH": mh}, recs)
    assert costs == {"REF": 0.5}  # MH is incomplete: not an observed batch cost yet


def test_progress_separates_attempted_from_validly_completed():
    man = _manifest(seeds=(1, 2))
    cut = {"outcome": {"touchdown": False, "red_outcomes": ["active", "active"]},
           "steps_run": 120}
    recs = [_record(man[0]),                                  # valid
            _record(man[1], **cut),                           # attempted, not terminal
            _record(man[2]), _record(man[2]),                 # duplicate
            ]
    pr = bg.progress(man, recs, "I")
    assert pr["valid"] == [man[0]["episode_id"]]
    assert pr["missing"] == [man[3]["episode_id"]]
    assert pr["unresolved"] == {man[1]["episode_id"]: "not_terminal",
                                man[2]["episode_id"]: "duplicate"}
    wrong = bg.progress(man[:1], [_record(man[0], ident="OTHER")], "I")
    assert wrong["unresolved"] == {man[0]["episode_id"]: "identity_mismatch"}
    moved = bg.progress(man[:1], [_record(dict(man[0], seed=99), episode_id=man[0]["episode_id"])],
                        "I")
    assert moved["unresolved"] == {man[0]["episode_id"]: "manifest_mismatch"}


def test_completion_needs_a_clean_audit_and_a_validated_full_manifest():
    man = _manifest(seeds=(1,))
    recs = [_record(e) for e in man]
    clean = {"calls_checked": 10, "violations": []}
    assert bg.completion(man, recs, "I", clean)["done"]
    dirty = {"calls_checked": 10, "violations": [{"file": "x", "problems": ["cost"]}]}
    assert not bg.completion(man, recs, "I", dirty)["done"]
    assert not bg.completion(man, recs, "I", {"calls_checked": 0, "violations": []})["done"]
    assert not bg.completion(man, recs[:1], "I", clean)["done"]
    other = _manifest("MH")[0]
    extra = bg.completion(man, recs + [_record(other)], "I", clean)
    assert not extra["done"] and extra["reasons"] == ["1 records outside the manifest"]
