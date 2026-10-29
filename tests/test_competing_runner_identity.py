"""run_competing_teams: execution identity, resume refusal, repetition ids and
the development-only MH+LI condition (no model calls: rule backend only)."""

import json

import pytest

from scripts import run_competing_teams as rct


def test_repetition_is_part_of_the_episode_id_only_when_given():
    e = {"arm": "cbaa", "reds": 2, "family": "multi_bearing", "condition": "REF",
         "attacked": False, "seed": 3981}
    assert rct.episode_id(e) == "cbaa|r2|multi_bearing|REF|s3981"
    assert rct.episode_id(dict(e, rep=2)) == "cbaa|r2|multi_bearing|REF|s3981|rep2"


def test_identity_changes_with_the_study_version_and_a_mismatched_resume_is_refused(tmp_path):
    a = rct.execution_identity("rule", None, "s1")
    b = rct.execution_identity("rule", None, "s2")
    assert a["identity_hash"] != b["identity_hash"]
    assert a["safety_version"] == "safety.2"
    man = tmp_path / "m.jsonl"
    rct.main(["manifest", "--seeds", "3981", "--arms", "cbaa", "--intruders", "2",
              "--families", "multi_bearing", "--conditions", "REF", "--no-attack",
              "--out", str(man)])
    out = tmp_path / "r.jsonl"
    out.write_text(json.dumps({"episode_id": "x", "identity_hash": "other"}) + "\n")
    with pytest.raises(SystemExit, match="without an identity file"):
        rct.main(["run", "--manifest", str(man), "--out", str(out), "--backend", "rule"])
    (tmp_path / "r.jsonl.identity.json").write_text(json.dumps({"identity_hash": "other"}))
    with pytest.raises(SystemExit, match="refusing to resume"):
        rct.main(["run", "--manifest", str(man), "--out", str(out), "--backend", "rule"])


def test_identity_records_the_effective_design_configuration_only_when_used():
    from scripts import experiment_policy as ep
    from scripts.inference_profile import INTERFACE_VERSIONS

    plain = rct.execution_identity("rule", None, "s1")
    assert "design_interfaces" not in plain and "design_policy" not in plain
    assert rct.execution_identity("rule", None, "s1", designs=[None, None]) == plain
    d = rct.execution_identity("rule", None, "s1", designs=["v2.3mb", None])
    assert d["design_policy"] == {"v2.3mb": ep.DESIGN_POLICY}
    assert d["identity_hash"] != plain["identity_hash"]

    class P:
        def as_dict(self):
            return {"p": 1}

        def profile_hash(self):
            return "h"

    withp = rct.execution_identity("profile", P(), "s1", designs=["v2.3mb"])
    assert withp["design_interfaces"] == {"v2.3mb": INTERFACE_VERSIONS}


def test_a_pair_is_complete_only_when_both_halves_end_validly():
    man = [{"arm": "cbaa", "reds": 2, "family": "multi_bearing", "condition": "REF",
            "attacked": att, "seed": 1} for att in (True, False)]
    for e in man:
        e["episode_id"] = rct.episode_id(e)
    ida, idr = man[0]["episode_id"], man[1]["episode_id"]
    ok = {"outcome": {"touchdown": True, "red_outcomes": ["active", "active"]}, "steps_run": 50}
    horizon = {"outcome": {"touchdown": False, "red_outcomes": ["active", "neutralized"]},
               "steps_run": 400, "horizon_steps": 400}
    cut = {"outcome": {"touchdown": False, "red_outcomes": ["active", "active"]}, "steps_run": 120,
           "horizon_steps": 400}
    failed = dict(horizon, solver_failed=True)

    # every record copies its manifest entry's fields (episode_record)
    fa = {k: man[0].get(k) for k in rct.RECORD_MANIFEST_KEYS}
    fr = {k: man[1].get(k) for k in rct.RECORD_MANIFEST_KEYS}

    def status(a, r):
        res = [dict(a, episode_id=ida, **fa), dict(r, episode_id=idr, **fr)]
        return rct.pair_table(man, res)[0]["status"]

    assert status(ok, horizon) == "complete"
    assert status(cut, horizon) == "invalid_attacked"
    assert status(ok, failed) == "invalid_reference"
    assert status(cut, failed) == "invalid_both"
    assert rct.pair_table(man, [dict(ok, episode_id=ida, **fa)])[0]["status"] == "missing_reference"


def test_a_pair_needs_one_matching_record_per_half_under_one_identity():
    man = [{"arm": "cbaa", "reds": 2, "family": "multi_bearing", "condition": "REF",
            "attacked": att, "seed": 1, "design": "v2.2"} for att in (True, False)]
    for e in man:
        e["episode_id"] = rct.episode_id(e)
    ok = {"outcome": {"touchdown": True, "red_outcomes": ["active", "active"]}, "steps_run": 50}

    def rec(e, **kw):
        r = dict(ok, episode_id=e["episode_id"], identity_hash="I",
                 **{k: e.get(k) for k in rct.RECORD_MANIFEST_KEYS})
        r.update(kw)
        return r

    a, r = rec(man[0]), rec(man[1])
    assert rct.pair_table(man, [a, r], identity_hash="I")[0]["status"] == "complete"
    row = rct.pair_table(man, [a, r, rec(man[1])])[0]
    assert (row["status"], row["reference_problem"]) == ("invalid_reference", "duplicate")
    row = rct.pair_table(man, [a, rec(man[1], identity_hash="J")])[0]
    assert (row["status"], row["attacked_problem"]) == ("invalid_both", "identity_mismatch")
    row = rct.pair_table(man, [a, r], identity_hash="J")[0]
    assert row["status"] == "invalid_both"
    row = rct.pair_table(man, [a, rec(man[1], design="v2.1")])[0]
    assert (row["status"], row["reference_problem"]) == ("invalid_reference", "manifest_mismatch")
