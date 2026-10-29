"""Export the example episodes as replay bundles for the replay viewer (from the cache; no model is called).

One bundle per episode, from which the Supplementary episode figures, panel a of Fig. 7, the Supplementary
Videos and the README GIFs are drawn by the viewer (viewer/js/scene.js). The episodes are chosen by fixed rules
(each takes the first qualifying seed in order from 4001; illustrative, not evidence):

  E1_*          main study, ideal network, four intruders, seed 4003: the consensus, commander and independent teams
  c_commander   main study, limited range, four intruders, commander team: the first seed on which the team
                breached and at least one order that would have changed its addressee's task never reached it
                (analysis.commander_replays, missed orders)
  d_commander, e_commander_nearfrontier   airspace challenge, the first seed on which the low-cost commander team
                lost a defender to a civilian drone; the near-frontier commander team on the same seed
  f_nowarning, g_warning   disguised intruders, near-frontier commander team: the first seed on which it breached
                without the forewarning and did not breach with it (needs --bird-drone-dir)
  fig7a_cbaa    airspace challenge, consensus team at low authority, seed 4001

Each episode is re-simulated from the cache with the API keys removed, so a cache miss stops it. It must equal its
stored record on breach, breach step, breaching intruder, every intruder's outcome and kills, or nothing is
written; the exporter's own check (scripts/export_replay.check_bundle) then compares the bundle with the live
episode.

    python3 -m analysis.episode_bundles --data DATA [--out OUT] [--bird-drone-dir DIR] [key ...]

Writes OUT/bundles/<key>.replay.js and OUT/bundles/export_log.json.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

from analysis import common, replay

CHECK = replay.CHECK
# key: paper team, condition, setting (used for titles only; never read back as data)
PAPER = {
    "E1_commander": ("commander team", "ideal network", "main study"),
    "E1_cbaa": ("consensus team", "ideal network", "main study"),
    "E1_independent": ("independent team", "ideal network", "main study"),
    "c_commander": ("commander team", "limited range", "main study"),
    "d_commander": ("low-cost commander team", "ideal network", "airspace challenge"),
    "e_commander_nearfrontier": ("near-frontier commander team", "ideal network", "airspace challenge"),
    "f_nowarning": ("near-frontier commander team", "ideal network", "disguised intruders, without the forewarning"),
    "g_warning": ("near-frontier commander team", "ideal network", "disguised intruders, with the forewarning"),
    "fig7a_cbaa": ("consensus team at low authority", "ideal network", "airspace challenge"),
}


def select(data, out_dir, with_disguise):
    """{key: (run, episode_id)}."""
    out = {}
    for arm in ("commander", "cbaa", "independent"):
        out[f"E1_{arm}"] = (common.MAIN[0], f"{arm}|r4|multi_bearing|REF|s4003|repmatrix|design-v2.3mb")
    mo = json.load(open(os.path.join(out_dir, "commander_missed_orders.json")))
    assert mo["n_reproduced"] == mo["n"] == 300
    c = sorted((x for x in mo["episodes"] if x["condition"] == "MH" and not x["attacked"] and x["reds"] == 4
                and x["outcome"]["touchdown"] and x["missed_changed"] > 0), key=lambda x: x["seed"])[0]
    out["c_commander"] = (common.MAIN[0] if c["seed"] <= 4010 else common.MAIN[1], c["episode_id"])
    run = common.AIRSPACE[0]
    air = common.records(data, [run])
    cmd = sorted((r for r in air if r["rep"] == "img-CMD"), key=lambda r: r["seed"])
    d = next(r for r in cmd if any(ev["kind"] == "civilian" for ev in r["outcome"]["decoys"]["events"]))
    e = next(r for r in air if r["rep"] == "img-SOLCMD" and r["seed"] == d["seed"])
    out["d_commander"] = (run, d["episode_id"])
    out["e_commander_nearfrontier"] = (run, e["episode_id"])
    out["fig7a_cbaa"] = (run, next(r for r in air if r["seed"] == 4001 and r["rep"] == "img-B")["episode_id"])
    if with_disguise:
        run = common.DISGUISE[0]
        camo = common.records(data, [run])
        u = {r["seed"]: r for r in camo if r["rep"] == "camo-U-SOLCMD"}
        w = {r["seed"]: r for r in camo if r["rep"] == "camo-W-SOLCMD"}
        s = min(k for k in u if u[k]["outcome"]["touchdown"] and not w[k]["outcome"]["touchdown"])
        out["f_nowarning"] = (run, u[s]["episode_id"])
        out["g_warning"] = (run, w[s]["episode_id"])
    return out


def job(item):
    key, run, eid, data, out_dir, bird_drone_dir = item
    for k in [k for k in os.environ if k.startswith("OPENROUTER_API_KEY")]:
        os.environ.pop(k)
    from scripts import export_replay as ex
    from scripts.episode_builder import _Quiet

    rec = next(r for r in common.records(data, [run]) if r["episode_id"] == eid)
    entry = next(x for x in replay.manifest(run, bird_drone_dir) if x["episode_id"] == eid)
    sim = replay.build(entry, common.run_dir(data, run), common.identity(data, run))
    capture = ex.attach_capture(sim)
    with _Quiet():
        sim.run()
    o = sim.episode_outcome()
    mism = [k for k in CHECK if o[k] != rec["outcome"][k]]
    assert not mism, (key, eid, mism)
    team, condition, setting = PAPER[key]
    source = {"preset": "si-example", "arm": rec["arm"], "seed": int(rec["seed"]), "family": rec["family"],
              "belief_mode": sim.belief_mode, "cfg": {}, "label": f"{team}, {condition}, seed {rec['seed']}",
              "episode_id": eid, "paper": {"team": team, "condition": condition, "setting": setting},
              "record_outcome": {k: rec["outcome"][k] for k in CHECK}}
    ep = ex.episode_bundle(sim, capture, source)
    ex.check_bundle(ep, sim, capture)
    bundles = os.path.join(out_dir, "bundles")
    os.makedirs(bundles, exist_ok=True)
    path = os.path.join(bundles, f"{key}.replay.js")
    ex.write_bundle(ex.bundle_document([ep], argv=[key]), path)
    fates = {}
    for row in (ep["command"] or {}).get("orders", []):
        fates[row["fate"]] = fates.get(row["fate"], 0) + 1
    return {"key": key, "episode_id": eid, "bytes": os.path.getsize(path), "n_steps": ep["dims"]["n_steps"],
            "outcome": {k: o[k] for k in CHECK}, "record_equal": True, "bundle_check": "passed",
            "order_fates": fates, "false_beliefs": len(ep.get("beliefs_held", {}))}


def main(data, out_dir, keys, bird_drone_dir=None, workers=4):
    chosen = select(data, out_dir, bird_drone_dir is not None)
    items = sorted((k, *v) for k, v in chosen.items() if not keys or k in keys)
    assert not keys or len(items) == len(keys), keys
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        res = list(pool.map(job, [(k, run, eid, data, out_dir, bird_drone_dir) for k, run, eid in items]))
    log_path = os.path.join(out_dir, "bundles", "export_log.json")
    if keys and os.path.exists(log_path):
        old = json.load(open(log_path))["episodes"]
        res = sorted([r for r in old if r["key"] not in keys] + res, key=lambda r: r["key"])
    log = {"exported": len(res), "outcome_mismatches": 0, "format_version": 2,
           "note": "every episode equals its stored record on " + ", ".join(CHECK) +
                   "; every bundle passed export_replay.check_bundle against the live re-run",
           "episodes": res}
    common.dump(log, log_path)
    return res


if __name__ == "__main__":
    ap = common.parser(__doc__.split("\n")[0])
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--bird-drone-dir", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for r in main(a.data, common.out_dir(a), a.keys, a.bird_drone_dir, a.workers):
        print(r["key"], r["n_steps"], r["outcome"]["touchdown"], r["order_fates"], r["bytes"])
