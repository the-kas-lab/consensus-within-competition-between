"""The airspace challenge (birds and civilian drones among the intruders) and the disguised intruders, per team.

Airspace (5 seeds, ideal network, four intruders; five teams: B = consensus team, CMD = commander team,
W = consensus team at high authority, WSOL and SOLCMD = the same two with the near-frontier model). From the
records and the cache:

  * per team: breaches, intruders destroyed, defenders lost to birds / civilian drones (by kind) and to
    intruders, and the matched episodes of the main study without birds and civilian drones (same seeds);
  * per model call (answered, before validation), birds and civilian drones against real intruders:
      B          priority label (LOW / MEDIUM / HIGH; MEDIUM if omitted)
      W, WSOL    contact priority (number 0..2; 1 if omitted)
      CMD/SOLCMD whether the answer assigned the contact a pursue task

Disguise (5 seeds, ideal network, four intruders; 5 teams x 2 conditions: U = the airspace prompt, W = the same
prompt plus the forewarning that intruders may be disguised; every real intruder shown as a photograph of a
bird-shaped drone). The airspace episodes are the paired reference (same seeds, contacts, spawns and prompts as U;
intruders shown as attack drones), recomputed with the same code so every row uses one definition. Per team and
condition: breaches, intruders destroyed, defenders lost to birds / civilian drones, episode end; per model call,
for every contact by its true kind (drone = a real intruder, here shown as a bird drone), whether it was set aside
(B: priority LOW; W/WSOL: priority 0, also < 1; CMD/SOLCMD: no pursue assignment in the answer). Speed cue (real
intruders only): the speed |v| reported in that call's prompt. Birds and civilian drones never exceed the
intruders' per-axis v_max (2.0), so |v| > 2.05 (margin for the 2-decimal print) is a speed none of them can show;
the set-aside share of real intruders is split at that threshold.

    python3 -m analysis.airspace --data DATA [--out OUT]       ->  OUT/airspace.json, OUT/disguise.json
"""
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

from analysis import common

SEEDS = (4001, 4002, 4003, 4004, 4005)
FAST = 2.05
_VEL = (re.compile(r"contact_id=(\d+)\s.*?believed_velocity=\[(-?[\d.]+),\s*(-?[\d.]+)\]"),
        re.compile(r"contact (\d+) predicted pos=\[[^\]]*\] vel=\[(-?[\d.]+),\s*(-?[\d.]+)\]"))
ARMS = ("B", "CMD", "W", "SOLCMD", "WSOL")


def label(r):
    return r["rep"].split("img-", 1)[1]


def kind_of(r, cid):
    k = ((r["outcome"].get("decoys") or {}).get("kinds") or {})
    return k.get(str(cid)) or k.get(int(cid)) or "drone"


def outcome_row(r):
    o = r["outcome"]
    d = o.get("decoys") or {}
    by_kind = Counter(e["kind"] for e in d.get("events", []) if e["event"] == "destroyed")
    return {"breach": bool(o["touchdown"]), "reds_destroyed": int(o["num_neutralized"]),
            "blues_lost_to_contacts": int(d.get("n_destroyed", 0)),
            "blues_lost_to_contacts_by_kind": dict(by_kind),
            "blues_lost_total": int(o["n_blues_disabled"]),
            "end": ("breach" if o["touchdown"] else
                    "all reds resolved" if all(x != "active" for x in o["red_outcomes"]) else "horizon")}


# -- matched episodes of the main study without birds and civilian drones (ideal network, four intruders)

def _ref(recs, arm, keep=lambda r: True):
    out = {}
    for r in recs:
        if (r["arm"] == arm and r["reds"] == 4 and r["condition"] == "REF" and not r["attacked"]
                and r["seed"] in SEEDS and keep(r)):
            out[r["seed"]] = r
    return out


def references(data):
    solw = {r["seed"]: r for r in common.valid_records(data, common.HIGH_AUTHORITY_NEAR_FRONTIER)
            if r["seed"] in SEEDS}
    main = common.records(data, common.MAIN)
    return {"B": _ref(main, "cbaa"), "CMD": _ref(main, "commander"),
            "W": _ref(common.records(data, common.HIGH_AUTHORITY_LOW_COST), "cbaa_sw2"),
            "WSOL": solw,
            "SOLCMD": _ref(common.records(data, common.NEAR_FRONTIER_COMMANDER), "commander")}


# -- per-call treatment of birds and civilian drones against real intruders

def contact_calls(cache_dir, by_slug_record):
    """Yield (label, kind, value) per contact the model was prompted about."""
    for f in glob.glob(os.path.join(cache_dir, "*", "*.json")):
        slug_dir = os.path.basename(os.path.dirname(f))
        rec = json.load(open(f))
        prov = rec.get("provenance") or {}
        vc = prov.get("validation_context") or {}
        if rec.get("kind") == "transport_error" or not rec.get("content"):
            continue
        slug = slug_dir.replace("__near_frontier", "")
        r = by_slug_record.get(slug)
        if r is None:
            continue
        lab = label(r)
        try:
            ans = json.loads(rec["content"])
        except ValueError:
            continue
        kind = vc.get("kind")
        if kind == "defender" and lab == "B":
            pr = {int(x["contact_id"]): str(x["priority"]).upper() for x in ans.get("threat_priorities") or []
                  if isinstance(x, dict) and "contact_id" in x}
            for c in vc.get("believed_contacts") or []:
                yield lab, kind_of(r, c), pr.get(int(c), "MEDIUM")
        elif kind == "strategy_defender" and lab in ("W", "WSOL"):
            pr = {}
            for x in ans.get("contact_priorities") or []:
                try:
                    pr[int(x["contact_id"])] = float(x["priority"])
                except (KeyError, TypeError, ValueError):
                    pass
            for c in vc.get("believed_contacts") or []:
                yield lab, kind_of(r, c), pr.get(int(c), 1.0)
        elif kind == "commander" and lab in ("CMD", "SOLCMD"):
            cat = vc.get("catalogue") or []
            contacts = sorted({int(t.split(":")[1]) for t in cat if t.startswith("NEUTRALIZE:")})
            tasks = set((ans.get("task_by_agent") or {}).values())
            assigned = {int(t.split(":")[1]) for t in tasks if isinstance(t, str) and t.startswith("NEUTRALIZE:")}
            for c in contacts:
                yield lab, kind_of(r, c), c in assigned


def summarise_calls(rows):
    out = defaultdict(lambda: defaultdict(Counter))
    for lab, kind, v in rows:
        out[lab][kind][str(v) if not isinstance(v, float) else
                       ("0" if v == 0 else "<1" if v < 1 else "1" if v == 1 else ">1")] += 1
    return {lab: {k: dict(c) for k, c in kinds.items()} for lab, kinds in out.items()}


def summarise_calls(rows):
    out = defaultdict(lambda: defaultdict(Counter))
    for lab, kind, v in rows:
        out[lab][kind][str(v) if not isinstance(v, float) else
                       ("0" if v == 0 else "<1" if v < 1 else "1" if v == 1 else ">1")] += 1
    return {lab: {k: dict(c) for k, c in kinds.items()} for lab, kinds in out.items()}



def airspace(data):
    run = common.AIRSPACE[0]
    R = common.records(data, [run])
    bad = common.problems(R) + [(r["episode_id"], "no contact block") for r in R if "decoys" not in r["outcome"]]
    assert not bad, bad
    by_slug = {common.slug(r["episode_id"]): r for r in R}
    out = {"n_records": len(R), "arms": {}, "calls_answered": {}}
    refs = references(data)
    for lab in ARMS:
        rows = sorted((r for r in R if label(r) == lab), key=lambda r: r["seed"])
        per = {r["seed"]: outcome_row(r) for r in rows}
        a = {"episodes": len(rows), "per_seed": per,
             "breaches": sum(v["breach"] for v in per.values()),
             "reds_destroyed": sum(v["reds_destroyed"] for v in per.values()),
             "blues_lost_to_contacts": sum(v["blues_lost_to_contacts"] for v in per.values()),
             "blues_lost_to_contacts_by_kind": dict(sum((Counter(v["blues_lost_to_contacts_by_kind"])
                                                       for v in per.values()), Counter())),
             "blues_lost_total": sum(v["blues_lost_total"] for v in per.values())}
        if refs.get(lab):
            ref = {s: refs[lab][s] for s in per if s in refs[lab]}
            a["no_contact_reference"] = {
                "seeds": sorted(ref), "source_arm": lab,
                "breaches": sum(bool(x["outcome"]["touchdown"]) for x in ref.values()),
                "reds_destroyed": sum(int(x["outcome"]["num_neutralized"]) for x in ref.values()),
                "blues_lost_total": sum(int(x["outcome"]["n_blues_disabled"]) for x in ref.values()),
                "per_seed_breach": {s: bool(x["outcome"]["touchdown"]) for s, x in sorted(ref.items())}}
        out["arms"][lab] = a
    cache = os.path.join(common.run_dir(data, run), "cache", common.identity(data, run))
    out["calls_answered"] = summarise_calls(contact_calls(cache, by_slug))
    return out


# -- disguised intruders

def disguise_label(r):
    return r["rep"].split("camo-", 1)[1].split("-", 1)[1]


def disguise_cond(r):
    return r["rep"].split("camo-", 1)[1].split("-", 1)[0]


def disguise_outcome_row(r):
    o = r["outcome"]
    d = o.get("decoys") or {}
    by_kind = Counter(e["kind"] for e in d.get("events", []) if e["event"] == "destroyed")
    lost_decoys = int(d.get("n_destroyed", 0))
    return {"breach": bool(o["touchdown"]), "reds_destroyed": int(o["num_neutralized"]),
            "blues_lost_to_birds": int(by_kind.get("bird", 0)),
            "blues_lost_to_civilians": int(by_kind.get("civilian", 0)),
            "blues_lost_to_contacts": lost_decoys,
            "blues_lost_total": int(o["n_blues_disabled"]),
            "end": ("breach" if o["touchdown"] else
                    "all reds resolved" if all(x != "active" for x in o["red_outcomes"][:4]) else "time limit")}


def speeds(user_message):
    out = {}
    for line in str(user_message).splitlines():
        for rx in _VEL:
            m = rx.search(line)
            if m:
                out[int(m.group(1))] = math.hypot(float(m.group(2)), float(m.group(3)))
    return out


def disguise_calls(cache_dir, by_slug, label_fn):
    """Yield (label, cond, kind, set_aside, extra) per contact the model was prompted about."""
    for f in glob.glob(os.path.join(cache_dir, "*", "*.json")):
        slug_dir = os.path.basename(os.path.dirname(f))
        rec = json.load(open(f))
        prov = rec.get("provenance") or {}
        vc = prov.get("validation_context") or {}
        if rec.get("kind") == "transport_error" or not rec.get("content"):
            continue
        r = by_slug.get(slug_dir.replace("__near_frontier", ""))
        if r is None:
            continue
        lab, cond = label_fn(r)
        try:
            ans = json.loads(rec["content"])
        except ValueError:
            continue
        sp = speeds((prov.get("request") or {}).get("user_message", ""))
        kind = vc.get("kind")
        if kind == "defender" and lab == "B":
            pr = {int(x["contact_id"]): str(x["priority"]).upper() for x in ans.get("threat_priorities") or []
                  if isinstance(x, dict) and "contact_id" in x}
            for c in vc.get("believed_contacts") or []:
                v = pr.get(int(c), "MEDIUM")
                yield lab, cond, kind_of(r, c), v == "LOW", {"value": v, "speed": sp.get(int(c))}
        elif kind == "strategy_defender" and lab in ("W", "WSOL"):
            pr = {}
            for x in ans.get("contact_priorities") or []:
                try:
                    pr[int(x["contact_id"])] = float(x["priority"])
                except (KeyError, TypeError, ValueError):
                    pass
            for c in vc.get("believed_contacts") or []:
                v = pr.get(int(c), 1.0)
                yield lab, cond, kind_of(r, c), v == 0, {"value": v, "below1": v < 1, "speed": sp.get(int(c))}
        elif kind == "commander" and lab in ("CMD", "SOLCMD"):
            cat = vc.get("catalogue") or []
            contacts = sorted({int(t.split(":")[1]) for t in cat if t.startswith("NEUTRALIZE:")})
            tasks = set((ans.get("task_by_agent") or {}).values())
            assigned = {int(t.split(":")[1]) for t in tasks if isinstance(t, str) and t.startswith("NEUTRALIZE:")}
            for c in contacts:
                yield lab, cond, kind_of(r, c), c not in assigned, {"speed": sp.get(int(c))}


def summarise_disguise_calls(rows):
    out = defaultdict(lambda: defaultdict(lambda: Counter()))
    for lab, cond, kind, aside, extra in rows:
        cell = out[f"{lab}|{cond}"]
        cell[kind]["n"] += 1
        cell[kind]["set_aside"] += int(bool(aside))
        if "below1" in extra:
            cell[kind]["below1"] += int(bool(extra["below1"]))
        if kind == "drone":
            s = extra.get("speed")
            band = "unknown" if s is None else ("fast" if s > FAST else "not_fast")
            cell[f"drone_{band}"]["n"] += 1
            cell[f"drone_{band}"]["set_aside"] += int(bool(aside))
    res = {}
    for cell, kinds in sorted(out.items()):
        res[cell] = {k: dict(c, pct=round(100.0 * c["set_aside"] / c["n"]) if c["n"] else None)
                     for k, c in sorted(kinds.items())}
    return res


def team_rows(R, label_fn):
    out = {}
    for key in sorted({label_fn(r) for r in R}):
        rows = sorted((r for r in R if label_fn(r) == key), key=lambda r: r["seed"])
        per = {r["seed"]: disguise_outcome_row(r) for r in rows}
        out["|".join(key)] = {
            "episodes": len(rows), "per_seed": per,
            **{k: sum(v[k] for v in per.values()) for k in
               ("breach", "reds_destroyed", "blues_lost_to_birds", "blues_lost_to_civilians",
                "blues_lost_to_contacts", "blues_lost_total")},
            "ends": dict(Counter(v["end"] for v in per.values()))}
    return out



def disguise(data):
    run = common.DISGUISE[0]
    R = common.records(data, [run])
    bad = common.problems(R) + [(r["episode_id"], "no contact block") for r in R if "decoys" not in r["outcome"]]
    assert not bad, bad
    by_slug = {common.slug(r["episode_id"]): r for r in R}
    lab_c = lambda r: (disguise_label(r), disguise_cond(r))  # noqa: E731
    cache = os.path.join(common.run_dir(data, run), "cache", common.identity(data, run))
    out = {"n_records": len(R), "fast_threshold": FAST, "teams": team_rows(R, lab_c),
           "calls": summarise_disguise_calls(disguise_calls(cache, by_slug, lab_c))}
    arun = common.AIRSPACE[0]
    D = [r for r in common.records(data, [arun]) if label(r) in ARMS]
    lab_d = lambda r: (label(r), "airspace")  # noqa: E731
    by_slug_d = {common.slug(r["episode_id"]): r for r in D}
    acache = os.path.join(common.run_dir(data, arun), "cache", common.identity(data, arun))
    out["airspace_reference"] = {"teams": team_rows(D, lab_d),
                                 "calls": summarise_disguise_calls(disguise_calls(acache, by_slug_d, lab_d))}
    return out


if __name__ == "__main__":
    a = common.parser(__doc__.split("\n")[0]).parse_args()
    o = common.out_dir(a)
    common.dump(airspace(a.data), os.path.join(o, "airspace.json"))
    common.dump(disguise(a.data), os.path.join(o, "disguise.json"))
