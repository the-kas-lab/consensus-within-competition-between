"""Safeguards for a paid, multi-batch live run. Pure functions, so every decision the driver takes is tested:

* ``audit``: every live call against the reservation assumptions (prompt
  tokens <= request UTF-8 bytes + overhead, completion <= max_tokens, billed
  cost <= the reserved worst case);
* ``reconcile``: the account's usage change against the ledger's committed
  change since a recorded baseline. Spend the account shows beyond the ledger
  stops the run; an account that lags the ledger is only logged -- a lagging
  account total never releases budget (the ledger's liabilities are kept);
* ``batch_costs`` / ``projection``: observed batch costs reconstructed from
  the saved episode records (so a restart keeps them), and a projection of the
  remaining batches that never falls below the pre-run estimate;
* ``progress`` / ``completion``: attempted versus validly completed
  executions; unresolved failures stop the run; completion needs a clean audit
  and a validated full manifest.
"""
import glob
import json
import os

from scripts.run_competing_teams import manifest_status

REQUEST_OVERHEAD_TOKENS = 256


def audit(cache_root, price_in, price_out, overhead=REQUEST_OVERHEAD_TOKENS):
    """Every live cache record under ``cache_root`` (records with a request
    body and usage) against the reservation assumptions."""
    rows, bad = 0, []
    for f in sorted(glob.glob(os.path.join(cache_root, "**", "*.json"), recursive=True)):
        if os.path.basename(f) == "namespace.json":
            continue
        try:
            with open(f) as fh:
                rec = json.load(fh)
        except ValueError:
            bad.append({"file": f, "problems": ["unreadable record"]})
            continue
        prov, usage = rec.get("provenance") or {}, rec.get("usage") or {}
        body = prov.get("request_body")
        if not body or not usage:
            continue
        rows += 1
        nbytes = (sum(len(str(m["content"]).encode("utf-8")) for m in body["messages"])
                  + len(json.dumps(body.get("response_format")).encode("utf-8")))
        mt = int(body.get("max_tokens") or 0)
        pt, ct = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        worst = (nbytes + overhead) * price_in + mt * price_out
        problems = []
        if pt > nbytes + overhead:
            problems.append(f"prompt_tokens {pt} > bytes+{overhead} {nbytes + overhead}")
        if mt and ct > mt:
            problems.append(f"completion {ct} > max_tokens {mt}")
        if (usage.get("cost_usd") or 0) > worst + 1e-9:
            problems.append(f"cost {usage.get('cost_usd')} > reserved {worst:.6f}")
        if problems:
            bad.append({"file": f, "problems": problems})
    return {"calls_checked": rows, "violations": bad}


def reconcile(baseline, account_usage, ledger_committed, tolerance):
    """Account usage change vs ledger committed change since ``baseline``
    ({"account_usage", "ledger_committed"}, recorded once before the first
    batch). discrepancy = account delta - ledger delta.

    * discrepancy > tolerance: the account shows spend the ledger does not
      cover (another key, an unledgered call) -> "unaccounted_spend" (stop);
    * discrepancy < -tolerance: the account lags the ledger -> "account_lags"
      (log; the ledger's liabilities are kept, nothing is released);
    * else "ok".
    ``committed`` is the larger of the two views of cumulative spend, so a
    lag can only make the projection more conservative, never less."""
    d_acc = float(account_usage) - float(baseline["account_usage"])
    d_led = float(ledger_committed) - float(baseline["ledger_committed"])
    disc = d_acc - d_led
    status = ("unaccounted_spend" if disc > tolerance else
              "account_lags" if disc < -tolerance else "ok")
    return {"account_delta": round(d_acc, 6), "ledger_delta": round(d_led, 6),
            "discrepancy": round(disc, 6), "status": status,
            "committed": float(ledger_committed) + max(0.0, disc)}


def batch_costs(batches, records):
    """{batch tag: summed ``cost_usd_written``} for every batch whose
    executions ALL have a record, reconstructed from the saved records.
    ``batches``: {tag: [manifest entries]}."""
    by_id = {}
    for r in records:
        by_id.setdefault(r["episode_id"], []).append(r)
    out = {}
    for tag, entries in batches.items():
        if entries and all(e["episode_id"] in by_id for e in entries):
            out[tag] = sum(float(x.get("cost_usd_written") or 0.0)
                           for e in entries for x in by_id[e["episode_id"]])
    return out


def projection(committed, remaining_batches, observed_batch_costs, pre_run_batch_estimate):
    """Committed spend plus the remaining batches at the larger of the
    pre-run per-batch estimate and the most expensive observed batch: a cheap
    early batch never lowers the projection below the pre-run estimate."""
    per_batch = max([float(pre_run_batch_estimate)] + [float(c) for c in observed_batch_costs])
    return {"per_batch": per_batch, "remaining_batches": int(remaining_batches),
            "projected_committed": float(committed) + per_batch * int(remaining_batches)}


def progress(manifest, records, identity_hash):
    """Attempted versus validly completed executions of ``manifest``.
    ``unresolved``: attempted (some record exists) but not validly completed
    -- an invalid terminal state, a duplicate, a manifest or identity
    mismatch. A driver stops on any of these rather than re-running or
    counting them as done."""
    st = manifest_status(manifest, records, identity_hash)
    probs = st["problems"]
    return {"valid": sorted(i for i, p in probs.items() if p is None),
            "missing": sorted(i for i, p in probs.items() if p == "missing"),
            "unresolved": {i: p for i, p in sorted(probs.items()) if p not in (None, "missing")},
            "extra_records": st["extra_records"]}


def completion(manifest, records, identity_hash, audit_result):
    """Done only when the audit is clean (and checked at least one call) and
    every prespecified execution has exactly one valid record of the expected
    identity, with no records outside the manifest."""
    pr = progress(manifest, records, identity_hash)
    reasons = []
    if audit_result["violations"]:
        reasons.append(f"{len(audit_result['violations'])} reservation violations")
    if not audit_result["calls_checked"]:
        reasons.append("audit checked no calls")
    if pr["missing"]:
        reasons.append(f"{len(pr['missing'])} executions missing")
    if pr["unresolved"]:
        reasons.append(f"{len(pr['unresolved'])} executions not validly completed")
    if pr["extra_records"]:
        reasons.append(f"{len(pr['extra_records'])} records outside the manifest")
    return {"done": not reasons, "reasons": reasons, "valid": len(pr["valid"]),
            "prespecified": len(manifest)}
