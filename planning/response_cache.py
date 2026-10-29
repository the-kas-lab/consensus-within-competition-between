"""Content-addressed cache for LLM planning responses.

Built **before** the first paid call, deliberately. Every LLM experiment in this
workstream is a paired design over seeds, which means the same episode is
re-run whenever an analysis is re-cut, a test is added, or a reviewer asks for a
replay. Without a cache each of those re-runs is a fresh bill for an answer
already paid for; with one, only the first run costs anything.

It is also the only thing that makes a replay *reproducible*. ``temperature=0``
is not determinism -- no provider guarantees it, and a self-hosted server will
not either. A cached raw response replayed through the same parser is
byte-identical by construction.


**The key carries the prompt builder's version.** A content-addressed cache
keyed on the prompt alone invites one specific failure: edit the prompt, and
every stale entry answers a question no longer being asked, silently and
cheaply. Versioning the builder closes that at the key rather than relying on
anyone remembering to clear a directory.

**Key schema 2 adds the endpoint.** Schema 1 keyed on the model id
alone, so OpenRouter and a self-hosted server serving the same model id would
have shared entries. The endpoint is part of *which
question was asked of whom*, so it is part of the address.

**Records carry provenance, and failures are records.** Schema 1 stored only
``content`` and ``usage``: which model and provider produced a response was
recoverable only by recomputing a key, and a transport exception left no record
at all, so a replay of an episode that hit one silently asked the network again
and got a different episode. Now every record says what was asked, of which
endpoint and model, and what came back (``response``) or what failed
(``transport_error``). A replay reproduces both.
"""

import datetime as _dt
import hashlib
import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

KEY_SCHEMA = 2
RESPONSE = "response"
TRANSPORT_ERROR = "transport_error"


def normalize_endpoint(endpoint) -> str:
    """One spelling per endpoint: ``https://openrouter.ai/api/v1`` and
    ``https://openrouter.ai/api/v1/`` (how the OpenAI client reports it) must
    address the same entries."""
    if endpoint is None:
        return "unspecified"
    return str(endpoint).strip().rstrip("/")


@dataclass(frozen=True)
class ServingConfig:
    """What actually serves a model id: the namespace a cache entry belongs to.

    A model id plus an endpoint is NOT enough
    for a self-hosted server: an alias at ``localhost:8000``,
    and the checkpoint, revision, precision or server behind it can change
    while both strings stay the same. An entry written under one serving
    configuration must not answer a request made under another.

    Every field is part of the key whenever a ``ServingConfig`` is given. A
    field that could not be established is written as the literal string the
    caller declares (e.g. ``"unrecorded"``), never guessed: declaring it later
    then opens a new namespace rather than silently reusing this one.
    """

    checkpoint: str
    revision: Optional[str] = None
    server: Optional[str] = None
    server_version: Optional[str] = None
    dtype: Optional[str] = None
    quantization: Optional[str] = None
    max_model_len: Optional[int] = None
    # Which running server process answered, e.g. vLLM's
    # process_start_time_seconds. Not an immutable identity of the weights
    # (repeated "unrecorded" fields cannot distinguish restarts), but a restart
    # always opens a fresh namespace. In the key only when set, so
    # every earlier ServingConfig key is unchanged.
    instance: Optional[str] = None

    def as_dict(self) -> Dict:
        d = asdict(self)
        if self.instance is None:
            del d["instance"]
        return d


def cache_key(
    *,
    prompt_builder_version: str,
    system_prompt: str,
    user_message: Any,
    model: str,
    endpoint: str,
    decode_config: Optional[Dict] = None,
    serving: Optional[Dict] = None,
    request: Optional[Dict] = None,
) -> str:
    """The content address of one planning call.

    ``user_message`` is typed ``Any`` rather than ``str`` because the vision
    planners send an OpenAI multi-modal content *list*; ``sort_keys`` + the
    default ``str`` coercion keeps such a payload hashable without this module
    needing to know its shape.

    Every field that can change the model's answer is in the key -- including
    **who** was asked. ``endpoint`` has no default on purpose: a caller that
    forgets it must fail, not silently share entries across providers.

    ``serving`` (a ``ServingConfig.as_dict()``) enters the payload only when
    given, so every key computed before it existed is unchanged.

    ``request``: the backend request configuration --
    ``RequestConfig.as_dict()`` plus the JSON schema name and body actually
    sent. Also in the payload only when given. Without it the schema was
    implied by the prompt version alone; with it, a changed schema (a
    commander's task enum, an intruder's phase) can never be answered by an
    entry written under another, and neither can another provider pin.
    """
    payload = {
        "schema": KEY_SCHEMA,
        "v": str(prompt_builder_version),
        "system": system_prompt,
        "user": user_message,
        "model": str(model),
        "endpoint": normalize_endpoint(endpoint),
        "decode": decode_config or {},
    }
    if serving:
        payload["serving"] = dict(serving)
    if request:
        payload["request"] = dict(request)
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def legacy_cache_key_v1(
    *,
    prompt_builder_version: str,
    system_prompt: str,
    user_message: Any,
    model: str,
    decode_config: Optional[Dict] = None,
) -> str:
    """Schema 1's key, verbatim. Used ONLY to migrate entries written before
    schema 2, whose endpoint is not in their address. Never used
    for a lookup in a live or replay run."""
    payload = {
        "v": str(prompt_builder_version),
        "system": system_prompt,
        "user": user_message,
        "model": str(model),
        "decode": decode_config or {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class ResponseCache:
    """Raw model responses and transport failures, addressed by what was asked.

    ``path=None`` keeps everything in memory -- the default for tests, which
    want the semantics without a directory to clean up. With a path, entries are
    one JSON file each, written atomically (temp file + ``os.replace``) so a run
    interrupted mid-write leaves no half-file for the next run to parse.

    The cache stores the **raw response string**, never a parsed object. Parsing
    is the planner's job and has its own fallback behaviour; caching the parse
    would hide a schema change behind a hit.

    **Thread-safe.** Per-agent calls within one planning epoch may be issued
    concurrently (their views are frozen before any call), so counters and the
    in-memory map are updated under one lock. File writes are already atomic.
    """

    def __init__(self, path: Optional[str] = None, enabled: bool = True):
        self.path = path
        self.enabled = bool(enabled)
        self._memory: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self.path and self.enabled:
            os.makedirs(self.path, exist_ok=True)

    def _file(self, key: str) -> str:
        return os.path.join(self.path, f"{key}.json")

    def get_record(self, key: str) -> Optional[Dict]:
        """The whole cached record -- response or recorded failure -- or None.
        Counts a hit or a miss either way.

        A corrupt or unreadable entry is treated as a **miss**, not an error: a
        cache that can crash a 400-step episode is worse than no cache. A record
        with no ``kind`` predates schema 2's records and is a response.
        """
        if not self.enabled:
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            record = self._memory.get(key)
        if record is None and self.path:
            try:
                with open(self._file(key), "r", encoding="utf-8") as f:
                    record = json.load(f)
            except (OSError, ValueError):
                record = None
        with self._lock:
            if record is None:
                self.misses += 1
                return None
            self.hits += 1
        record = dict(record)
        record.setdefault("kind", RESPONSE)
        return record

    def get(self, key: str) -> Optional[str]:
        """The cached raw response string, or None -- including for a recorded
        transport failure, which has no content. Callers that must replay
        failures use ``get_record``."""
        record = self.get_record(key)
        if record is None or record["kind"] != RESPONSE:
            return None
        return record.get("content")

    def put(self, key: str, content: str, usage: Optional[Dict] = None,
            provenance: Optional[Dict] = None) -> None:
        """Record one response. ``usage`` is the provider's token counts, kept
        so spend can be reported from the cache rather than estimated;
        ``provenance`` says what was asked, of whom, and who answered."""
        self._write(key, {
            "kind": RESPONSE,
            "content": content,
            "usage": usage or {},
            "provenance": provenance or {},
        })

    def put_failure(self, key: str, error_type: str, error_message: str = "",
                    provenance: Optional[Dict] = None) -> None:
        """Record a call that raised instead of answering.

        Replayed as the same failure, so an episode that lost a call to the
        network reproduces as the episode that actually ran -- not as a
        different episode in which the replay happened to get through.
        """
        self._write(key, {
            "kind": TRANSPORT_ERROR,
            "error_type": str(error_type),
            "error_message": str(error_message)[:500],
            "usage": {},
            "provenance": provenance or {},
        })

    def _write(self, key: str, record: Dict) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._memory[key] = record
            self.writes += 1
        if not self.path:
            return
        fd, tmp = tempfile.mkstemp(dir=self.path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
            os.replace(tmp, self._file(key))
        except BaseException:
            # Never let a cache-write failure take the episode down with it.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def usage_totals(self) -> Dict[str, int]:
        """Token totals over everything **this cache object wrote**.

        In-memory only, and deliberately so: it answers "what did this run
        spend", not "what is in the directory". A run served entirely from disk
        spent nothing and reports zeros, which is the honest answer. Recorded
        transport failures are counted apart: they cost no tokens that the
        provider reported, and are not calls that returned. Migrated records
        are counted apart too: re-addressing an entry buys nothing.
        """
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        failures = migrated = 0
        with self._lock:
            records = list(self._memory.values())
        for record in records:
            if record.get("kind", RESPONSE) != RESPONSE:
                failures += 1
                continue
            if (record.get("provenance") or {}).get("migrated_from_key_schema"):
                # Re-addressed, not bought. Counting its usage would report the
                # original run's spend a second time.
                migrated += 1
                continue
            usage = record.get("usage") or {}
            totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            totals["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            totals["calls"] += 1
        if failures:
            totals["transport_failures"] = failures
        if migrated:
            totals["migrated"] = migrated
        return totals

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "writes": self.writes}
