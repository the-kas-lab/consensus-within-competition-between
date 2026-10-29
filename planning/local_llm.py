"""Per-agent LLM role planning: one planner per blue, on one blue's belief.

The LLM planner. See

What this replaces, and why it is not just the old planner called N times:

* ``LLMRolePlanner`` (``planner.py:896``) is a *commander*. It consumes a
  team-level ``PlannerObservation``, assigns every blue at once, and hands its
  targets back through ``self.last_target_red_ids`` -- a side channel that works
  only because there is exactly one planner object. With one planner per agent,
  "the planner's last targets" stops naming anything.
* Here ``plan_one(view) -> PlannerOutput`` returns roles **and** targets
  explicitly, and these classes set no ``last_*`` attribute at all. The side
  channel is not worked around on this path; it does not exist on it.

**Independent context.** Each ``LocalLLMPlanner`` is a distinct object for one
``agent_id``, holding no reference to any other agent's planner, view or output.
Calls are single-turn and stateless, so there is no channel through which epoch
k's answer could reach epoch k's other calls. ``Simulation._plan_per_agent``
already freezes every view before any call runs; this module's job
is not to reintroduce a leak behind that guarantee.

**Failure is silent to the run and loud to the record.** A schema-invalid or
failed response falls back to DEFEND *for that one agent* and the episode
continues -- which, per-agent, is a real improvement over the team planner,
where one bad response defended the whole team. Every such event is recorded in
``fallback_events`` and surfaced by ``Simulation.episode_outcome()``. A run with
a high fallback rate is a run of the *rule* fallback wearing an LLM's name, and
no result may be read before that number has been.
"""

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, ValidationError, field_validator
from typing_extensions import Literal

from planning.planner import (
    _OPENROUTER_BASE_URL,
    _create_completion_with_backoff,
    PlannerOutput,
)
from planning.prompt_builder import (
    PROMPT_BUILDER_VERSION,
    PromptContext,
    build_system_prompt,
    build_user_message,
)
from planning.response_cache import (
    RESPONSE,
    TRANSPORT_ERROR,
    ResponseCache,
    ServingConfig,
    cache_key,
    legacy_cache_key_v1,
    normalize_endpoint,
    utc_now,
)
from planning.request_config import RequestConfig
from planning.roles import BlueRole
from planning.spend import BILLED, ESTIMATED, NOT_BILLED, UNVERIFIED, BudgetExhausted, SpendLedger
from planning.stops import StopRun

# The model a planner uses when none is given. Every reported run names its model
# explicitly through an inference profile (scripts/inference_profile.py).
DEFAULT_LOCAL_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"


@dataclass(frozen=True)
class DecodeConfig:
    """Decode settings, frozen and part of the cache key.

    ``temperature=0`` is **not** determinism -- no provider guarantees it. It is
    here to reduce variance, and nothing about reproducibility rests on it. The
    cache is what makes a replay byte-identical.

    ``max_tokens`` is 2000 by default because reasoning models spend the budget
    on reasoning first and emit the answer last: too small a cap does not
    truncate the answer, it prevents the answer entirely.

    ``reasoning_effort`` is an OpenRouter routing hint and defaults to **off**,
    so a self-hosted OpenAI-compatible server is sent nothing it would not
    understand.

    ``enable_thinking`` is a chat-template switch for self-hosted servers, sent
    as ``extra_body={"chat_template_kwargs": {"enable_thinking": False}}``.
    ``None``, the default, sends nothing and leaves the server's own default in
    force.

    It is in the cache key only when set, so every key written before it
    existed is unchanged, and a thinking-off call can never be served a
    thinking-on response.
    """

    # None = omit the field: some hosted reasoning models (e.g. OpenAI's)
    # refuse a temperature, and a pinned endpoint must support every
    # parameter sent (require_parameters).
    temperature: Optional[float] = 0.0
    max_tokens: int = 2000
    reasoning_effort: Optional[str] = None
    enable_thinking: Optional[bool] = None
    # OpenRouter's unified ``reasoning`` request object, as canonical JSON
    # (e.g. '{"enabled": false}', '{"effort": "low"}', '{"max_tokens": 512}'),
    # for a hosted model whose reasoning is on by default. Sent only toward
    # OpenRouter (RequestConfig refuses it for vLLM); absent from the key and
    # provenance when None, like enable_thinking.
    reasoning: Optional[str] = None

    def __post_init__(self):
        if self.reasoning is not None:
            obj = json.loads(self.reasoning)
            if not isinstance(obj, dict) or not obj:
                raise ValueError("reasoning must be a non-empty JSON object")
            object.__setattr__(self, "reasoning", json.dumps(obj, sort_keys=True))
            if self.reasoning_effort:
                raise ValueError("set reasoning or reasoning_effort, not both")

    def as_dict(self) -> Dict:
        d = asdict(self)
        if self.enable_thinking is None:
            # Absent, not None: the key of every earlier record stays exactly
            # what it was (tests/test_thinking_and_serving.py pins the digests).
            del d["enable_thinking"]
        else:
            d["enable_thinking"] = bool(self.enable_thinking)
        if self.reasoning is None:
            del d["reasoning"]
        return d

    def extra_body(self) -> Optional[Dict]:
        body = {}
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if self.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}
        return body or None


class ReplayMiss(StopRun):
    """A replay-only planner was asked a question its cache cannot answer.

    Deliberately NOT turned into a fallback. A replay that silently became a
    DEFEND would be a different episode passed off as the recorded one; the
    replay must stop, loudly, at the first question it cannot answer.
    """


class TransportFailure(RuntimeError):
    """A call that raised instead of answering -- live, or replayed from its
    record. Carries the original exception's type name, so a live failure and
    its replay produce the identical fallback event."""

    def __init__(self, error_type: str, message: str = ""):
        super().__init__(f"{error_type}: {message}" if message else error_type)
        self.error_type = str(error_type)


class _LocalRoleDecision(BaseModel):
    """One agent's decision for itself. Not a list -- that is the whole point.

    **Syntax is normalised; semantics never are.** Measured on the first valid
    smoke run: 4 of 36 responses came back with ``"RECON"`` rather than
    ``"recon"`` and with the irrelevant target field simply omitted. Neither is
    a tactical error -- ``"RECON"`` unambiguously means recon, and an agent that
    is not neutralizing has no contact id to give -- and both were partly caused
    by a prompt that spells the roles in capitals. The prompt now says exactly
    what the JSON must contain, and this model accepts the two harmless
    variants.

    The line is drawn at guessing: a missing target defaults to -1, which is
    *not* a believed contact and *not* a valid region, so a NEUTRALIZE with no
    contact id still falls back to DEFEND and is still recorded. Repairing
    syntax keeps a well-judged decision; inventing a target would manufacture
    one.
    """

    role: str
    target_contact_id: int = -1
    target_region_id: int = -1
    reasoning: str = ""

    @field_validator("role", mode="before")
    @classmethod
    def _normalise_role(cls, value):
        if isinstance(value, str):
            value = value.strip().lower()
        if value not in ("defend", "neutralize", "recon"):
            raise ValueError(
                f"role must be defend, neutralize or recon -- got {value!r}"
            )
        return value


_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"type": "string", "enum": ["defend", "neutralize", "recon"]},
        "target_contact_id": {
            "type": "integer",
            "description": (
                "contact_id you will pursue -- must be one of the contact_id "
                "values you were shown. -1 for defend/recon."
            ),
        },
        "target_region_id": {
            "type": "integer",
            "description": (
                "region_id you will patrol toward -- must be one of the "
                "region_id values you were shown. -1 for defend/neutralize."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "One sentence, referring to the actual picture.",
        },
    },
    "required": ["role", "target_contact_id", "target_region_id", "reasoning"],
    "additionalProperties": False,
}


def build_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> OpenAI:
    """An OpenAI-compatible client, pointed wherever the caller says.

    ``base_url`` is an argument rather than a module constant so that pointing
    at another server, for example ``http://localhost:8000/v1``, is a call-site
    change. Defaults to OpenRouter, which every reported run uses.

    A local server usually needs no credential, so an empty key is accepted for
    a non-default base URL and rejected for the default one -- silently sending
    no key to OpenRouter would fail later and less clearly.

    ``timeout`` (seconds per request) and ``max_retries`` default to the
    client library's own, so existing callers are unchanged. A run that must
    not hang on a stalled server sets both: the library retries a timed-out
    request by itself, so the worst case per call is about
    ``timeout * (max_retries + 1)``.
    """
    import os

    resolved_base = base_url or _OPENROUTER_BASE_URL
    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not resolved_key:
        if resolved_base == _OPENROUTER_BASE_URL:
            raise ValueError(
                "An OpenRouter API key is required -- pass api_key=, or set the "
                "OPENROUTER_API_KEY environment variable."
            )
        resolved_key = "not-needed"
    extra = {}
    if timeout is not None:
        extra["timeout"] = float(timeout)
    if max_retries is not None:
        extra["max_retries"] = int(max_retries)
    return OpenAI(base_url=resolved_base, api_key=resolved_key, **extra)


def _billed_cost(usage) -> Optional[float]:
    """The billed cost OpenRouter reports in ``usage.cost`` under usage
    accounting, or None (vLLM, a fake client, accounting off)."""
    if usage is None:
        return None
    cost = getattr(usage, "cost", None)
    if cost is None:
        extra = getattr(usage, "model_extra", None) or {}
        cost = extra.get("cost")
    try:
        return None if cost is None else float(cost)
    except (TypeError, ValueError):
        return None


def _token_categories(usage) -> Dict:
    """Billed token categories beyond prompt/completion, when the backend
    reports them: reasoning tokens (inside completion_tokens), prompt tokens
    read from the provider's cache, and prompt tokens written to it. Absent
    keys mean "not reported", never zero."""
    if usage is None:
        return {}

    def field(obj, name):
        if obj is None:
            return None
        v = getattr(obj, name, None)
        if v is None and isinstance(obj, dict):
            v = obj.get(name)
        if v is None:
            v = (getattr(obj, "model_extra", None) or {}).get(name)
        return v

    out = {}
    ctd = field(usage, "completion_tokens_details")
    ptd = field(usage, "prompt_tokens_details")
    for key, obj, name in (("reasoning_tokens", ctd, "reasoning_tokens"),
                           ("cached_tokens", ptd, "cached_tokens"),
                           ("cache_write_tokens", ptd, "cache_write_tokens")):
        v = field(obj, name)
        if v is not None:
            try:
                out[key] = int(v)
            except (TypeError, ValueError):
                pass
    return out


# -- contact images -----------------------
# Off unless a planner is given an image lookup (contact_id -> image path). With
# none, every request, cache key and record is byte-identical to before.
IMAGE_MODE_VERSION = "img.1"
# Inserted after the shared objective-only brief (coordination.mission_policy.
# blue_brief_v1), whose last sentence is _BRIEF_END. Facts only, no advice.
_BRIEF_END = "Reds can change direction at any time, and reds your team has not detected may exist."
DECOY_BRIEF_V1 = (
    " Not every contact is a red. Besides the red intruders, the airspace holds birds and "
    "civilian drones. They move in the same airspace, cannot breach, and are not part of the "
    "objective. A blue that touches any contact, red, bird or civilian drone, destroys it and is "
    "permanently disabled. The episode ends without a breach once every red has been destroyed "
    "or has left the arena, whatever the birds and civilian drones do. A contact's id does not "
    "tell what it is. One image of each contact listed below is attached after this message, "
    "labelled with its contact_id.")
# (bird-drone intruders): the warned image mode adds one heads-up sentence
# after "A contact's id does not tell what it is." and is otherwise img.1. Opt-in.
DECOY_WARNING_V1 = "Some red intruders may be disguised as birds."
_ID_SENTENCE = "A contact's id does not tell what it is."
WARNED_IMAGE_MODE_VERSION = "img.1w"
DECOY_BRIEFS = {
    IMAGE_MODE_VERSION: DECOY_BRIEF_V1,
    WARNED_IMAGE_MODE_VERSION: DECOY_BRIEF_V1.replace(
        _ID_SENTENCE, f"{_ID_SENTENCE} {DECOY_WARNING_V1}", 1),
}
# Ledger reservation per attached image, in the ledger's byte unit (an upper
# bound on its tokens; providers bill a low-detail image well below this).
IMAGE_RESERVE_BYTES = 6000


def with_decoy_brief(system_prompt: str, mode: str = IMAGE_MODE_VERSION) -> str:
    if system_prompt.count(_BRIEF_END) != 1:
        raise ValueError("image mode needs the objective-only brief (blue_brief_v1) in the "
                         "system prompt, exactly once")
    if mode not in DECOY_BRIEFS:
        raise ValueError(f"unknown image mode {mode!r}; expected one of {sorted(DECOY_BRIEFS)}")
    return system_prompt.replace(_BRIEF_END, _BRIEF_END + DECOY_BRIEFS[mode], 1)


def _image_sha(path: str) -> str:
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _image_parts(user_message: str, images):
    """The user content as parts: the text, then per image a label and the
    image (low detail, as the co-author's Experiment II)."""
    from planning.planner import _encode_image_data_uri

    parts = [{"type": "text", "text": user_message}]
    for cid, path in images:
        parts.append({"type": "text", "text": f"image of contact_id {int(cid)}:"})
        parts.append({"type": "image_url",
                      "image_url": {"url": _encode_image_data_uri(path), "detail": "low"}})
    return parts


def _image_manifest(images) -> str:
    return "\n".join(f"[attached image: contact_id {int(cid)} sha256 {_image_sha(path)}]"
                      for cid, path in images)


class LocalLLMPlanner:
    """One blue's planner. Independent context, explicit output.

    Deliberately **no schema-validation retry.** The team planner retries twice,
    which made sense when one bad response defended the entire team; here a slip
    costs one agent one cycle. More importantly, a retry would make the call
    unreproducible unless the whole attempt *sequence* were cached, so a single
    call keeps "the cache replays exactly what happened" true. Rate-limit
    backoff is unaffected -- a 429 is not a semantic retry.
    """

    # What this planner asks for. Class attributes so the auction bid-modifier bridge
    # can reuse the whole cached, provenance-recording, failure-replaying call
    # path with its own schema and prompt version -- and so this class's keys
    # stay exactly what they were.
    _SCHEMA_NAME = "local_role_decision"
    _SCHEMA = _RESPONSE_SCHEMA
    _PROMPT_VERSION = PROMPT_BUILDER_VERSION

    def __init__(
        self,
        agent_id: int,
        context: PromptContext,
        model: str = DEFAULT_LOCAL_MODEL,
        client: Optional[OpenAI] = None,
        cache: Optional[ResponseCache] = None,
        decode: Optional[DecodeConfig] = None,
        endpoint: Optional[str] = None,
        replay_only: bool = False,
        migrate_legacy_v1: bool = False,
        serving: Optional[ServingConfig] = None,
        request_timeout_s: Optional[float] = None,
        request_config: Optional[RequestConfig] = None,
        ledger: Optional[SpendLedger] = None,
        max_rate_limit_retries: int = 5,
        max_connection_retries: int = 2,
    ):
        self.agent_id = int(agent_id)
        self.context = context
        self.model = str(model)
        if replay_only and client is not None:
            raise ValueError("a replay-only planner refuses a client: it must not reach a network")
        self.client = client
        self.cache = cache if cache is not None else ResponseCache()
        self.decode = decode or DecodeConfig()
        # Part of the cache key (schema 2): who was asked, not only which model.
        self.endpoint = normalize_endpoint(endpoint)
        # What serves the model behind the id, when declared: checkpoint,
        # revision, server version, dtype, quantization, context length. In the
        # key only when set, so an alias pointing at a new checkpoint cannot be
        # served the old one's answers.
        self.serving = serving
        # Per request, sent to the client. Deliberately NOT in the key: it
        # cannot change an answer, only whether one arrives, and a timeout is
        # recorded and replayed as the transport failure it was.
        self.request_timeout_s = None if request_timeout_s is None else float(request_timeout_s)
        self.replay_only = bool(replay_only)
        self.migrate_legacy_v1 = bool(migrate_legacy_v1)
        # Backend request configuration (provider pin, accounting, strict
        # schema): in the key with the schema actually sent, and refused when
        # it would send one backend's fields to another (request_config.py).
        self.request_config = request_config
        if request_config is not None:
            request_config.check(self.decode, self.endpoint)
        # Spend guard shared across planners, threads and processes. Every
        # attempt reserves its worst case before it is sent (planning/spend.py).
        self.ledger = ledger
        self.max_rate_limit_retries = int(max_rate_limit_retries)
        self.max_connection_retries = int(max_connection_retries)
        # Provenance of every record this planner was served or wrote, by
        # endpoint -- so "did any of this come from somewhere other than
        # OpenRouter?" is answerable from the diagnostics.
        self.endpoints_served: Counter = Counter()
        self.transport_failures = 0
        self.migrated_v1 = 0
        # Per-agent diagnostics. Read by PerAgentLLMPlanner, which aggregates
        # them for episode_outcome().
        self.fallback_events: List[Dict] = []
        self.last_reasoning: Optional[str] = None
        self.calls = 0
        self.cache_hits = 0
        self.last_key: Optional[str] = None

    # -- the call ---------------------------------------------------------

    # Contact images. None = off (the class default).
    image_lookup = None
    image_mode = IMAGE_MODE_VERSION

    def enable_images(self, lookup, mode: str = IMAGE_MODE_VERSION) -> None:
        """Attach one image per believed contact to every call (``lookup``:
        contact_id -> image path or None), add the mode's non-hostile brief
        (DECOY_BRIEFS) to the system prompt, and version the prompt (cache keys
        cannot collide with text-only calls or across modes)."""
        if mode not in DECOY_BRIEFS:
            raise ValueError(f"unknown image mode {mode!r}; expected one of {sorted(DECOY_BRIEFS)}")
        if self.image_lookup is not None:
            if mode != self.image_mode:
                raise ValueError(f"images already enabled in mode {self.image_mode!r}, not {mode!r}")
            return
        self.image_lookup = lookup
        self.image_mode = mode
        self._PROMPT_VERSION = f"{self._PROMPT_VERSION}+{mode}"

    def _image_request(self, system_prompt, contact_ids):
        """(system prompt, images) for this call: unchanged and None when off."""
        if self.image_lookup is None:
            return system_prompt, None
        images = [(int(c), self.image_lookup(int(c))) for c in sorted(set(int(c) for c in contact_ids))]
        return with_decoy_brief(system_prompt, self.image_mode), [(c, p) for c, p in images if p]

    def _request_provenance(self, system_prompt, user_message) -> Dict:
        """What was asked, of whom -- stored with the record, readable without
        recomputing any key."""
        return {
            "key_schema": 2,
            "recorded_at": utc_now(),
            "model": self.model,
            "endpoint": self.endpoint,
            "prompt_builder_version": self._PROMPT_VERSION,
            "decode": self.decode.as_dict(),
            **({"serving": self.serving.as_dict()} if self.serving is not None else {}),
            **({"request_timeout_s": self.request_timeout_s}
               if self.request_timeout_s is not None else {}),
            **({"request_config": self.request_config.as_dict()}
               if self.request_config is not None else {}),
            "request": {"system_prompt": system_prompt, "user_message": user_message},
        }

    def _request_key_part(self) -> Optional[Dict]:
        """The request config plus the JSON schema actually sent, for the key.
        None without a request config, so legacy keys are unchanged."""
        if self.request_config is None:
            return None
        return {**self.request_config.as_dict(), "schema_name": self._SCHEMA_NAME,
                "schema": self._SCHEMA}

    def _migrate_v1(self, key, system_prompt, user_message) -> Optional[Dict]:
        """Re-address one schema-1 entry under its schema-2 key.

        Only for OpenRouter: schema-1 entries have no endpoint in their address,
        and the only ones that exist were written by the LLM planner smoke, which used
        OpenRouter alone. The provenance says the endpoint is ASSERTED from that
        record, not observed.
        """
        if self.endpoint != normalize_endpoint(_OPENROUTER_BASE_URL):
            return None
        if self.serving is not None or self.decode.enable_thinking is not None:
            return None  # schema-1 entries predate both, so they answer neither
        if self._PROMPT_VERSION != PROMPT_BUILDER_VERSION:
            return None  # schema 1 only ever held the LLM planner role decisions
        old = self.cache.get_record(legacy_cache_key_v1(
            prompt_builder_version=PROMPT_BUILDER_VERSION,
            system_prompt=system_prompt,
            user_message=user_message,
            model=self.model,
            decode_config=self.decode.as_dict(),
        ))
        if old is None or old["kind"] != RESPONSE:
            return None
        provenance = self._request_provenance(system_prompt, user_message)
        provenance["migrated_from_key_schema"] = 1
        provenance["endpoint_basis"] = (
            "asserted: schema-1 entries were written only by the LLM planner smoke, "
            "which used OpenRouter alone"
        )
        self.cache.put(key, old.get("content"), old.get("usage"), provenance)
        self.migrated_v1 += 1
        return self.cache.get_record(key)

    def _fetch(self, system_prompt: str, user_message: str,
               validation_context: Optional[Dict] = None, images=None) -> str:
        """The raw response string, from the cache if it is there.

        A cache hit makes **no network call at all**, which is what makes a
        repeated episode free and is asserted by injecting a client that raises
        on use. A cached **transport failure** is replayed as that failure.
        A replay-only planner raises ``ReplayMiss`` on a miss instead of calling
        anything.

        ``validation_context`` is what the semantic validator needs to judge the
        answer offline (believed contacts, candidate tasks, catalogue, the
        intruder's observation). Stored in the record's provenance, never in
        the key: it is derived from the same state as the prompt.
        """
        images = [(int(c), str(p)) for c, p in (images or [])]
        # The attached images enter the key through their hashes; with
        # none, key_message is the user message itself (keys unchanged).
        key_message = user_message if not images else user_message + "\n" + _image_manifest(images)
        key = cache_key(
            prompt_builder_version=self._PROMPT_VERSION,
            system_prompt=system_prompt,
            user_message=key_message,
            model=self.model,
            endpoint=self.endpoint,
            decode_config=self.decode.as_dict(),
            serving=None if self.serving is None else self.serving.as_dict(),
            request=self._request_key_part(),
        )
        self.last_key = key  # which record answered (or will answer) this call
        record = self.cache.get_record(key)
        if record is None and self.migrate_legacy_v1:
            record = self._migrate_v1(key, system_prompt, user_message)
        if record is not None:
            self.cache_hits += 1
            self.endpoints_served[
                (record.get("provenance") or {}).get("endpoint", "unrecorded")
            ] += 1
            if record["kind"] == TRANSPORT_ERROR:
                self.transport_failures += 1
                raise TransportFailure(record.get("error_type", "unknown"))
            return record.get("content") or ""

        if self.replay_only:
            raise ReplayMiss(
                f"agent {self.agent_id}: no cached record for this prompt at "
                f"{self.model} @ {self.endpoint}. A replay-only run makes no call."
            )
        if self.client is None:
            raise RuntimeError(
                "No client configured and no cached response for this prompt. "
                "A cache-only planner cannot answer a question it has not been "
                "asked before."
            )
        provenance = self._request_provenance(system_prompt, user_message)
        if images:
            provenance["request"]["images"] = [
                {"contact_id": c, "file": os.path.basename(p), "sha256": _image_sha(p)}
                for c, p in images]
        if validation_context is not None:
            provenance["validation_context"] = validation_context
        rc = self.request_config
        extra = rc.extra_body(self.decode) if rc is not None else self.decode.extra_body()
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",
                 "content": user_message if not images else _image_parts(user_message, images)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": self._SCHEMA_NAME,
                    "strict": True if rc is None else bool(rc.strict_schema),
                    "schema": self._SCHEMA,
                },
            },
            max_tokens=self.decode.max_tokens,
            **({"temperature": self.decode.temperature}
               if self.decode.temperature is not None else {}),
            **({"extra_body": extra} if extra else {}),
        )
        if rc is not None:
            # The raw request, exactly as sent (the transport timeout aside).
            provenance["request_body"] = json.loads(json.dumps(kwargs))
            if images:
                # Hashes, not base64: the images are files in the repo.
                for part in provenance["request_body"]["messages"][1]["content"]:
                    if part.get("type") == "image_url":
                        part["image_url"]["url"] = "omitted (see request.images)"
        if self.request_timeout_s is not None:
            kwargs["timeout"] = self.request_timeout_s
        started = time.monotonic()
        self.calls += 1
        try:
            if self.ledger is not None or rc is not None:
                response = self._create_with_attempts(kwargs, provenance)
            else:
                response = _create_completion_with_backoff(self.client, **kwargs)
            # A reasoning model that exhausts its budget returns content=None.
            # That is a real response and must be cached as one: cached as None
            # it would be indistinguishable from "no entry" on read, so the
            # replay would go back to the network and the failure would not
            # reproduce. "" is falsy but not missing.
            content = response.choices[0].message.content or ""
        except BudgetExhausted:
            # Not a transport failure and not cached: nothing was sent. The
            # run stops here (StopRun), so no fallback is ever recorded for it.
            self.calls -= 1
            raise
        except Exception as exc:  # noqa: BLE001 -- recorded, then re-raised typed
            # Before schema 2 this fell straight through to a fallback with no
            # record, so a replay asked the network again and ran a different
            # episode. Recorded now, and replayed as the same failure.
            provenance["latency_s"] = round(time.monotonic() - started, 3)
            self.cache.put_failure(key, type(exc).__name__, str(exc), provenance)
            self.transport_failures += 1
            self.endpoints_served[self.endpoint] += 1
            raise TransportFailure(type(exc).__name__, str(exc)) from exc

        usage = getattr(response, "usage", None)
        extra_fields = getattr(response, "model_extra", None) or {}
        provenance.update({
            "latency_s": round(time.monotonic() - started, 3),
            "response_model": getattr(response, "model", None),
            "response_id": getattr(response, "id", None),
            # OpenRouter names the upstream provider that served the call.
            "provider": getattr(response, "provider", None) or extra_fields.get("provider"),
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
        })
        if rc is not None:
            # The raw response, as the client parsed it (model_dump keeps the
            # provider-specific fields: provider, usage.cost, native finish).
            dump = getattr(response, "model_dump", None)
            provenance["raw_response"] = dump() if callable(dump) else None
        usage_record = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
        usage_record.update(_token_categories(usage))
        cost = _billed_cost(usage)
        if cost is not None:
            usage_record["cost_usd"] = cost
        # Cached even when it turns out to be schema-invalid, deliberately: the
        # replay then reproduces the same fallback rather than quietly asking
        # again and getting a different answer.
        self.cache.put(key, content, usage_record, provenance)
        self.endpoints_served[self.endpoint] += 1
        return content

    def _create_with_attempts(self, kwargs: Dict, provenance: Dict):
        """One call, every attempt recorded in ``provenance["attempts"]`` and,
        under a spend ledger, reserved before it is sent and settled after
        (planning/spend.py). Used whenever a request config or a ledger is set.

        Retries (transport/provider faults only -- an answer that arrives is
        never re-asked, however poor): HTTP 429 and connection errors that are
        not timeouts by default (``max_rate_limit_retries``,
        ``max_connection_retries``; a stale keep-alive connection fails
        within milliseconds, before the server sees the request). A request config's ``retry``
        (planning/request_config.RetryPolicy) replaces those defaults and can
        also retry provider 5xx errors, a 200 response carrying no choices
        (an upstream error in the body), and timeouts -- each a declared,
        frozen choice, since a timed-out request may still be billed. Under a
        ledger a failed attempt stays committed at its worst case. Every
        attempt has an id and a UTC start time. The client must not retry by
        itself (build it with ``max_retries=0``), or attempts would go
        unrecorded. ``BudgetExhausted`` propagates unrecorded.
        """
        import uuid

        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

        # UTF-8 bytes (= characters for ASCII): the unit the ledger's
        # "bytes_upper" reservation bounds tokens by.
        def _content_bytes(c):
            if isinstance(c, str):
                return len(c.encode("utf-8"))
            return sum(len(x["text"].encode("utf-8")) if x.get("type") == "text"
                       else IMAGE_RESERVE_BYTES for x in c)  # image parts

        prompt_chars = (sum(_content_bytes(m["content"]) for m in kwargs["messages"])
                        + len(json.dumps(kwargs["response_format"]).encode("utf-8")))
        attempts = provenance.setdefault("attempts", [])
        led = self.ledger
        policy = getattr(self.request_config, "retry", None)
        limits = ({"rate_limit": self.max_rate_limit_retries,
                   "connection": self.max_connection_retries, "server_error": 0,
                   "empty_choices": 0, "timeout": 0}
                  if policy is None else policy.limits())
        used = {k: 0 for k in limits}

        def retry(kind):
            if used[kind] >= limits[kind]:
                return False
            used[kind] += 1
            time.sleep((1.0 * 2 ** (used[kind] - 1)) if kind == "rate_limit" else 0.5 * used[kind])
            return True

        while True:
            rid = led.reserve(prompt_chars, kwargs["max_tokens"]) if led is not None else None
            t0 = time.monotonic()
            base = {"attempt_id": uuid.uuid4().hex, "started_at": utc_now()}
            try:
                response = self.client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                if led is not None:
                    led.settle(rid, NOT_BILLED)
                attempts.append(dict(base, outcome=NOT_BILLED, error_type=type(exc).__name__,
                                     status_code=429, retry_kind="rate_limit",
                                     latency_s=round(time.monotonic() - t0, 3)))
                if retry("rate_limit"):
                    continue
                raise
            except Exception as exc:  # noqa: BLE001 -- settled, recorded, retried or re-raised
                status = getattr(exc, "status_code", None)
                outcome = NOT_BILLED if (isinstance(exc, APIStatusError) and status == 402) else UNVERIFIED
                amount = led.settle(rid, outcome) if led is not None else 0.0
                if isinstance(exc, APITimeoutError):
                    kind = "timeout"
                elif isinstance(exc, APIConnectionError):
                    kind = "connection"
                elif isinstance(exc, APIStatusError) and status is not None and status >= 500:
                    kind = "server_error"
                else:
                    kind = None
                attempts.append(dict(base, outcome=outcome, error_type=type(exc).__name__,
                                     status_code=status, committed_usd=amount, retry_kind=kind,
                                     latency_s=round(time.monotonic() - t0, 3)))
                if kind is not None and retry(kind):
                    continue
                raise
            usage = getattr(response, "usage", None)
            choices = getattr(response, "choices", None)
            if led is None:
                amount, outcome = 0.0, "ok"
            else:
                cost = _billed_cost(usage)
                if cost is not None:
                    amount, outcome = led.settle(rid, BILLED, cost), BILLED
                else:
                    amount = led.settle(rid, ESTIMATED, led.cost_of(
                        getattr(usage, "prompt_tokens", 0) or 0,
                        getattr(usage, "completion_tokens", 0) or 0))
                    outcome = ESTIMATED
            rec = dict(base, outcome=outcome, latency_s=round(time.monotonic() - t0, 3))
            if led is not None:
                rec["committed_usd"] = amount
            if not choices:
                # An upstream error delivered as a 200 with no choices.
                extra = getattr(response, "model_extra", None) or {}
                rec.update(error_type="EmptyChoices", retry_kind="empty_choices",
                           error_body=extra.get("error"))
                attempts.append(rec)
                if retry("empty_choices"):
                    continue
                raise RuntimeError(f"provider returned no choices: {extra.get('error')!r}")
            if (limits.get("provider_abort", 0)
                    and getattr(choices[0], "finish_reason", None) == "error"):
                # The provider aborted generation mid-answer (a 200 whose choice
                # finished "error"): a provider fault, retried under the policy;
                # once the retries are spent the call is a transport failure.
                rec.update(error_type="ProviderAbort", retry_kind="provider_abort",
                           completion_tokens=getattr(usage, "completion_tokens", None))
                attempts.append(rec)
                if retry("provider_abort"):
                    continue
                raise RuntimeError("provider aborted generation (finish_reason=error)")
            attempts.append(rec)
            return response

    # -- the decision -----------------------------------------------------

    def _defend(self, view, reason: str) -> PlannerOutput:
        out = PlannerOutput()
        out.roles[int(view.agent_id)] = BlueRole.DEFEND
        self.last_reasoning = reason
        return out

    def plan_one(self, view, max_age=None) -> PlannerOutput:
        """This agent's role and target, from this agent's frozen snapshot.

        Returns a PlannerOutput carrying **only this agent's slot**, matching
        ``RuleBasedBlueRolePlanner.plan_one``: ``_plan_per_agent`` merges the
        slots, and an agent that decided for anyone else would be deciding with
        an authority the decentralized condition does not give it.
        """
        system_prompt = build_system_prompt(self.context)
        user_message = build_user_message(view, self.context, max_age=max_age)
        me = int(view.agent_id)

        try:
            content = self._fetch(system_prompt, user_message)
            if not (content or "").strip():
                # Distinguished from bad JSON on purpose. "The model returned
                # nothing" and "the model returned something invalid" have
                # different causes and different fixes -- the first is almost
                # always a token budget spent on reasoning -- and a diagnosis
                # belongs in the record rather than in whoever reads it next.
                return self._record_fallback(view, "empty_response", None)
            decision = _LocalRoleDecision.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as exc:
            return self._record_fallback(view, "schema_failure", type(exc).__name__)
        except StopRun:
            # Never a fallback: see ReplayMiss / BudgetExhausted.
            raise
        except TransportFailure as exc:
            # The ORIGINAL exception's type, live or replayed, so the two
            # produce the identical event.
            return self._record_fallback(view, "call_failure", exc.error_type)
        except Exception as exc:  # noqa: BLE001 -- network/API failure
            # A failed call must not take a 400-step episode down with it. It
            # must also never be mistaken for a decision, which is what the
            # recorded event is for.
            return self._record_fallback(view, "call_failure", type(exc).__name__)

        role = BlueRole(decision.role)
        out = PlannerOutput()

        if role == BlueRole.NEUTRALIZE:
            believed = set(view.fresh_targets(max_age))
            if decision.target_contact_id not in believed:
                # Pursuing something it does not believe in is the one thing an
                # agent under fog of war must not be able to do -- it would let
                # ground truth back in through the planner.
                return self._record_fallback(
                    view,
                    "invalid_target_contact_id",
                    None,
                    attempted_role=role.value,
                    invalid_value=int(decision.target_contact_id),
                    reasoning=decision.reasoning,
                )
            out.target_red_ids[me] = int(decision.target_contact_id)
        elif role == BlueRole.RECON:
            regions = {
                int(r["region_id"]) for r in (view.coverage_regions() or [])
            }
            if decision.target_region_id not in regions:
                return self._record_fallback(
                    view,
                    "invalid_target_region_id",
                    None,
                    attempted_role=role.value,
                    invalid_value=int(decision.target_region_id),
                    reasoning=decision.reasoning,
                )
            out.recon_target_regions[me] = int(decision.target_region_id)

        out.roles[me] = role
        self.last_reasoning = decision.reasoning
        return out

    def _record_fallback(
        self,
        view,
        kind: str,
        error_type: Optional[str],
        attempted_role: Optional[str] = None,
        invalid_value: Optional[int] = None,
        reasoning: Optional[str] = None,
    ) -> PlannerOutput:
        event = {
            "sim_step": int(view.now),
            "agent_id": int(view.agent_id),
            "kind": kind,
        }
        if error_type is not None:
            event["error_type"] = error_type
        if attempted_role is not None:
            event["attempted_role"] = attempted_role
        if invalid_value is not None:
            event["invalid_value"] = invalid_value
        self.fallback_events.append(event)
        return self._defend(
            view, f"(fallback to DEFEND: {kind}) {reasoning or error_type or ''}".strip()
        )


class PerAgentLLMPlanner:
    """The object ``Simulation`` holds. Owns one ``LocalLLMPlanner`` per blue.

    It exists so the N-planner requirement does not become N-planner surgery
    inside ``Simulation``: ``_plan_per_agent`` keeps handing one object one view
    at a time, which is the interface it already has.

    It holds **no shared decision state**. Planners are created on first sight of
    an agent, dispatch is by ``view.agent_id``, and the only thing aggregated is
    diagnostics -- after each call has already returned.
    """

    def __init__(
        self,
        context: PromptContext,
        model: str = DEFAULT_LOCAL_MODEL,
        client: Optional[OpenAI] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache: Optional[ResponseCache] = None,
        decode: Optional[DecodeConfig] = None,
        build_client_on_demand: bool = True,
        replay_only: bool = False,
        migrate_legacy_v1: bool = False,
        serving: Optional[ServingConfig] = None,
        request_timeout_s: Optional[float] = None,
        request_config: Optional[RequestConfig] = None,
        ledger: Optional[SpendLedger] = None,
    ):
        self.context = context
        self.serving = serving
        self.request_config = request_config
        self.ledger = ledger
        self.request_timeout_s = request_timeout_s
        self.model = str(model)
        self.cache = cache if cache is not None else ResponseCache()
        self.decode = decode or DecodeConfig()
        self.replay_only = bool(replay_only)
        self.migrate_legacy_v1 = bool(migrate_legacy_v1)
        if self.replay_only:
            # EXPLICITLY offline: no client is built, none is accepted, and a
            # cache miss stops the run (ReplayMiss) rather than calling out.
            # "Happens to hit the cache" is not a mode; this is.
            if client is not None:
                raise ValueError("replay_only refuses a client: a replay must not reach a network")
            self.endpoint = normalize_endpoint(base_url or _OPENROUTER_BASE_URL)
        else:
            if client is None and build_client_on_demand:
                # Under a ledger the client must not retry by itself: every
                # attempt is reserved and settled by the planner.
                client = build_client(api_key=api_key, base_url=base_url,
                                      max_retries=0 if ledger is not None else None)
            # The endpoint is part of the cache key. Declared by base_url, else
            # read off the client (the OpenAI client reports its own), else
            # "unspecified" -- a fake client in a test.
            self.endpoint = normalize_endpoint(
                base_url or getattr(client, "base_url", None)
            )
        self.client = client
        self.planners: Dict[int, LocalLLMPlanner] = {}

    _LOCAL_CLASS = LocalLLMPlanner

    def _local_kwargs(self) -> Dict:
        return {}

    def planner_for(self, agent_id: int) -> LocalLLMPlanner:
        agent_id = int(agent_id)
        if agent_id not in self.planners:
            self.planners[agent_id] = self._LOCAL_CLASS(
                agent_id=agent_id,
                context=self.context,
                model=self.model,
                client=self.client,
                cache=self.cache,
                decode=self.decode,
                endpoint=self.endpoint,
                replay_only=self.replay_only,
                migrate_legacy_v1=self.migrate_legacy_v1,
                serving=self.serving,
                request_timeout_s=self.request_timeout_s,
                request_config=self.request_config,
                ledger=self.ledger,
                **self._local_kwargs(),
            )
            if getattr(self, "image_lookup", None) is not None:
                self.planners[agent_id].enable_images(self.image_lookup, self.image_mode)
        return self.planners[agent_id]

    image_mode = IMAGE_MODE_VERSION

    def enable_images(self, lookup, mode: str = IMAGE_MODE_VERSION) -> None:
        """: every per-agent planner, existing and future, attaches images
."""
        if mode not in DECOY_BRIEFS:
            raise ValueError(f"unknown image mode {mode!r}; expected one of {sorted(DECOY_BRIEFS)}")
        self.image_lookup = lookup
        self.image_mode = mode
        for planner in self.planners.values():
            planner.enable_images(lookup, mode)

    def plan_one(self, view, max_age=None) -> PlannerOutput:
        return self.planner_for(view.agent_id).plan_one(view, max_age=max_age)

    # -- diagnostics ------------------------------------------------------

    @property
    def fallback_events(self) -> List[Dict]:
        """Every agent's fallbacks, in (step, agent) order.

        Sorted rather than concatenated in planner-creation order so the record
        reads as an episode timeline and two runs of the same episode produce
        the same list.
        """
        events = [e for p in self.planners.values() for e in p.fallback_events]
        return sorted(events, key=lambda e: (e["sim_step"], e["agent_id"]))

    def diagnostics(self) -> Dict:
        """What must be read before any result from this planner is read."""
        events = self.fallback_events
        by_kind: Dict[str, int] = {}
        for event in events:
            by_kind[event["kind"]] = by_kind.get(event["kind"], 0) + 1
        served: Counter = Counter()
        for p in self.planners.values():
            served.update(p.endpoints_served)
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "mode": "replay_only" if self.replay_only else "live",
            "decode": self.decode.as_dict(),
            "serving": None if self.serving is None else self.serving.as_dict(),
            "request_timeout_s": self.request_timeout_s,
            **({"request_config": self.request_config.as_dict()}
               if self.request_config is not None else {}),
            "endpoints_served": dict(sorted(served.items())),
            "transport_failures": sum(p.transport_failures for p in self.planners.values()),
            "migrated_v1": sum(p.migrated_v1 for p in self.planners.values()),
            "prompt_builder_version": self._LOCAL_CLASS._PROMPT_VERSION,
            "n_planners": len(self.planners),
            "calls": sum(p.calls for p in self.planners.values()),
            "cache_hits": sum(p.cache_hits for p in self.planners.values()),
            "fallbacks": len(events),
            "fallbacks_by_kind": by_kind,
            "events": events,
            "usage": self.cache.usage_totals(),
        }
