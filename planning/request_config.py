"""Which backend is asked, and how: per-backend request configuration.

Two backends serve the same call path (``LocalLLMPlanner._fetch``):

* ``"vllm"`` -- a self-hosted OpenAI-compatible server. A thinking switch
  travels as ``chat_template_kwargs``, a chat-template argument that means
  nothing anywhere else.
* ``"openrouter"`` -- the hosted router. A measured run must reach ONE
  declared upstream endpoint: the provider is pinned (``order``), automatic
  fallback to another provider is disabled, and the provider must support
  every supplied parameter (``require_parameters``) -- otherwise a provider
  that silently ignores ``response_format`` could answer. vLLM-specific flags
  are REFUSED here, not dropped: a DecodeConfig that asks for
  ``enable_thinking`` cannot be sent to OpenRouter at all.

A ``RequestConfig`` is part of the cache key (with the JSON schema actually
sent) and of every record's provenance, so an answer obtained under one
provider pin, schema or accounting setting never answers another. It is
opt-in: a planner without one sends exactly the requests, and computes
exactly the keys, it did before.
"""

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

OPENROUTER = "openrouter"
VLLM = "vllm"
BACKENDS = (OPENROUTER, VLLM)


def backend_of(endpoint: Optional[str]) -> Optional[str]:
    """The backend an endpoint URL belongs to, or None if not recognised."""
    e = str(endpoint or "").lower()
    if "openrouter.ai" in e:
        return OPENROUTER
    if e.startswith("http://localhost") or e.startswith("http://127.0.0.1"):
        return VLLM
    return None


@dataclass(frozen=True)
class ProviderRouting:
    """OpenRouter provider routing for a measured run: one pinned endpoint.

    ``order`` names provider endpoints by OpenRouter slug (e.g.
    ``"siliconflow/fp8"``: the provider and its declared quantization).
    ``allow_fallbacks=False`` means a request the pinned endpoint cannot serve
    FAILS (and is recorded) instead of being served by someone else.
    """

    order: Tuple[str, ...]
    allow_fallbacks: bool = False
    require_parameters: bool = True
    quantizations: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        if not self.order:
            raise ValueError("ProviderRouting needs at least one pinned provider")
        object.__setattr__(self, "order", tuple(str(p) for p in self.order))
        if self.quantizations is not None:
            object.__setattr__(self, "quantizations", tuple(str(q) for q in self.quantizations))

    def body(self) -> Dict:
        out = {"order": list(self.order), "allow_fallbacks": bool(self.allow_fallbacks),
               "require_parameters": bool(self.require_parameters)}
        if self.quantizations is not None:
            out["quantizations"] = list(self.quantizations)
        return out


@dataclass(frozen=True)
class RetryPolicy:
    """How many times each kind of transport/provider fault is retried within
    one logical request. Frozen before scoring and recorded with every
    record; an answer that arrives is never retried, whatever its quality.

    ``timeout`` retries can double-bill (the provider may finish the first
    request); the ledger keeps every failed attempt at its worst case.

    ``provider_abort``: a 200 response
    whose choice finished with ``finish_reason == "error"`` -- the upstream
    provider aborted generation (measured once on InferenceNet: 2 tokens, then
    ``error``). It is a provider fault, not an answer, so it is retried; when
    the retries are spent the call is a transport failure. 0 (the default)
    keeps the earlier behaviour: such a response is returned as an answer and
    judged by the validators. Omitted from ``limits()`` when 0, so every
    earlier cache key is unchanged.
    """

    rate_limit: int = 5
    connection: int = 2
    server_error: int = 2
    empty_choices: int = 2
    timeout: int = 1
    provider_abort: int = 0

    def limits(self) -> Dict[str, int]:
        out = {"rate_limit": int(self.rate_limit), "connection": int(self.connection),
               "server_error": int(self.server_error),
               "empty_choices": int(self.empty_choices), "timeout": int(self.timeout)}
        if int(self.provider_abort):
            out["provider_abort"] = int(self.provider_abort)
        return out


@dataclass(frozen=True)
class RequestConfig:
    """Backend-specific request settings, in the cache key and provenance.

    ``usage_accounting`` asks OpenRouter to return the billed cost with each
    response (``usage: {"include": true}``), which the spend ledger settles
    against. ``strict_schema`` is the ``json_schema.strict`` flag; it is a
    field so that it is visibly part of the address.
    """

    backend: str
    routing: Optional[ProviderRouting] = None
    usage_accounting: bool = False
    strict_schema: bool = True
    retry: Optional[RetryPolicy] = None

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r}; expected one of {BACKENDS}")
        if self.backend == VLLM and (self.routing is not None or self.usage_accounting):
            raise ValueError("provider routing and usage accounting are OpenRouter request "
                             "fields; a vLLM request config must not carry them")

    def check(self, decode, endpoint) -> None:
        """Refuse a configuration that would send the wrong backend's fields."""
        found = backend_of(endpoint)
        if found is not None and found != self.backend:
            raise ValueError(f"request config is for {self.backend!r} but the endpoint "
                             f"{endpoint!r} is {found!r}")
        if self.backend == VLLM and getattr(decode, "reasoning", None) is not None:
            raise ValueError("DecodeConfig.reasoning is OpenRouter's reasoning object; it "
                             "is refused for vLLM (use enable_thinking there)")
        if self.backend == OPENROUTER:
            if getattr(decode, "enable_thinking", None) is not None:
                raise ValueError(
                    "enable_thinking is a self-hosted server's chat-template switch "
                    "(chat_template_kwargs); it is refused for OpenRouter rather than "
                    "sent blindly. Use a DecodeConfig without it.")
            if self.routing is None:
                raise ValueError("an OpenRouter request config must pin a provider "
                                 "(ProviderRouting): measured runs never auto-route")

    def extra_body(self, decode) -> Optional[Dict]:
        """The non-standard request body fields for this backend."""
        if self.backend == VLLM:
            return decode.extra_body()
        body = {"provider": self.routing.body()}
        if self.usage_accounting:
            body["usage"] = {"include": True}
        if getattr(decode, "reasoning_effort", None):
            body["reasoning"] = {"effort": decode.reasoning_effort}
        if getattr(decode, "reasoning", None):
            body["reasoning"] = json.loads(decode.reasoning)
        return body

    def as_dict(self) -> Dict:
        out = {"backend": self.backend,
               "routing": None if self.routing is None else self.routing.body(),
               "usage_accounting": bool(self.usage_accounting),
               "strict_schema": bool(self.strict_schema)}
        if self.retry is not None:
            # Absent when unset, so every earlier key is unchanged.
            out["retry"] = self.retry.limits()
        return out
