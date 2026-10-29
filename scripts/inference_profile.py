"""Inference profiles: one frozen, versioned description of how a model is asked.

Added for the multi-candidate model-selection study. A profile names everything that decides what a model call is,
and nothing that depends on the episode:

* backend, model id, pinned provider endpoint (OpenRouter tag, e.g.
  ``"xiaomi/fp8"``; fallbacks off, every supplied parameter required);
* declared provenance the endpoint does expose (quantization as listed) and
  what it does not (``revision``/``precision`` left ``"not_exposed"`` -- never
  invented);
* prices from the endpoint listing, with its retrieval time, including the
  reasoning and cache categories where listed;
* decoding: temperature, OpenRouter's reasoning object, and an output
  allowance PER INTERFACE (defender, commander, intruder) -- never one global
  cap by accident;
* request timeout and the frozen retry policy.

``profile_hash`` is part of every execution identity; two runs with different
hashes are different configurations. The interface versions used by new
experiments are frozen here too (``NEW_EXPERIMENT_INTERFACES``).
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

INTERFACES = ("defender", "commander", "intruder")

# The interface versions every reported episode uses: the defender bridge, the
# intruder prompt and the commander prompt, over the three-task model. The
# prompts print perception.sensing's default sensor geometry, so a runner must
# check the episode uses it (``check_heading_prompt_geometry``).
DESIGN = "v2.3mb"
INTERFACE_VERSIONS = {
    "bridge_version": "m8b.7",
    "task_set": "three_task",
    "intruder_prompt": "intruder.4",
    "commander_prompt": "commander.10",
}


def check_heading_prompt_geometry(sim) -> None:
    """Refuse an episode whose sensor geometry differs from the one the
    prompts state (planning/prompt_builder.SENSOR_GEOMETRY_TEXT:
    perception.sensing's default 60-degree wedge, 35 units)."""
    import math

    from perception import sensing

    half = math.degrees(float(sim.fov_half_angle_rad))
    if (abs(half - sensing.DEFAULT_FOV_HALF_ANGLE_DEG) > 1e-9
            or abs(float(sim.fov_range) - sensing.DEFAULT_FOV_RANGE) > 1e-9):
        raise ValueError(
            f"the prompts state a {2 * sensing.DEFAULT_FOV_HALF_ANGLE_DEG:g}-degree, "
            f"{sensing.DEFAULT_FOV_RANGE:g}-unit sensor, but this episode uses half-angle "
            f"{half:g} deg, range {float(sim.fov_range):g}; refusing")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LOCAL_BASE_URL = "http://localhost:8000/v1"


@dataclass(frozen=True)
class InferenceProfile:
    profile_id: str
    backend: str  # "openrouter" | "vllm"
    model: str
    max_tokens: Dict[str, int]
    temperature: Optional[float] = 0.0  # None = not sent
    reasoning: Optional[str] = None  # OpenRouter reasoning object, canonical JSON
    provider: Optional[str] = None  # OpenRouter endpoint tag (pinned)
    price: Dict[str, float] = field(default_factory=dict)  # USD per token
    price_source: str = ""
    provenance: Dict[str, str] = field(default_factory=dict)
    timeout_s: float = 120.0
    retry: Dict[str, int] = field(default_factory=lambda: {
        "rate_limit": 5, "connection": 2, "server_error": 2, "empty_choices": 2, "timeout": 1,
        "provider_abort": 2})
    serving: Optional[Dict] = None  # vLLM ServingConfig fields

    def __post_init__(self):
        if self.backend not in ("openrouter", "vllm"):
            raise ValueError(f"unknown backend {self.backend!r}")
        missing = set(INTERFACES) - set(self.max_tokens)
        if missing:
            raise ValueError(f"profile {self.profile_id}: no output allowance for {sorted(missing)}")
        if self.backend == "openrouter":
            if not self.provider:
                raise ValueError(f"profile {self.profile_id}: OpenRouter needs a pinned provider")
            for k in ("prompt", "completion"):
                if k not in self.price:
                    raise ValueError(f"profile {self.profile_id}: price lacks {k!r}")
        if self.reasoning is not None:
            obj = json.loads(self.reasoning)
            object.__setattr__(self, "reasoning", json.dumps(obj, sort_keys=True))

    # -- identity ------------------------------------------------------------

    def as_dict(self) -> Dict:
        return json.loads(json.dumps(asdict(self), sort_keys=True))

    def profile_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: Dict) -> "InferenceProfile":
        return cls(**d)

    # -- request objects -------------------------------------------------------

    def endpoint(self) -> str:
        return OPENROUTER_BASE_URL if self.backend == "openrouter" else LOCAL_BASE_URL

    def decode(self, interface: str):
        from planning.local_llm import DecodeConfig

        if self.backend == "vllm":
            return DecodeConfig(temperature=self.temperature,
                                max_tokens=int(self.max_tokens[interface]),
                                enable_thinking=False)
        return DecodeConfig(temperature=self.temperature,
                            max_tokens=int(self.max_tokens[interface]),
                            reasoning=self.reasoning)

    def request_config(self):
        from planning.request_config import (OPENROUTER, VLLM, ProviderRouting, RequestConfig,
                                             RetryPolicy)

        retry = RetryPolicy(**self.retry)
        if self.backend == "vllm":
            return RequestConfig(VLLM, retry=retry)
        return RequestConfig(OPENROUTER, routing=ProviderRouting((self.provider,)),
                             usage_accounting=True, retry=retry)

    def ledger_price(self):
        """Worst-case price per token for ledger reservations: the larger of
        the prompt and cache-write prices for input, and of the completion and
        reasoning prices for output. Settlement uses the billed cost."""
        from planning.spend import Price

        p = self.price
        return Price(max(p["prompt"], p.get("cache_write", 0.0) or 0.0),
                     max(p["completion"], p.get("reasoning", 0.0) or 0.0),
                     source=f"{self.profile_id} ({self.price_source})")

    def serving_config(self):
        if self.serving is None:
            return None
        from planning.response_cache import ServingConfig

        return ServingConfig(**self.serving)

    def planner_kwargs(self, interface: str, cache, client=None, ledger=None,
                       replay_only: bool = False) -> Dict:
        """LocalLLMPlanner keyword arguments for one interface."""
        kw = dict(model=self.model, client=client, cache=cache, replay_only=replay_only,
                  decode=self.decode(interface), endpoint=self.endpoint(),
                  request_config=self.request_config(), request_timeout_s=self.timeout_s)
        if self.backend == "openrouter":
            kw["ledger"] = ledger
        else:
            kw["serving"] = self.serving_config()
        return kw

    def build_client(self):
        import os

        from planning.local_llm import build_client

        if self.backend == "vllm":
            return build_client(api_key=os.environ.get("LOCAL_LLM_API_KEY"),
                                base_url=LOCAL_BASE_URL, timeout=self.timeout_s, max_retries=0)
        return build_client(base_url=OPENROUTER_BASE_URL, timeout=self.timeout_s, max_retries=0)


def load_profiles(path: str) -> Dict[str, InferenceProfile]:
    with open(path) as f:
        data = json.load(f)
    return {d["profile_id"]: InferenceProfile.from_dict(d) for d in data["profiles"]}


def code_version(repo_dir: Optional[str] = None) -> Dict:
    """HEAD commit and whether tracked files differ from it (untracked files are not
    code)."""
    import os
    import subprocess

    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def git(*args):
        return subprocess.run(["git", "-C", repo_dir, *args], capture_output=True,
                              text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
