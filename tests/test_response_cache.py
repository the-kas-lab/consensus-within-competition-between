"""The response cache: what it must key on, and what it must never collide.

A cache for paid calls has exactly one unacceptable failure -- answering a
question it was never asked -- so most of these tests are collision tests.
"""

import json
import os

from planning.response_cache import ResponseCache, cache_key


BASE = dict(
    prompt_builder_version="1",
    system_prompt="system",
    user_message="user",
    model="openai/gpt-oss-20b",
    endpoint="https://openrouter.ai/api/v1",
    decode_config={"temperature": 0.0},
)


def _key(**overrides):
    params = dict(BASE)
    params.update(overrides)
    return cache_key(**params)


# --- what must change the key ---------------------------------------------

def test_every_field_is_load_bearing_in_the_key():
    """Each field, changed alone, must move the address.

    Written as one test over all six because the failure it guards is
    'somebody added a field to the payload and forgot the key', and a per-field
    test would not notice a *missing* field either.
    """
    base = _key()
    assert _key(prompt_builder_version="2") != base
    assert _key(system_prompt="other") != base
    assert _key(user_message="other") != base
    assert _key(model="qwen/qwen3-30b-a3b-instruct-2507") != base
    assert _key(decode_config={"temperature": 0.7}) != base
    # Key schema 2: who was asked. OpenRouter and a local vLLM serving the same
    # model id are two different questions and must not share an entry.
    assert _key(endpoint="http://localhost:8000/v1") != base


def test_the_prompt_builder_version_separates_otherwise_identical_prompts():
    """The specific failure the version exists to close: an edited builder that
    happens to produce the same string for *this* view would still be a
    different builder, and its entries must not be reused."""
    assert _key(prompt_builder_version="1") != _key(prompt_builder_version="2")


def test_identical_inputs_give_an_identical_key():
    assert _key() == _key()


def test_decode_config_key_order_does_not_matter():
    """Canonicalized, not stringified: two equal configs written in different
    orders are the same question and must hit the same entry."""
    a = _key(decode_config={"temperature": 0.0, "max_tokens": 256})
    b = _key(decode_config={"max_tokens": 256, "temperature": 0.0})
    assert a == b


def test_a_multimodal_user_message_is_hashable():
    """Vision planners send a content *list*, not a string. The key must not
    care -- and two different image payloads must not collide."""
    one = _key(user_message=[{"type": "text", "text": "a"}])
    two = _key(user_message=[{"type": "text", "text": "b"}])
    assert one != two


# --- store semantics -------------------------------------------------------

def test_miss_then_hit_in_memory():
    cache = ResponseCache()
    key = _key()
    assert cache.get(key) is None
    cache.put(key, '{"role": "defend"}')
    assert cache.get(key) == '{"role": "defend"}'
    assert cache.stats() == {"hits": 1, "misses": 1, "writes": 1}


def test_entries_survive_a_new_cache_object_on_the_same_directory(tmp_path):
    """The property the whole design rests on: a *later run* pays nothing."""
    key = _key()
    ResponseCache(path=str(tmp_path)).put(key, "cached-content")

    fresh = ResponseCache(path=str(tmp_path))
    assert fresh.get(key) == "cached-content"
    assert fresh.stats()["hits"] == 1


def test_a_disabled_cache_never_hits_and_never_writes(tmp_path):
    """Opting out must be total -- otherwise 'no cache' runs would still be
    served stale answers from an earlier run's directory."""
    cache = ResponseCache(path=str(tmp_path), enabled=False)
    key = _key()
    cache.put(key, "content")
    assert cache.get(key) is None
    assert cache.stats()["writes"] == 0
    assert os.listdir(tmp_path) == []


def test_a_corrupt_entry_is_a_miss_not_a_crash(tmp_path):
    """A cache that can kill a 400-step episode is worse than no cache."""
    cache = ResponseCache(path=str(tmp_path))
    key = _key()
    with open(os.path.join(str(tmp_path), f"{key}.json"), "w") as f:
        f.write("{not json")

    assert cache.get(key) is None
    assert cache.stats()["misses"] == 1


def test_writes_are_atomic_leaving_no_partial_files(tmp_path):
    cache = ResponseCache(path=str(tmp_path))
    cache.put(_key(), "content")
    names = os.listdir(str(tmp_path))
    assert names == [f"{_key()}.json"], names
    assert ".tmp" not in "".join(names)


def test_stored_record_carries_usage_for_spend_reporting(tmp_path):
    cache = ResponseCache(path=str(tmp_path))
    key = _key()
    cache.put(key, "content", {"prompt_tokens": 1100, "completion_tokens": 90})

    with open(os.path.join(str(tmp_path), f"{key}.json")) as f:
        record = json.load(f)
    assert record["usage"]["prompt_tokens"] == 1100

    totals = cache.usage_totals()
    assert totals == {"prompt_tokens": 1100, "completion_tokens": 90, "calls": 1}


def test_usage_totals_report_this_runs_spend_not_the_directorys(tmp_path):
    """A run served entirely from disk spent nothing, and must say so. Reporting
    the directory's totals would inflate every replay's cost report."""
    ResponseCache(path=str(tmp_path)).put(
        _key(), "content", {"prompt_tokens": 1100, "completion_tokens": 90}
    )

    replay = ResponseCache(path=str(tmp_path))
    assert replay.get(_key()) == "content"
    assert replay.usage_totals() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "calls": 0,
    }
