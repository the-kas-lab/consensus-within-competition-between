"""non-hostile contacts and contact images.

non-hostile contacts (birds, civilian drones) live in the red state arrays after the real
reds. Off by default; when on, the real reds and blues must spawn exactly as
without non-hostile contacts, non-hostile contacts never breach or score, contact destroys a blue and the
non-hostile contact, the win condition counts real reds, and the intruder team never sees a
non-hostile contact. The image path must leave every text-only request and key unchanged.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.local_llm import (  # noqa: E402
    DECOY_BRIEF_V1,
    IMAGE_MODE_VERSION,
    LocalLLMPlanner,
    with_decoy_brief,
)
from planning.response_cache import ResponseCache  # noqa: E402
from scripts import run_competing_teams as rct  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = {4: os.path.join(ROOT, "assets/contact_photos/bird_1.jpeg"),
       5: os.path.join(ROOT, "assets/contact_photos/civilian_1.jpeg"),
       6: os.path.join(ROOT, "assets/contact_photos/civilian_2.jpeg")}
DECOYS = {"birds": 1, "civilians": 2, "images": IMG}


def _entry(arm="cbaa", seed=4001, decoys=None):
    e = {"arm": arm, "reds": 4, "family": "multi_bearing", "condition": "REF",
         "attacked": False, "seed": seed, "design": "v2.3mb"}
    if decoys is not None:
        e.update(decoys=decoys, decoy_tag="b1c2")
    return e


def _spawned(e):
    sim, _ = rct.build(e, "rule")
    sim.setup_agents()  # build + setup back to back: the legacy global RNG is seeded per build
    return sim


@pytest.mark.parametrize("seed", [4001, 4003])
def test_real_reds_and_blues_spawn_as_without_decoys(seed):
    plain = _spawned(_entry(seed=seed))
    r0, b0 = plain.red_state.copy(), plain.x_current.copy()
    with_d = _spawned(_entry(seed=seed, decoys=DECOYS))
    nx = with_d.red_nx
    assert with_d.num_real_reds == 4 and with_d.num_reds == 7
    assert np.array_equal(r0, with_d.red_state[: 4 * nx])
    assert np.array_equal(b0, with_d.x_current)
    assert with_d.red_kinds == ["drone"] * 4 + ["bird", "civilian", "civilian"]


def test_off_by_default_leaves_config_unchanged():
    sim, _ = rct.build(_entry(), "rule")
    assert "decoys" not in sim._init_args
    assert sim.num_real_reds == sim.num_reds and not sim.decoy_indices
    assert "decoys" not in rct.execution_identity("rule")
    assert "decoys" in rct.execution_identity("rule", decoys=True)
    assert rct.episode_id(_entry()) == rct.episode_id(dict(_entry(), decoy_tag=None))


def test_decoy_contact_destroys_both_without_score():
    sim = _spawned(_entry(decoys=DECOYS))
    sim.belief_stores = {}
    sim.resolution_info = "legacy"
    nx = sim.red_nx
    j = 5
    sim.red_state[j * nx : j * nx + 2] = sim.x_current[0:2]  # non-hostile on blue 0
    breached = sim._resolve_active_reds(step=1)
    assert not breached
    assert sim.red_outcomes[j] == "decoy_destroyed" and sim.blue_disabled[0]
    assert sim.num_neutralized == 0 and sim.blue_score == 0.0
    assert sim.decoy_events[-1]["event"] == "destroyed"


def test_decoy_inside_region_is_not_a_breach():
    sim = _spawned(_entry(decoys=DECOYS))
    nx = sim.red_nx
    sim.red_state[4 * nx : 4 * nx + 2] = sim.defense_center  # (cannot happen by script)
    assert not sim._resolve_active_reds(step=1)
    assert not sim.touchdown


def test_episode_runs_and_intruders_never_see_decoys():
    from scripts.episode_builder import _Quiet

    sim, _ = rct.build(_entry(arm="commander", decoys=DECOYS), "rule")
    with _Quiet():
        sim.run()
    o = sim.episode_outcome()
    assert len(o["red_outcomes"]) == 4
    assert set(o["decoys"]["kinds"]) == {4, 5, 6}
    assert rct.terminal_problem({"outcome": o, "steps_run": len(sim.history_blues) - 1,
                                 "horizon_steps": sim.sim_steps}) is None
    assert {x["red"] for x in sim.intruder_log if "red" in x} <= {0, 1, 2, 3}
    H = np.asarray(sim.history_red)
    for t in range(len(H)):
        for j in (4, 5, 6):
            assert not sim._point_in_polygon(H[t, j * sim.red_nx : j * sim.red_nx + 2],
                                             sim.defended_shape_points)


# -- the image call path -------------------------------------------------------


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)
        self.finish_reason = "stop"


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = None
        self.model = "fake"
        self.id = "x"


class _Capture:
    def __init__(self):
        self.kwargs = []
        self.chat = type("c", (), {})()
        self.chat.completions = self

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        return _Resp('{"ok": true}')


def _planner(client, cache):
    return LocalLLMPlanner(0, context=None, client=client, cache=cache, endpoint="https://fake/v1")


def test_text_only_calls_are_unchanged_and_images_change_the_key(tmp_path):
    client = _Capture()
    p = _planner(client, ResponseCache(path=str(tmp_path / "c")))
    p._fetch("sys", "user")
    key_text = p.last_key
    assert client.kwargs[-1]["messages"][1]["content"] == "user"
    p._fetch("sys", "user", images=[])
    assert p.last_key == key_text  # an empty image list is a text-only call
    p._fetch("sys", "user", images=[(5, IMG[5])])
    assert p.last_key != key_text
    content = client.kwargs[-1]["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "user"}
    assert content[1]["text"] == "image of contact_id 5:"
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[2]["image_url"]["detail"] == "low"


def test_image_mode_versions_the_prompt_and_adds_the_brief():
    p = _planner(_Capture(), ResponseCache())
    v = p._PROMPT_VERSION
    assert p._image_request("anything", [1]) == ("anything", None)
    p.enable_images(IMG.get)
    assert p._PROMPT_VERSION == f"{v}+{IMAGE_MODE_VERSION}"
    brief = ("Reds can change direction at any time, and reds your team has not detected may "
             "exist.")
    system, images = p._image_request("A. " + brief + " B.", [6, 4, 1])
    assert DECOY_BRIEF_V1 in system and images == [(4, IMG[4]), (6, IMG[6])]
    with pytest.raises(ValueError):
        with_decoy_brief("no brief here")


def test_cache_record_holds_hashes_not_base64(tmp_path):
    from planning.request_config import OPENROUTER, ProviderRouting, RequestConfig

    cache = ResponseCache(path=str(tmp_path / "c"))
    p = LocalLLMPlanner(0, context=None, client=_Capture(), cache=cache,
                        endpoint="https://fake/v1", request_config=RequestConfig(OPENROUTER, routing=ProviderRouting(("fake",))))
    p._fetch("sys", "user", images=[(4, IMG[4])])
    rec = cache.get_record(p.last_key)
    blob = json.dumps(rec)
    assert "base64" not in blob
    assert rec["provenance"]["request"]["images"][0]["contact_id"] == 4
    assert len(rec["provenance"]["request"]["images"][0]["sha256"]) == 64


# --: the warned image mode (img.1w), opt-in ----------------------------

_BRIEF = "Reds can change direction at any time, and reds your team has not detected may exist."


def test_default_image_mode_is_img1_and_its_brief_is_unchanged():
    from planning.local_llm import DECOY_BRIEFS

    assert IMAGE_MODE_VERSION == "img.1" and DECOY_BRIEFS["img.1"] == DECOY_BRIEF_V1
    p = _planner(_Capture(), ResponseCache())
    p.enable_images(IMG.get)
    assert p.image_mode == "img.1"
    assert with_decoy_brief("A. " + _BRIEF + " B.") == with_decoy_brief("A. " + _BRIEF + " B.", "img.1")


def test_warned_mode_adds_exactly_one_sentence_after_the_id_sentence():
    from planning.local_llm import DECOY_BRIEFS, DECOY_WARNING_V1, WARNED_IMAGE_MODE_VERSION

    assert WARNED_IMAGE_MODE_VERSION == "img.1w"
    assert DECOY_WARNING_V1 == "Some red intruders may be disguised as birds."
    warned = DECOY_BRIEFS["img.1w"]
    assert warned.replace(" " + DECOY_WARNING_V1, "", 1) == DECOY_BRIEF_V1
    assert ("A contact's id does not tell what it is. Some red intruders may be disguised as birds. "
            "One image of each contact") in warned
    system = with_decoy_brief("A. " + _BRIEF + " B.", "img.1w")
    assert system == "A. " + _BRIEF + warned + " B."


def test_warned_mode_versions_the_prompt_and_changes_the_key(tmp_path):
    keys = {}
    for mode in ("img.1", "img.1w"):
        client = _Capture()
        p = _planner(client, ResponseCache(path=str(tmp_path / mode)))
        v = p._PROMPT_VERSION
        p.enable_images(IMG.get, mode)
        assert p._PROMPT_VERSION == f"{v}+{mode}"
        system, images = p._image_request("A. " + _BRIEF + " B.", [4])
        p._fetch(system, "user", images=images)
        keys[mode] = p.last_key
        assert ("disguised as birds" in client.kwargs[-1]["messages"][0]["content"]) == (mode == "img.1w")
    assert keys["img.1"] != keys["img.1w"]


def test_unknown_or_conflicting_image_mode_is_refused():
    p = _planner(_Capture(), ResponseCache())
    with pytest.raises(ValueError):
        p.enable_images(IMG.get, "img.9")
    p.enable_images(IMG.get, "img.1w")
    p.enable_images(IMG.get, "img.1w")  # idempotent in the same mode
    with pytest.raises(ValueError):
        p.enable_images(IMG.get, "img.1")
    with pytest.raises(ValueError):
        with_decoy_brief("A. " + _BRIEF, "img.9")


def test_per_agent_source_passes_the_mode_to_existing_and_new_planners():
    from planning.local_llm import PerAgentLLMPlanner

    src = PerAgentLLMPlanner(context=None, client=_Capture(), cache=ResponseCache(),
                             base_url="https://fake/v1")
    first = src.planner_for(0)
    src.enable_images(IMG.get, "img.1w")
    later = src.planner_for(3)
    assert first.image_mode == later.image_mode == "img.1w"
    assert first._PROMPT_VERSION.endswith("+img.1w") and later._PROMPT_VERSION.endswith("+img.1w")


def test_image_mode_absent_leaves_ids_and_identities_unchanged():
    e = _entry(decoys=DECOYS)
    assert rct.episode_id(e) == rct.episode_id(dict(e, image_mode=None))
    assert rct.episode_id(dict(e, image_mode="img.1w")) != rct.episode_id(e)
    base = rct.execution_identity("rule", decoys=True)
    assert rct.execution_identity("rule", decoys=True, image_modes=["img.1"]) == base
    warned = rct.execution_identity("rule", decoys=True, image_modes=["img.1", "img.1w"])
    assert "image_modes" not in base["decoys"]
    assert "disguised as birds" in warned["decoys"]["image_modes"]["img.1w"]
    assert warned["identity_hash"] != base["identity_hash"]
