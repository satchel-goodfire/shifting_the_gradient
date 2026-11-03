"""The third-party notice (2026-11-06) names real paths, ships the license
it cites, and matches the provenance and changes it states."""

import fnmatch
import glob
import json
import os
import re

from grad_shift.paper.spec import REPO_ROOT

NOTICE = os.path.join(REPO_ROOT, "THIRD_PARTY_NOTICES.md")
QUESTIONS = os.path.join(REPO_ROOT, "data", "trait_questions")
# persona_vectors' own traits, and the traits derived from them (derived ->
# opposite), as THIRD_PARTY_NOTICES.md states them.
UPSTREAM = ("apathetic", "evil", "hallucinating", "humorous", "impolite",
            "optimistic", "sycophantic")
DERIVED = {"empathetic": "apathetic", "good": "evil", "lucid": "hallucinating",
           "pessimistic": "optimistic", "polite": "impolite",
           "serious": "humorous", "sincere": "sycophantic"}


def _read(path):
    with open(os.path.join(REPO_ROOT, path)) as f:
        return f.read()


def _trait(version, trait):
    return json.loads(_read(f"data/trait_questions/{version}/{trait}.json"))


def _notice_paths():
    """Backticked repository paths and glob patterns in the notice."""
    return [p for p in re.findall(r"`([^`]+)`", _read("THIRD_PARTY_NOTICES.md"))
            if "/" in p and "<" not in p]


def _prompts(path):
    data = json.loads(_read(path))
    return data if isinstance(data, list) else data["prompts"]


def _quotes_persona_vectors(path):
    """True if the prompt file contains a positive persona instruction of
    one of persona_vectors' own traits."""
    pos = {i["pos"] for t in UPSTREAM for i in _trait("extract", t)["instruction"]}
    return bool(set(_prompts(path)) & pos)


def test_licenses_ship():
    assert _read("LICENSE").startswith("MIT License")
    apache = _read("LICENSES/Apache-2.0.txt")
    assert "Apache License" in apache and "Version 2.0, January 2004" in apache


def test_every_path_the_notice_names_exists():
    paths = _notice_paths()
    assert paths
    for p in paths:
        if any(c in p for c in "*?["):
            assert glob.glob(os.path.join(REPO_ROOT, p)), p
        else:
            assert os.path.exists(os.path.join(REPO_ROOT, p)), p


def test_listed_prompt_files_are_exactly_those_quoting_persona_vectors():
    """Every prompt file that quotes persona_vectors is covered by a pattern
    in the notice, and every covered file does quote it."""
    patterns = [p for p in _notice_paths() if p.startswith("data/prompt_files/")]
    assert patterns
    files = sorted(os.path.relpath(f, REPO_ROOT) for f in glob.glob(
        os.path.join(REPO_ROOT, "data", "prompt_files", "**", "*.json"),
        recursive=True))
    for f in files:
        listed = any(fnmatch.fnmatchcase(f, p) for p in patterns)
        assert listed == _quotes_persona_vectors(f), f
    # The notice's caveat: only the first five of evil_10samp's prompts.
    ten = _prompts("data/prompt_files/evil_10samp.json")
    pos = {i["pos"] for i in _trait("extract", "evil")["instruction"]}
    assert set(ten[:5]) <= pos and not set(ten[5:]) & pos


def test_listed_prompt_banks_are_persona_instructions():
    """The four banks are the traits' positive persona instructions."""
    for trait in ("evil", "sycophantic", "hallucinating", "optimistic"):
        pos = {i["pos"] for i in _trait("extract", trait)["instruction"]}
        assert set(_prompts(f"data/prompt_files/{trait}.json")) <= pos, trait


def test_trait_files_match_the_stated_changes():
    """Upstream traits: negative instructions emptied except in
    extract/sycophantic.json. Derived traits: the opposite trait's
    questions, empty negatives, their own judge prompt."""
    for version in ("extract", "eval"):
        for trait in UPSTREAM:
            negs = [i["neg"] for i in _trait(version, trait)["instruction"]]
            if (version, trait) == ("extract", "sycophantic"):
                assert all(negs), "extract/sycophantic.json keeps its negatives"
            else:
                assert negs and not any(negs), f"{version}/{trait}.json"
        for derived, opposite in DERIVED.items():
            d, o = _trait(version, derived), _trait(version, opposite)
            assert d["questions"] == o["questions"], f"{version}/{derived}.json"
            assert all(i["pos"] for i in d["instruction"])
            assert not any(i["neg"] for i in d["instruction"])
            assert d["eval_prompt"] != o["eval_prompt"]
    # Where the opposite trait kept its negatives, they are the derived
    # trait's positives (checkable only for sincere <- extract/sycophantic).
    syco = _trait("extract", "sycophantic")["instruction"]
    assert ([i["pos"] for i in _trait("extract", "sincere")["instruction"]]
            == [i["neg"] for i in syco])
    listed = set(os.listdir(os.path.join(QUESTIONS, "extract")))
    assert listed == {f"{t}.json" for t in (*UPSTREAM, *DERIVED,
                                             "pirate", "spanish", "allcaps")}
