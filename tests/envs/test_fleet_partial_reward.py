"""Tests for `FleetTaskEnv._parse_partial_reward`.

The canonical 1-epoch run had BU=0/48 and CU=0/44 PASSes. Every single
rollout had visible partial competence (BU 1/3 apps, CU 3/5 apps), but
the verifier aggregates with binary `min()`, collapsing all of them to
reward=0. Workflow synthesis wf_1c2d5247 attributed the entire zero
gradient signal to this collapse. This module verifies the parser turns
the per-app stdout into a real fractional reward, including:

  - Multi-app `verify_multi_app_<uuid>` aggregator output (BU+CU).
  - Trivial-pass exclusion: an app that passed only because no changes were
    expected (e.g. medora on a Lifeline-only task) must NOT inflate the
    fractional score. It is dropped from numerator AND denominator.
  - Fallback to single-accumulator parsing for legacy verifiers.

Run:
    pytest tests/envs/test_fleet_partial_reward.py -v
"""

from __future__ import annotations

import pytest

from envs.fleet_env.task_env import FleetTaskEnv


# Real BU rollout from `task_eoxaxej50ghe_n_1781527454806_8dgebl230`
# (session 4e4d0d80 in the canonical-v3 run). Trimmed accumulator contents
# but kept structure verbatim.
BU_REAL_MULTI_APP_STDOUT = """<<< VERIFY_outlook <<<
>>> ERROR_ACCUMULATOR >>>
["[X] No sent email found with subject '13/11/2025 Dr appointment'", "[X] 'Flagged Email' task list not found", '[X] No todo task found for the flagged email']
<<< ERROR_ACCUMULATOR <<<
>>> SUCCESS_ACCUMULATOR >>>
['[C] Expected changes were found in the database diff']
<<< SUCCESS_ACCUMULATOR <<<
>>> VERIFY_outlook >>>
App outlook: 0
<<< VERIFY_lifeline <<<
>>> ERROR_ACCUMULATOR >>>
['[X] No new appointment found for patient 9001', '[X] Could not find new follow-up appointment for diff validation']
<<< ERROR_ACCUMULATOR <<<
>>> SUCCESS_ACCUMULATOR >>>
[]
<<< SUCCESS_ACCUMULATOR <<<
>>> VERIFY_lifeline >>>
App lifeline: 0
<<< VERIFY_medora <<<
>>> SUCCESS_ACCUMULATOR >>>
['[C] Verified no unexpected changes in the Medora database']
<<< SUCCESS_ACCUMULATOR <<<
>>> VERIFY_medora >>>
App medora: 1
Combined result: 1/3 apps passed"""


class TestMultiAppParsing:
    def test_real_bu_rollout_excludes_trivial_medora_pass(self):
        # Medora's pass is "no unexpected changes" — trivial. After exclusion,
        # 2 real apps remain (outlook, lifeline), both 0/2 → reward 0.0.
        score = FleetTaskEnv._parse_partial_reward(BU_REAL_MULTI_APP_STDOUT)
        assert score == pytest.approx(0.0)

    def test_one_real_app_pass_yields_half(self):
        stdout = """<<< VERIFY_a <<<
>>> ERROR_ACCUMULATOR >>>
['[X] failed']
<<< ERROR_ACCUMULATOR <<<
>>> SUCCESS_ACCUMULATOR >>>
[]
<<< SUCCESS_ACCUMULATOR <<<
>>> VERIFY_a >>>
App a: 0
<<< VERIFY_b <<<
>>> SUCCESS_ACCUMULATOR >>>
['[C] Did the work']
<<< SUCCESS_ACCUMULATOR <<<
>>> VERIFY_b >>>
App b: 1
Combined result: 1/2 apps passed"""
        score = FleetTaskEnv._parse_partial_reward(stdout)
        assert score == pytest.approx(0.5)

    def test_all_real_apps_pass_yields_one(self):
        stdout = """<<< VERIFY_a <<<
>>> SUCCESS_ACCUMULATOR >>>
['[C] Did the a work']
<<< SUCCESS_ACCUMULATOR <<<
>>> VERIFY_a >>>
App a: 1
<<< VERIFY_b <<<
>>> SUCCESS_ACCUMULATOR >>>
['[C] Did the b work']
<<< SUCCESS_ACCUMULATOR <<<
>>> VERIFY_b >>>
App b: 1
Combined result: 2/2 apps passed"""
        score = FleetTaskEnv._parse_partial_reward(stdout)
        assert score == pytest.approx(1.0)

    def test_all_apps_trivial_pass_yields_none(self):
        # Every app passed only because no changes were required. There are
        # zero "real" checks to grade — return None so the caller can decide
        # whether to fall back to the binary score or treat as no signal.
        stdout = """<<< VERIFY_a <<<
>>> SUCCESS_ACCUMULATOR >>>
['[C] Verified no unexpected changes in the a database']
<<< SUCCESS_ACCUMULATOR <<<
>>> VERIFY_a >>>
App a: 1
<<< VERIFY_b <<<
>>> SUCCESS_ACCUMULATOR >>>
['[C] No filesystem changes required for this task']
<<< SUCCESS_ACCUMULATOR <<<
>>> VERIFY_b >>>
App b: 1
Combined result: 2/2 apps passed"""
        score = FleetTaskEnv._parse_partial_reward(stdout)
        assert score is None

    def test_three_app_mix_with_one_trivial_pass_yields_one_half(self):
        # Real BU verifier shape: one trivial (medora "no unexpected changes"),
        # one real-pass (outlook completed), one real-fail (lifeline missed).
        # Excluding medora: 1/2 = 0.5.
        stdout = """<<< VERIFY_outlook <<<
>>> SUCCESS_ACCUMULATOR >>>
['[C] Email sent to spouse', '[C] Email is flagged']
<<< SUCCESS_ACCUMULATOR <<<
>>> VERIFY_outlook >>>
App outlook: 1
<<< VERIFY_lifeline <<<
>>> ERROR_ACCUMULATOR >>>
['[X] No appointment found']
<<< ERROR_ACCUMULATOR <<<
>>> SUCCESS_ACCUMULATOR >>>
[]
<<< SUCCESS_ACCUMULATOR <<<
>>> VERIFY_lifeline >>>
App lifeline: 0
<<< VERIFY_medora <<<
>>> SUCCESS_ACCUMULATOR >>>
['[C] Verified no unexpected changes in the Medora database']
<<< SUCCESS_ACCUMULATOR <<<
>>> VERIFY_medora >>>
App medora: 1
Combined result: 2/3 apps passed"""
        score = FleetTaskEnv._parse_partial_reward(stdout)
        assert score == pytest.approx(0.5)


class TestSingleAccumulatorFallback:
    def test_legacy_single_accumulator_still_works(self):
        # No `App X: N` lines → multi-app path returns None, fall back to the
        # original single-accumulator parser.
        stdout = """Some preamble
>>> ERROR_ACCUMULATOR >>>
['[X] one', '[X] two']
<<< ERROR_ACCUMULATOR <<<
>>> SUCCESS_ACCUMULATOR >>>
['[C] three']
<<< SUCCESS_ACCUMULATOR <<<
done."""
        score = FleetTaskEnv._parse_partial_reward(stdout)
        # 1 success / 3 total
        assert score == pytest.approx(1 / 3)


class TestNoAccumulators:
    def test_empty_stdout_returns_none(self):
        assert FleetTaskEnv._parse_partial_reward("") is None

    def test_random_text_returns_none(self):
        assert FleetTaskEnv._parse_partial_reward("Verifier ran cleanly. result: 0.0") is None
