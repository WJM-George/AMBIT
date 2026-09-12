from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.diagnostics.audit_vae_mixture_state_identifiability import (
    DRAW_COUNT,
    _direction_group_report,
    _semantic_branch_rows,
    _shared_posterior_draws,
)
from scripts.t2a.eval.summarize_vae_mixture_state_identifiability import (
    summarize as summarize_mixture,
)
from scripts.t2a.eval.summarize_vae_causal_source_presence import (
    summarize as summarize_causal,
)
from scripts.t2a.eval.diagnostics.adjudicate_vae_causal_source_presence import (
    _assignment_row,
    _source_causal_report,
)
from scripts.vae.train.overfit_spatial_family_capacity import (
    _crop_starts,
    _expanded_vae_config,
    _loss_summary,
    _training_schedule,
)
from scripts.vae.eval.summarize_spatial_family_capacity import (
    summarize as summarize_capacity,
)
from stable_audio_tools.training.autoencoders import cosine_schedule_value


class SharedPosteriorDrawTest(unittest.TestCase):
    def test_one_epsilon_is_shared_across_target_states(self) -> None:
        mean = torch.zeros(3, 2, 4)
        mean[0] = 2.0
        mean[1] = 0.5
        stdev = torch.ones_like(mean)
        stdev[0] = 0.7
        stdev[1] = 0.2
        stdev[2] = 0.7
        draws, epsilon = _shared_posterior_draws(
            mean, stdev, draw_count=DRAW_COUNT, seed=123
        )
        expected = (mean[0] - mean[1]).unsqueeze(0) + 0.5 * epsilon
        self.assertTrue(torch.allclose(draws[:, 0] - draws[:, 1], expected))
        # Equal posterior scales cancel the stochastic term exactly.
        self.assertTrue(
            torch.allclose(
                draws[:, 0] - draws[:, 2],
                (mean[0] - mean[2]).unsqueeze(0).expand(DRAW_COUNT, -1, -1),
            )
        )


class LatentDirectionGateTest(unittest.TestCase):
    def test_stable_orthogonal_sources_pass(self) -> None:
        means = torch.zeros(2, 4, 2)
        means[0, 0] = 1.0
        means[1, 1] = 1.0
        draws = means.unsqueeze(0).repeat(DRAW_COUNT, 1, 1, 1)
        draws[:, 0, 2] = 0.02
        draws[:, 1, 3] = -0.02
        report = _direction_group_report(
            draws,
            means,
            source_ids=["source_0", "source_1"],
            group="full",
            n_w=2,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(all(row["status"] == "PASS" for row in report["sources"]))

    def test_aliased_source_means_block(self) -> None:
        means = torch.zeros(2, 4, 2)
        means[:, 0] = 1.0
        draws = means.unsqueeze(0).repeat(DRAW_COUNT, 1, 1, 1)
        report = _direction_group_report(
            draws,
            means,
            source_ids=["source_0", "source_1"],
            group="full",
            n_w=2,
        )
        self.assertEqual(report["status"], "BLOCK")
        self.assertTrue(all(row["status"] == "BLOCK" for row in report["sources"]))


class SemanticGateTest(unittest.TestCase):
    def _anchors(self) -> list[dict]:
        return [
            {"label": "Speech", "target_valid": True},
            {"label": "Vehicle", "target_valid": True},
        ]

    def test_seven_of_eight_draws_pass(self) -> None:
        target_clap = torch.tensor([[1.0, 0.1], [0.1, 1.0]])
        target_ast = torch.tensor([[0.8, 0.1], [0.1, 0.7]])
        clap_draws = [target_clap.clone() for _ in range(DRAW_COUNT)]
        ast_draws = [target_ast.clone() for _ in range(DRAW_COUNT)]
        # One posterior draw may fail without changing the frozen 7/8 decision.
        clap_draws[-1][0] = torch.tensor([0.1, 0.9])
        ast_draws[-1][0] = torch.tensor([0.1, 0.8])
        rows = _semantic_branch_rows(
            branch="test",
            source_ids=["source_0", "source_1"],
            target_clap_matrix=target_clap,
            draw_clap_matrices=clap_draws,
            ast_anchors=self._anchors(),
            target_ast_matrix=target_ast,
            draw_ast_matrices=ast_draws,
            target_rms=[1.0, 1.0],
            draw_rms=[[1.0, 1.0] for _ in range(DRAW_COUNT)],
        )
        self.assertEqual(rows[0]["status"], "PASS")
        self.assertEqual(rows[0]["clap_correct_draw_count"], 7)
        self.assertEqual(rows[0]["ast_correct_draw_count"], 7)

    def test_target_ambiguity_abstains(self) -> None:
        target_clap = torch.tensor([[0.5, 0.49], [0.1, 1.0]])
        target_ast = torch.tensor([[0.8, 0.1], [0.1, 0.7]])
        rows = _semantic_branch_rows(
            branch="test",
            source_ids=["source_0", "source_1"],
            target_clap_matrix=target_clap,
            draw_clap_matrices=[target_clap] * DRAW_COUNT,
            ast_anchors=self._anchors(),
            target_ast_matrix=target_ast,
            draw_ast_matrices=[target_ast] * DRAW_COUNT,
            target_rms=[1.0, 1.0],
            draw_rms=[[1.0, 1.0] for _ in range(DRAW_COUNT)],
        )
        self.assertEqual(rows[0]["status"], "ABSTAIN_CALIBRATION")


class PanelSummaryTest(unittest.TestCase):
    def _report(self, family_rows: list[dict]) -> dict:
        return {
            "schema": "stable_audio_tools.vae_mixture_state_identifiability_audit",
            "schema_version": 1,
            "family_ranks": [row["family_rank"] for row in family_rows],
            "draw_count": 8,
            "shared_posterior_epsilon": True,
            "seed": 1,
            "thresholds": {"frozen": True},
            "rf_times": [0.25, 0.5, 0.75, 1.0],
            "vae_checkpoint_sha256": "abc",
            "clap_model": "clap",
            "ast_model": "ast",
            "families": family_rows,
        }

    def _family(
        self,
        rank: int,
        *,
        semantic: str = "PASS",
        latent: str = "PASS",
    ) -> dict:
        if semantic == "ABSTAIN_CALIBRATION":
            outcome = "ABSTAIN_CALIBRATION"
        elif semantic != "PASS":
            outcome = "VAE_SOURCE_IDENTITY_LOSS"
        elif latent != "PASS":
            outcome = "RECOVERABLE_BUT_ENTANGLED"
        else:
            outcome = "STABLE_RECOVERABLE"
        branch_status = semantic
        branch = {
            "status": branch_status,
            "clap_correct_draw_count": 8 if semantic == "PASS" else 0,
            "ast_correct_draw_count": 8 if semantic == "PASS" else 0,
            "ast_target_retention": {"median": 1.0 if semantic == "PASS" else 0.0},
            "active_rms_retention": {"median": 1.0},
        }
        return {
            "family_rank": rank,
            "family_id": f"family_{rank}",
            "source_count": 1,
            "source_outcomes": [
                {
                    "source_id": "source_0",
                    "semantic_status": semantic,
                    "latent_status": latent,
                    "outcome": outcome,
                }
            ],
            "semantic": {
                "sources": [
                    {
                        "source_id": "source_0",
                        "status": semantic,
                        "isolated": dict(branch),
                        "intervention": dict(branch),
                    }
                ]
            },
            "outcome": outcome,
        }

    def test_family5_semantic_failure_selects_vae_loss(self) -> None:
        report = self._report(
            [self._family(5, semantic="BLOCK"), self._family(6)]
        )
        result = summarize_mixture([report], expected_ranks=[5, 6])
        self.assertEqual(result["decision"], "VAE_SOURCE_IDENTITY_LOSS")

    def test_family5_latent_failure_selects_entangled(self) -> None:
        report = self._report(
            [self._family(5, latent="BLOCK"), self._family(6)]
        )
        result = summarize_mixture([report], expected_ranks=[5, 6])
        self.assertEqual(result["decision"], "RECOVERABLE_BUT_ENTANGLED")

    def test_all_sources_stable_passes(self) -> None:
        report = self._report([self._family(5), self._family(6)])
        result = summarize_mixture([report], expected_ranks=[5, 6])
        self.assertEqual(result["decision"], "STABLE_RECOVERABLE")


class CausalPresenceAdjudicationTest(unittest.TestCase):
    def test_assignment_requires_own_identity_and_gap(self) -> None:
        matrix = torch.tensor([[0.8, 0.1], [0.4, 0.45]])
        first = _assignment_row(
            matrix, 0, source_ids=["source_0", "source_1"], min_gap=0.05
        )
        second = _assignment_row(
            matrix, 1, source_ids=["source_0", "source_1"], min_gap=0.10
        )
        self.assertTrue(first["valid"])
        self.assertTrue(second["correct"])
        self.assertFalse(second["valid"])

    def _causal_tensors(self, *, post_gap_scale: float = 1.0) -> dict:
        # States are full, minus-source-0, minus-source-1.  Rows are persistent
        # spatial slots and columns are frozen source identities/AST anchors.
        target_clap = torch.tensor(
            [
                [[0.9, 0.1], [0.1, 0.9]],
                [[0.1, 0.1], [0.1, 0.9]],
                [[0.9, 0.1], [0.1, 0.1]],
            ]
        )
        target_ast = torch.tensor(
            [
                [[0.8, 0.1], [0.1, 0.8]],
                [[0.1, 0.1], [0.1, 0.8]],
                [[0.8, 0.1], [0.1, 0.1]],
            ]
        )
        decoded_clap = target_clap.unsqueeze(0).repeat(DRAW_COUNT, 1, 1, 1)
        decoded_ast = target_ast.unsqueeze(0).repeat(DRAW_COUNT, 1, 1, 1)
        target_instances = torch.eye(2).unsqueeze(0).repeat(3, 1, 1)
        decoded_to_target = target_instances.unsqueeze(0).repeat(
            DRAW_COUNT, 1, 1, 1
        )
        if post_gap_scale < 1.0:
            # Collapse only source-0's decoded full/minus causal gaps while
            # leaving exact-target calibration unchanged.
            decoded_clap[:, 1, 0, 0] = (
                decoded_clap[:, 0, 0, 0] - 0.8 * post_gap_scale
            )
            decoded_ast[:, 1, 0, 0] = (
                decoded_ast[:, 0, 0, 0] - 0.7 * post_gap_scale
            )
        return {
            "target_clap": target_clap,
            "decoded_clap": decoded_clap,
            "target_clap_instances": target_instances,
            "decoded_to_target_clap": decoded_to_target,
            "target_ast": target_ast,
            "decoded_ast": decoded_ast,
            "target_rms": torch.ones(3, 2),
            "decoded_rms": torch.ones(DRAW_COUNT, 3, 2),
            "anchors": [
                {"target_valid": True},
                {"target_valid": True},
            ],
        }

    def test_direct_full_minus_presence_passes_when_causal_gaps_survive(self) -> None:
        report = _source_causal_report(
            self._causal_tensors(),
            removed_index=0,
            source_ids=["source_0", "source_1"],
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["draw_pass_count"], DRAW_COUNT)

    def test_direct_full_minus_presence_blocks_when_causal_gap_collapses(self) -> None:
        report = _source_causal_report(
            self._causal_tensors(post_gap_scale=0.1),
            removed_index=0,
            source_ids=["source_0", "source_1"],
        )
        self.assertEqual(report["status"], "BLOCK")
        self.assertEqual(
            report["failure_draw_counts"]["removed_ast_causal_gap"], DRAW_COUNT
        )


class CausalPresencePanelSummaryTest(unittest.TestCase):
    def _family(self, rank: int, status: str) -> dict:
        outcome = {
            "PASS": "CAUSAL_PRESENCE_PRESERVED",
            "BLOCK": "CAUSAL_SOURCE_PRESENCE_LOSS",
            "ABSTAIN_CALIBRATION": "ABSTAIN_CALIBRATION",
        }[status]
        return {
            "family_rank": rank,
            "family_id": f"family_{rank}",
            "source_count": 1,
            "sources": [
                {
                    "source_id": "source_0",
                    "status": status,
                    "failure_draw_counts": {
                        "removed_ast_causal_gap": DRAW_COUNT
                        if status == "BLOCK"
                        else 0
                    },
                }
            ],
            "outcome": outcome,
        }

    def _report(self, families: list[dict]) -> dict:
        return {
            "schema": "stable_audio_tools.vae_causal_source_presence_adjudication",
            "schema_version": 2,
            "adjudication": "differential_causal_gap_plus_exact_target_instance_fidelity",
            "family_ranks": [family["family_rank"] for family in families],
            "draw_count": DRAW_COUNT,
            "shared_posterior_epsilon": True,
            "seed": 1,
            "thresholds": {"frozen": True},
            "vae_checkpoint_sha256": "abc",
            "clap_model": "clap",
            "ast_model": "ast",
            "families": families,
        }

    def test_family5_failure_confirms_vae_causal_loss(self) -> None:
        result = summarize_causal(
            [self._report([self._family(5, "BLOCK"), self._family(6, "PASS")])],
            expected_ranks=[5, 6],
        )
        self.assertEqual(result["combined_decision"], "VAE_CAUSAL_SOURCE_PRESENCE_LOSS")

    def test_direct_pass_with_subtraction_failure_selects_entanglement(self) -> None:
        subtraction = {
            "schema": "stable_audio_tools.vae_mixture_state_identifiability_summary",
            "decision": "VAE_SOURCE_IDENTITY_LOSS",
        }
        result = summarize_causal(
            [self._report([self._family(5, "PASS"), self._family(6, "PASS")])],
            expected_ranks=[5, 6],
            subtraction_summary=subtraction,
        )
        self.assertEqual(
            result["combined_decision"], "NONLINEAR_MIXTURE_STATE_ENTANGLEMENT"
        )


class VaeCapacityOverfitProtocolTest(unittest.TestCase):
    def test_z128_probe_changes_only_latent_boundaries(self) -> None:
        base = {
            "model": {
                "latent_dim": 64,
                "encoder": {"config": {"latent_dim": 128, "channels": 8}},
                "decoder": {"config": {"latent_dim": 64, "channels": 8}},
                "bottleneck": {"type": "grouped_vae", "config": {"n_w": 40}},
                "downsampling_ratio": 1024,
            }
        }
        wide = _expanded_vae_config(base, 128)
        self.assertEqual(wide["model"]["latent_dim"], 128)
        self.assertEqual(wide["model"]["encoder"]["config"]["latent_dim"], 256)
        self.assertEqual(wide["model"]["decoder"]["config"]["latent_dim"], 128)
        self.assertEqual(wide["model"]["bottleneck"]["config"]["n_w"], 40)
        self.assertEqual(base["model"]["latent_dim"], 64)

    def test_cosine_loss_schedule_is_clamped_and_hits_midpoint(self) -> None:
        schedule = {
            "type": "cosine",
            "start_step": 10,
            "end_step": 20,
            "start_weight": 0.0,
            "end_weight": 0.3,
        }
        self.assertEqual(cosine_schedule_value(schedule, 0), 0.0)
        self.assertEqual(cosine_schedule_value(schedule, 10), 0.0)
        self.assertAlmostEqual(cosine_schedule_value(schedule, 15), 0.15)
        self.assertEqual(cosine_schedule_value(schedule, 20), 0.3)
        self.assertEqual(cosine_schedule_value(schedule, 30), 0.3)

    def test_three_crops_cover_both_ends_and_schedule_balances_states(self) -> None:
        starts = _crop_starts(442_368, 176_400, count=3)
        self.assertEqual(starts, [0, 132_984, 265_968])
        schedule = _training_schedule(["full", "minus", "isolated"], starts, 18)
        counts = {name: 0 for name in ("full", "minus", "isolated")}
        pairs = set()
        for name, start in schedule:
            counts[name] += 1
            pairs.add((name, start))
        self.assertEqual(counts, {"full": 6, "minus": 6, "isolated": 6})
        self.assertEqual(len(pairs), 9)

    def test_loss_summary_is_state_balanced_and_reports_improvement(self) -> None:
        rows = []
        for step in range(1, 43):
            rows.append(
                {
                    "state_name": "a" if step % 2 else "b",
                    "total": 2.0 if step <= 21 else 1.0,
                    "mrstft": 1.5 if step <= 21 else 0.75,
                }
            )
        summary = _loss_summary(rows)
        self.assertAlmostEqual(summary["total_ratio"], 0.5)
        self.assertAlmostEqual(summary["mrstft_ratio"], 0.5)
        self.assertEqual(summary["states"]["a"]["count"], 21)
        self.assertEqual(summary["states"]["b"]["count"], 21)

    def _training_report(self) -> dict:
        return {
            "schema": "stable_audio_tools.vae_spatial_family_capacity_overfit",
            "family_rank": 5,
            "family_id": "family_5",
            "status": "OPTIMIZATION_PASS",
            "loss_summary": {"total_ratio": 0.5, "mrstft_ratio": 0.6},
            "inference_contract": "one unified latent and one decode",
            "checkpoints": [
                {"step": 128, "sha256": "hash128", "bytes": 100},
                {"step": 256, "sha256": "hash256", "bytes": 100},
            ],
        }

    def _evaluation(self, checkpoint_hash: str, status: str) -> dict:
        outcome = (
            "CAUSAL_PRESENCE_PRESERVED"
            if status == "PASS"
            else "CAUSAL_SOURCE_PRESENCE_LOSS"
        )
        source = {
            "source_id": "source_0",
            "status": status,
            "draw_pass_count": 8 if status == "PASS" else 0,
            "clap_causal_gap_retention": {"median": 1.0},
            "ast_causal_gap_retention": {"median": 1.0},
            "failure_draw_counts": {
                "removed_ast_causal_gap": 0 if status == "PASS" else 8
            },
        }
        return {
            "schema": "stable_audio_tools.vae_causal_source_presence_adjudication",
            "schema_version": 2,
            "posterior_mean_included": True,
            "vae_checkpoint_sha256": checkpoint_hash,
            "thresholds": {"frozen": True},
            "families": [
                {
                    "family_rank": 5,
                    "outcome": outcome,
                    "sources": [source],
                    "posterior_mean": {
                        "outcome": outcome,
                        "sources": [source],
                    },
                }
            ],
        }

    def test_capacity_summary_blocks_when_every_snapshot_fails(self) -> None:
        result = summarize_capacity(
            self._training_report(),
            [
                self._evaluation("hash128", "BLOCK"),
                self._evaluation("hash256", "BLOCK"),
            ],
        )
        self.assertEqual(
            result["decision"],
            "ORDINARY_RECONSTRUCTION_CAPACITY_NOT_DEMONSTRATED",
        )
        self.assertEqual(
            result["checkpoint_disposition"]["action"],
            "delete_rejected_checkpoints",
        )

    def test_capacity_summary_accepts_a_sampled_and_mean_pass(self) -> None:
        result = summarize_capacity(
            self._training_report(),
            [
                self._evaluation("hash128", "BLOCK"),
                self._evaluation("hash256", "PASS"),
            ],
        )
        self.assertEqual(result["decision"], "CAPACITY_DEMONSTRATED")
        self.assertEqual(result["accepted_steps"], [256])


if __name__ == "__main__":
    unittest.main()
