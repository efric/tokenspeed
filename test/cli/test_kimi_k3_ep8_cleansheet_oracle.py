# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/megamoe/kimi_k3_ep8_cleansheet_oracle.py"
)
_SPEC = importlib.util.spec_from_file_location("kimi_k3_ep8_cleansheet_oracle", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
oracle = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = oracle
_SPEC.loader.exec_module(oracle)


class KimiK3EP8CleanSheetOracleTest(unittest.TestCase):
    def test_structural_self_check(self) -> None:
        oracle.self_check()

    def test_balanced_and_skew_counts(self) -> None:
        balanced = oracle.build_oracle(16, "balanced_unique")
        self.assertEqual(balanced["rank_route_counts"], [32] * 8)

        skewed = oracle.build_oracle(16, "rank_skew")
        self.assertEqual(skewed["rank_route_counts"], [256] + [0] * 7)
        self.assertEqual(skewed["ranks"][0]["unique_local_experts"], 16)

    def test_hot_routes_expose_weight_reuse_without_changing_rank_balance(self) -> None:
        unique = oracle.build_oracle(16, "balanced_unique")
        hot = oracle.build_oracle(16, "hot_balanced")
        self.assertEqual(unique["rank_route_counts"], hot["rank_route_counts"])
        self.assertEqual(hot["ranks"][0]["unique_local_experts"], 2)
        self.assertGreater(unique["ranks"][0]["unique_local_experts"], 2)

    def test_row_models_keep_native_route_direct_distinct_from_grouped(self) -> None:
        case = oracle.build_oracle(8, "balanced_unique")
        rank = case["ranks"][0]
        rows = rank["row_work_models"]
        self.assertEqual(rows["native_route_direct"], 16)
        self.assertEqual(rows["grouped_bm64_counterfactual"], 16 * 64)
        self.assertEqual(rows["stable_bucketed_1_2_4_8_16"], 16)

    def test_b1_uses_output_centric_w2_without_route_output_workspace(self) -> None:
        case = oracle.build_oracle(1, "balanced_unique")
        self.assertEqual(
            case["workspace"]["local_bytes"]["w2_route_output_bf16_batched_only"],
            0,
        )
        self.assertEqual(case["ranks"][0]["tasks"]["w2_output_centric_n32"], 112)

    def test_b16_workspace_and_network_formulas(self) -> None:
        case = oracle.build_oracle(16, "hot_balanced")
        self.assertEqual(
            case["workspace"]["symmetric_bytes"]["joint_producer_bf16"],
            344064,
        )
        self.assertEqual(
            case["workspace"]["symmetric_bytes"]["ready_epoch_flags_u64"],
            21504,
        )
        self.assertEqual(
            case["workspace"]["symmetric_bytes"]["completion_epoch_flags_u64"],
            21504,
        )
        self.assertEqual(
            case["network_payload_bytes"]["remote_reads_per_rank"], 2408448
        )
        self.assertEqual(case["workspace"]["communication_tiles"], 336)

    def test_architecture_records_placement_and_communication_gates(self) -> None:
        architecture = oracle._architecture()
        placement = architecture["k1_candidate"]["physical_placement_gate"]
        controls = architecture["measured_controls"]
        isa = controls["historical_ordinary_b1_isa"]
        native = controls["native_ep8_route_direct"]
        joint = controls["native_b1_joint_expert_shared"]
        protocol = architecture["communication_contract"]
        serving = architecture["serving_correctness_gate"]

        self.assertEqual(placement["expected_compute_units"], 256)
        self.assertIn("xcd", placement["report_fields"])
        self.assertFalse(isa["present_on_refreshed_base"])
        self.assertEqual(isa["text_bytes"], 80384)
        self.assertEqual(isa["s_clause_instructions"], 0)
        self.assertEqual(native["organization"], "route_direct_warp_gemv")
        self.assertEqual(native["target_batches"], [1, 8, 16])
        self.assertEqual(native["maximum_tokens"], 16)
        self.assertFalse(native["sorts_routes"])
        self.assertFalse(native["pads_routes_to_bm64"])
        self.assertEqual(joint["lane_order"], ["shared_hidden", "routed_latent"])
        self.assertIn(
            "do not overwrite until local and every remote consumer completion g is known",
            protocol["peer_protocol"],
        )
        self.assertIn(
            "the workspace contains alternate payload slots",
            protocol["rejected_assumptions"],
        )
        choices = {
            choice["name"] for choice in protocol["peer_load_publication_choices"]
        }
        self.assertEqual(
            choices,
            {"per_subgroup_system_acquire", "leader_to_versioned_lds"},
        )
        self.assertIn(
            "reject mixed generations or early completion",
            protocol["subgroup_stale_payload_litmus"],
        )
        self.assertIn("do not require", serving["batch_8_16"])
        self.assertIn(
            "positive loaded-code and MegaMoE dispatch proof",
            serving["all_batches"],
        )

    def test_refresh_metadata_rejects_a_stale_production_baseline(self) -> None:
        self.assertEqual(
            oracle.REFRESHED_BASE_COMMIT,
            "e784229526ce11d272a3c4a0b3f64ab9a8973491",
        )
        self.assertEqual(oracle.SUPPORTED_BATCHES, (1, 8, 16))

        artifact = oracle.artifact_contract()
        self.assertFalse(artifact["production_megamoe_package_present"])
        self.assertFalse(artifact["production_code_changed"])


if __name__ == "__main__":
    unittest.main()
