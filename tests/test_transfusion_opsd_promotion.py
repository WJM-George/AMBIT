import pytest

from stable_audio_tools.training.transfusion_opsd.promotion import ARMS, PromotionRule, PairedMeasurement, assess_promotion


def test_promotion_requires_all_routes_and_counts_scenes_not_repeated_seeds():
    rules = [PromotionRule(arm, "all", "primary", True, .4, min_clusters=4) for arm in sorted(ARMS)]
    rules += [PromotionRule(arm, "all", "protection", False, .4, min_clusters=4) for arm in sorted(ARMS)]
    data = [PairedMeasurement(rule.arm, "all", rule.metric, f"scene-{scene}", .4, .6)
            for rule in rules for scene in range(4) for seed in range(3)]
    result = assess_promotion(data, rules, protocol_id="test", candidate_id="test", resamples=2000)
    assert result["accepted"] and all(r["clusters"] == 4 for r in result["rules"])
    repeated = [PairedMeasurement(d.arm, d.group, d.metric, "same-scene", d.baseline, d.candidate) for d in data]
    assert not assess_promotion(repeated, rules, protocol_id="test", candidate_id="test", resamples=2000)["accepted"]
    with pytest.raises(ValueError, match="four"):
        assess_promotion(data, rules[:-1], protocol_id="test", candidate_id="test", resamples=2000)
