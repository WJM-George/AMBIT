import unittest

import torch

from stable_audio_tools.training.dit_teacher import DenseDiTRectifiedFlowTeacher


def _teacher_for_selection():
    teacher = object.__new__(DenseDiTRectifiedFlowTeacher)
    teacher.metadata_equals = {"curriculum_kind": "mixture_creation"}
    teacher.max_examples_per_group = 1
    teacher.group_key = "family_id"
    teacher.rotation_key = "family_turn_index"
    teacher.every_n_steps = 1
    teacher.step_offset = 0
    teacher._selection_fraction = None
    teacher._pulse_active = None
    return teacher


class DenseDiTTeacherSelectionTest(unittest.TestCase):
    def test_filter_and_turn_rotation(self):
        teacher = _teacher_for_selection()
        metadata = [
            {
                "curriculum_kind": kind,
                "family_id": family,
                "family_turn_index": turn,
            }
            for kind, family in (
                ("mixture_creation", "a"),
                ("isolated_source_creation", "b"),
            )
            for turn in range(4)
        ]
        enabled0 = teacher.select_enabled_samples(
            metadata, [True] * 8, step=0, device=torch.device("cpu")
        )
        enabled3 = teacher.select_enabled_samples(
            metadata, [True] * 8, step=3, device=torch.device("cpu")
        )
        self.assertEqual(enabled0, [True, False, False, False] + [False] * 4)
        self.assertEqual(enabled3, [False, False, False, True] + [False] * 4)
        self.assertAlmostEqual(float(teacher._selection_fraction), 1 / 8)

    def test_low_frequency_steps_still_rotate_turns(self):
        teacher = _teacher_for_selection()
        teacher.every_n_steps = 4
        metadata = [
            {
                "curriculum_kind": "mixture_creation",
                "family_id": "a",
                "family_turn_index": turn,
            }
            for turn in range(4)
        ]
        disabled = teacher.select_enabled_samples(
            metadata, [True] * 4, step=1, device=torch.device("cpu")
        )
        enabled4 = teacher.select_enabled_samples(
            metadata, [True] * 4, step=4, device=torch.device("cpu")
        )
        self.assertEqual(disabled, [False] * 4)
        self.assertEqual(enabled4, [False, True, False, False])
        self.assertEqual(float(teacher._pulse_active), 1.0)

    def test_loss_forwards_only_selected_examples(self):
        teacher = object.__new__(DenseDiTRectifiedFlowTeacher)
        teacher.channel_start = 0
        teacher.channel_end = 2
        teacher.loss_weight = 0.1
        teacher.student_time_min = 0.0
        teacher.student_time_max = 1.0
        teacher.last_metrics = {}
        teacher._selection_fraction = torch.tensor(0.5)
        teacher._pulse_active = torch.tensor(1.0)
        seen = {}

        def velocity(noised, times, conditioning, valid):
            seen["shape"] = tuple(noised.shape)
            seen["conditioning"] = list(conditioning)
            return torch.zeros_like(noised)

        teacher._teacher_velocity = velocity
        predicted = [torch.ones(2, 3), torch.full((2, 3), 9.0)]
        noised = [torch.zeros(2, 3), torch.zeros(2, 3)]
        loss = teacher.loss(
            predicted,
            noised,
            [torch.tensor(0.25), torch.tensor(0.75)],
            conditioning=[{"prompt": "kept"}, {"prompt": "dropped"}],
            active_masks=[torch.ones(3), torch.ones(3)],
            enabled_samples=[True, False],
        )
        self.assertEqual(seen["shape"], (1, 2, 3))
        self.assertEqual(seen["conditioning"], [{"prompt": "kept"}])
        self.assertAlmostEqual(float(loss), 0.1, places=6)
        self.assertAlmostEqual(
            float(teacher.last_metrics["selection_fraction"]), 0.5
        )

    def test_loss_filters_to_configured_student_time_range(self):
        teacher = object.__new__(DenseDiTRectifiedFlowTeacher)
        teacher.channel_start = 0
        teacher.channel_end = 2
        teacher.loss_weight = 0.1
        teacher.student_time_min = 0.0
        teacher.student_time_max = 0.5
        teacher.last_metrics = {}
        teacher._selection_fraction = torch.tensor(1.0)
        teacher._pulse_active = torch.tensor(1.0)
        seen = {}

        def velocity(noised, times, conditioning, valid):
            seen["times"] = times.tolist()
            seen["conditioning"] = list(conditioning)
            return torch.zeros_like(noised)

        teacher._teacher_velocity = velocity
        predicted = [torch.ones(2, 3), torch.full((2, 3), 9.0)]
        noised = [torch.zeros(2, 3), torch.zeros(2, 3)]
        loss = teacher.loss(
            predicted,
            noised,
            [torch.tensor(0.25), torch.tensor(0.75)],
            conditioning=[{"prompt": "kept"}, {"prompt": "dropped"}],
            active_masks=[torch.ones(3), torch.ones(3)],
            enabled_samples=[True, True],
        )
        self.assertEqual(seen["times"], [0.25])
        self.assertEqual(seen["conditioning"], [{"prompt": "kept"}])
        self.assertAlmostEqual(float(loss), 0.1, places=6)
        self.assertAlmostEqual(
            float(teacher.last_metrics["time_selection_fraction"]), 0.5
        )
        self.assertAlmostEqual(
            float(teacher.last_metrics["effective_selection_fraction"]), 0.5
        )


if __name__ == "__main__":
    unittest.main()
