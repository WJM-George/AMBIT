import unittest

from stable_audio_tools.training.transfusion_opsd.editing_stream import OrdinalStream, resize_stream_position


class StreamResizeTests(unittest.TestCase):
    def check_resize(self, counts):
        original = [OrdinalStream(range(256), seed=37, rank=r, world=4) for r in range(4)]
        consumed = [x for stream, count in zip(original, counts) for x in stream.take(count)]
        states = [s.state_dict() for s in original]
        current = [OrdinalStream(range(256), seed=37, rank=r, world=8,
            state=resize_stream_position(states, new_world=8, rank=r)) for r in range(8)]
        first = [x for s in current for x in s.take(1)]
        # Checkpoint while only some of the ragged warmup holes were drained.
        current = [OrdinalStream(range(256), seed=37, rank=r, world=8, state=s.state_dict())
                   for r, s in enumerate(current)]
        rest = [x for s in current for x in s.take(len(s.order)-s.cursor+len(s.pending_positions))]
        self.assertEqual(len(consumed + first + rest), 256)
        self.assertEqual(len(set(consumed + first + rest)), 256)

    def test_actual_warmup_cursors(self):
        self.check_resize([14, 14, 16, 16])

    def test_aligned_previous_behavior(self):
        self.check_resize([16, 16, 16, 16])

    def test_more_than_one_pending_example_and_nondivisible_frontier(self):
        self.check_resize([1, 5, 3, 7])

    def test_different_epochs_are_not_silently_merged(self):
        states = [dict(epoch=0, cursor=5, seed=37, rank=r, world=4) for r in range(4)]
        states[3]['epoch'] = 1
        with self.assertRaises(ValueError):
            resize_stream_position(states, new_world=8, rank=0)


if __name__ == '__main__':
    unittest.main()
