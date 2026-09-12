from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import unittest

from scripts.t2a.data.build_sceneplan_transfusion_editing_pairs import (
    _create_stage,
    _select_full,
)


class EditingFullPairSelectionTests(unittest.TestCase):
    def test_full_selection_reserves_eligible_families_before_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = _create_stage(Path(directory) / "stage.sqlite")
            try:
                rows = []
                for ordinal in range(12):
                    has_static = ordinal < 4
                    has_linear = 4 <= ordinal < 8
                    rows.append(
                        (
                            ordinal,
                            f"sample-{ordinal:02d}",
                            2,
                            "music_sound",
                            432,
                            int(has_static),
                            int(has_linear),
                            f"{ordinal:016x}",
                            f"{11 - ordinal:016x}",
                        )
                    )
                connection.executemany(
                    "INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?)", rows
                )
                connection.commit()
                # ``selected`` is intentionally empty here. The old full-mode
                # implementation incorrectly joined that empty table.
                _select_full(connection, {(2, "music_sound", 432): 12}, 9)
                selected = connection.execute(
                    """
                    SELECT s.operation_family,s.operation,c.has_static,c.has_linear
                    FROM selected AS s JOIN candidates AS c USING(source_ordinal)
                    ORDER BY s.output_ordinal
                    """
                ).fetchall()
            finally:
                connection.close()
        self.assertEqual(len(selected), 9)
        self.assertEqual(
            Counter(row[0] for row in selected),
            {
                "event_add_remove": 3,
                "stationary_azimuth_change": 3,
                "static_linear_toggle": 3,
            },
        )
        self.assertTrue(
            all(row[2] for row in selected if row[0] == "stationary_azimuth_change")
        )
        self.assertTrue(
            all(
                row[2] or row[3]
                for row in selected
                if row[0] == "static_linear_toggle"
            )
        )


if __name__ == "__main__":
    unittest.main()
