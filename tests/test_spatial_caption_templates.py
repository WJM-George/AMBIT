from __future__ import annotations

import unittest

from stable_audio_tools.data.spatial_caption_templates import (
    SEMANTIC_CAPTION_TEMPLATE_COUNT,
    SEMANTIC_CAPTION_TEMPLATE_VERSION,
    render_semantic_caption,
    validate_semantic_caption_metadata,
)


class SpatialCaptionTemplateTests(unittest.TestCase):
    def test_source_regions_cover_semantics_not_wrapper_text(self):
        sources = [
            {
                "source_id": "source_0",
                "event": {"label": "Music"},
                "content": {},
            },
            {
                "source_id": "source_1",
                "event": {"label": "Bird"},
                "content": {},
            },
        ]
        # Exercise every wrapper while keeping the other template axes fixed.
        for wrapper_index in range(8):
            template_id = wrapper_index * 128
            rendered = render_semantic_caption(sources, template_id=template_id)
            metadata = rendered.metadata()
            validate_semantic_caption_metadata(
                rendered.text,
                metadata,
                expected_source_ids=("source_0", "source_1"),
            )
            spans = [
                rendered.text[item["start"] : item["end"]]
                for item in metadata["source_regions"]
            ]
            self.assertEqual(spans, ["Music", "Bird"])

    def test_transcript_region_remains_nested_in_content_region(self):
        rendered = render_semantic_caption(
            [
                {
                    "source_id": "source_0",
                    "event": {"label": "speech"},
                    "content": {"transcript": "Turn left."},
                }
            ],
            # Wrapper includes lexical source-ID text before the content.
            template_id=1024,
        )
        metadata = rendered.metadata()
        validate_semantic_caption_metadata(
            rendered.text,
            metadata,
            expected_source_ids=("source_0",),
        )
        source = metadata["source_regions"][0]
        transcript = metadata["transcript_regions"][0]
        self.assertEqual(
            rendered.text[source["start"] : source["end"]],
            'speech saying "Turn left."',
        )
        self.assertEqual(
            rendered.text[transcript["start"] : transcript["end"]],
            "Turn left.",
        )
        self.assertLessEqual(source["start"], transcript["start"])
        self.assertLessEqual(transcript["end"], source["end"])

    def test_version_marks_content_only_region_contract(self):
        self.assertEqual(
            SEMANTIC_CAPTION_TEMPLATE_VERSION,
            "spatial_source_regions_v3",
        )


if __name__ == "__main__":
    unittest.main()
