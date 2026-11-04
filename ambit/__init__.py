"""AMBIT spatial audio inference."""

__version__ = "0.2.0"
__all__ = ["UnifiedPipeline", "GenerationPipeline", "EditingPipeline", "InferenceResult"]


def __getattr__(name):
    if name in {"UnifiedPipeline", "GenerationPipeline", "EditingPipeline"}:
        from . import pipeline

        return getattr(pipeline, name)
    if name == "InferenceResult":
        from .audio import InferenceResult

        return InferenceResult
    raise AttributeError(name)
