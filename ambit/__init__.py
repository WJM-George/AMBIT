"""AMBIT spatial audio inference."""

__version__ = "0.1.0"
__all__ = ["GenerationPipeline", "EditingPipeline", "InferenceResult"]


def __getattr__(name):
    if name in {"GenerationPipeline", "EditingPipeline"}:
        from . import pipeline

        return getattr(pipeline, name)
    if name == "InferenceResult":
        from .audio import InferenceResult

        return InferenceResult
    raise AttributeError(name)
