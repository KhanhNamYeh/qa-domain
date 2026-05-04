"""
Vietnamese word segmentation for PhoBERT input.

PhoBERT-v2 was pretrained on text where compound words are joined with `_`
(e.g. "con lân" -> "con_lân"). Feeding raw text bypasses these compound tokens
in the vocabulary and degrades embedding quality. This module wraps `pyvi` to
segment text on the fly. The segmenter is loaded lazily so importing this
module does not crash if pyvi is missing — only the actual call does, with a
clear error message.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _get_segmenter():
    try:
        from pyvi.ViTokenizer import tokenize as _seg
    except ImportError as e:
        raise ImportError(
            "pyvi is required for PhoBERT word segmentation. "
            "Install with: pip install pyvi"
        ) from e
    return _seg


def segment(text: str) -> str:
    """Segment a Vietnamese sentence into PhoBERT-compatible form.

    Empty / None input is passed through unchanged so callers don't need to
    guard against it.
    """
    if not text:
        return text
    return _get_segmenter()(text)


def desegment(text: str) -> str:
    """Reverse: undo the underscore-joining so the output reads naturally
    (used for metric computation against raw references)."""
    return text.replace("_", " ") if text else text
