"""Custom TTS text aggregator: trigger on commas/colons/semicolons too.

Pipecat default (`SimpleTextAggregator`) only flushes to TTS at sentence
boundaries (`. ! ? ; …`). For LLM responses that include commas (multi-clause
sentences), this means the user waits until the end of the first sentence
to hear anything — even when the first half is a self-contained phrase.

This aggregator extends the parent to ALSO flush on phrase punctuation
(`,` `;` `:`). Cuts TTFA for multi-clause replies. Trade-off: speech may
sound slightly more chopped between phrases.

Heuristics:
- Sentence boundaries (`. ! ?`) still go through the parent's NLTK lookahead
  (so "$29." vs "$29. Next" disambiguation still works).
- Phrase boundaries (`,` etc.) fire immediately when buffered text is at
  least MIN_PHRASE_LEN chars — avoids breaking on "1,000" or "Hi,".
"""
from collections.abc import AsyncIterator

from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


# Phrase-boundary punctuation. Subset of natural break points that are short
# of full-stop. Including ":" because LLM list-headers like "Three things:"
# are good cut points too.
PHRASE_BREAK_PUNCT: frozenset[str] = frozenset(
    {
        ",",
        ":",
        # ";" is already in SENTENCE_ENDING_PUNCTUATION (parent handles it)
        # Full-width / non-Latin equivalents:
        "，",  # full-width comma (CJK)
        "、",  # CJK ideographic comma
        "：",  # full-width colon
    }
)

# Minimum buffered length (chars, after strip) before a phrase break fires.
# 6 chars cuts "Hi," ("Hi," = 3 chars) but allows "Hello," ("Hello," = 6).
# Also prevents firing on "1," (only 2 chars) inside numbers like "1,000".
MIN_PHRASE_LEN: int = 6


class PhraseAwareTextAggregator(SimpleTextAggregator):
    """Flush TTS output at phrase-level punctuation in addition to sentences."""

    async def _check_sentence_with_lookahead(self, char: str):
        # Let the parent handle sentence-boundary detection (with NLTK lookahead).
        result = await super()._check_sentence_with_lookahead(char)
        if result:
            return result

        # If we're currently in lookahead mode for a sentence boundary, don't
        # trigger phrase cuts — the lookahead is meaningful and we want to
        # preserve the post-period text intact.
        if self._needs_lookahead:
            return None

        if char not in PHRASE_BREAK_PUNCT:
            return None

        # Buffer just received the phrase punctuation as last char. Cut here
        # if the accumulated phrase is long enough to be worth a TTS call.
        stripped = self._text.strip()
        if len(stripped) < MIN_PHRASE_LEN:
            return None

        phrase = self._text.strip(" ")
        self._text = ""
        return Aggregation(text=phrase, type=AggregationType.SENTENCE)
