# Upstream compatibility notes

The implementation was written against these inspected upstream interfaces on
2026-08-28:

- faster-whisper `transcribe.py` commit
  `ed9a06cd89a93e47838f564998a6c09b655d7f43`. Its normal generation path reads
  `result.sequences_ids[0]` and `result.scores[0]`, while this bundle preserves
  all returned hypotheses.
- CTranslate2 Whisper Python API documentation 4.8.1, whose `generate` method
  exposes `num_hypotheses`, `return_scores`, and `return_no_speech_prob`.
- Hugging Face Transformers `modeling_whisper.py` commit
  `281dd533060988a1de8d063c4c1ea72b304a2bb8`. The current conditional-generation
  model accepts tuple encoder outputs, enabling one shared encoder pass.
- Ollama native `/api/chat` documentation, including JSON-schema `format`,
  non-streaming responses, and local default endpoint behavior.

Upstream references:

- https://github.com/SYSTRAN/faster-whisper
- https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html
- https://github.com/huggingface/transformers
- https://docs.ollama.com/api/chat
- https://docs.ollama.com/capabilities/structured-outputs

Because `scripts/whisper_nbest.py` uses faster-whisper internals such as
`feature_extractor`, `get_prompt`, and the wrapped CTranslate2 model, pin the
working faster-whisper/CTranslate2 pair in the PoC and run the supplied API smoke
test whenever either dependency is upgraded.
