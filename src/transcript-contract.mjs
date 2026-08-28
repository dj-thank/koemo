import { createHash } from "node:crypto";

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonicalize(value[key])]),
    );
  }
  return value;
}

export function stableStringify(value) {
  return JSON.stringify(canonicalize(value));
}

export function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  Object.freeze(value);
  for (const child of Object.values(value)) deepFreeze(child);
  return value;
}

function normalizeCandidate(candidate, index) {
  if (!candidate || typeof candidate.text !== "string") {
    throw new TypeError(`candidate ${index} must contain text`);
  }
  return {
    id: candidate.id ?? index,
    text: candidate.text,
    tokenIds: Array.isArray(candidate.tokenIds) ? [...candidate.tokenIds] : [],
    whisperScore: Number.isFinite(Number(candidate.whisperScore))
      ? Number(candidate.whisperScore)
      : null,
    moraCtcScore: Number.isFinite(Number(candidate.moraCtcScore))
      ? Number(candidate.moraCtcScore)
      : null,
    localLmScore: Number.isFinite(Number(candidate.localLmScore))
      ? Number(candidate.localLmScore)
      : null,
    finalScore: Number.isFinite(Number(candidate.finalScore))
      ? Number(candidate.finalScore)
      : null,
  };
}

function evidencePayload(record) {
  return {
    schemaVersion: record.schemaVersion,
    audioSha256: record.audioSha256,
    observedTranscript: record.observedTranscript,
    selectedCandidateId: record.selectedCandidateId,
    candidates: record.candidates,
    moraUnits: record.moraUnits,
    uncertaintySpans: record.uncertaintySpans,
  };
}

export function createTranscriptRecord({
  audioSha256,
  candidates,
  selectedCandidateId,
  moraUnits = [],
  uncertaintySpans = [],
  metadata = {},
}) {
  if (!Array.isArray(candidates) || candidates.length === 0) {
    throw new TypeError("candidates must be a non-empty array");
  }

  const normalizedCandidates = candidates.map(normalizeCandidate);
  const ids = new Set(normalizedCandidates.map(({ id }) => id));
  if (ids.size !== normalizedCandidates.length) {
    throw new Error("candidate ids must be unique");
  }

  const selected = normalizedCandidates.find(({ id }) => id === selectedCandidateId);
  if (!selected) throw new Error("selectedCandidateId is not present in candidates");

  const record = {
    schemaVersion: "0.3.0",
    audioSha256: audioSha256 ?? null,
    observedTranscript: selected.text,
    selectedCandidateId,
    candidates: normalizedCandidates,
    moraUnits: structuredClone(moraUnits),
    uncertaintySpans: structuredClone(uncertaintySpans),
    normalizedTranscript: null,
    normalization: null,
    metadata: structuredClone(metadata),
    observedEvidenceSha256: null,
  };

  record.observedEvidenceSha256 = sha256(stableStringify(evidencePayload(record)));
  return deepFreeze(record);
}

export function assertObservedIntegrity(record) {
  if (!record || typeof record !== "object") throw new TypeError("record is required");
  const expected = sha256(stableStringify(evidencePayload(record)));
  if (expected !== record.observedEvidenceSha256) {
    throw new Error("observed transcript evidence was modified");
  }
  const selected = record.candidates.find(({ id }) => id === record.selectedCandidateId);
  if (!selected || selected.text !== record.observedTranscript) {
    throw new Error("observed transcript does not match the selected acoustic candidate");
  }
  return true;
}

export function addNormalization(record, {
  text,
  selectedCandidateId = null,
  model = null,
  rationale = null,
} = {}) {
  assertObservedIntegrity(record);
  if (typeof text !== "string") throw new TypeError("normalized text is required");

  if (selectedCandidateId != null) {
    const allowed = record.candidates.some(({ id }) => id === selectedCandidateId);
    if (!allowed) throw new Error("normalization selected an unknown candidate id");
  }

  const updated = {
    ...structuredClone(record),
    normalizedTranscript: text,
    normalization: {
      mode: selectedCandidateId == null ? "separate-rewrite" : "rank-only",
      selectedCandidateId,
      model,
      rationale,
    },
  };

  assertObservedIntegrity(updated);
  return deepFreeze(updated);
}
