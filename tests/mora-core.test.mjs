import test from "node:test";
import assert from "node:assert/strict";

import {
  countMora,
  mergeCharacterAlignment,
  splitMora,
  toKatakana,
  validateMoraUnits,
} from "../src/mora.mjs";
import {
  addNormalization,
  assertObservedIntegrity,
  createTranscriptRecord,
} from "../src/transcript-contract.mjs";
import {
  selectNormalizedCandidate,
  selectObservedCandidate,
} from "../src/asr-fusion.mjs";
import { validateRankOnlyResult } from "../src/local-lm-reranker.mjs";

test("normalizes hiragana and half-width kana", () => {
  assert.equal(toKatakana("きゃ ﾃｨ"), "キャ ティ");
});

test("treats contracted sounds as one mora", () => {
  assert.deepEqual(splitMora("きゃ").map(({ kana }) => kana), ["キャ"]);
  assert.equal(countMora("きゃく"), 2);
});

test("counts geminate, moraic nasal and long vowels independently", () => {
  assert.deepEqual(splitMora("がっこう").map(({ kana }) => kana), ["ガ", "ッ", "コ", "ウ"]);
  assert.deepEqual(splitMora("スーパー").map(({ kana }) => kana), ["ス", "ー", "パ", "ー"]);
  assert.equal(splitMora("しんぶん").filter(({ type }) => type === "moraic-nasal").length, 2);
});

test("merges character CTC spans into a timed mora", () => {
  const units = mergeCharacterAlignment([
    { char: "き", startMs: 10, endMs: 20, confidence: 0.9 },
    { char: "ゃ", startMs: 20, endMs: 30, confidence: 0.8 },
  ]);
  assert.equal(units.length, 1);
  assert.deepEqual(units[0], {
    index: 0,
    surface: "キャ",
    kana: "キャ",
    phones: null,
    type: "regular",
    charStart: null,
    charEnd: null,
    startMs: 10,
    endMs: 30,
    confidence: 0.8,
    source: "char-merge",
  });
  assert.equal(validateMoraUnits(units), true);
});

test("locks the observed transcript to acoustic evidence", () => {
  const record = createTranscriptRecord({
    audioSha256: "a".repeat(64),
    selectedCandidateId: "a",
    candidates: [
      { id: "a", text: "昨日学校を行きました", whisperScore: -1.0 },
      { id: "b", text: "昨日学校に行きました", whisperScore: -1.2 },
    ],
  });
  assert.equal(record.observedTranscript, "昨日学校を行きました");
  assert.equal(assertObservedIntegrity(record), true);
  assert.throws(() => {
    record.observedTranscript = "昨日学校に行きました";
  }, TypeError);
});

test("stores normalized text separately", () => {
  const record = createTranscriptRecord({
    selectedCandidateId: 0,
    candidates: [
      { id: 0, text: "学校を行きました" },
      { id: 1, text: "学校に行きました" },
    ],
  });
  const normalized = addNormalization(record, {
    text: "学校に行きました",
    selectedCandidateId: 1,
    model: "local-test",
  });
  assert.equal(normalized.observedTranscript, "学校を行きました");
  assert.equal(normalized.normalizedTranscript, "学校に行きました");
  assert.equal(assertObservedIntegrity(normalized), true);
});

test("observed selection ignores local language-model preference", () => {
  const candidates = [
    { id: 0, text: "学校を行きました", whisperScore: 0.9, moraCtcScore: 0.9, localLmScore: 0.1 },
    { id: 1, text: "学校に行きました", whisperScore: 0.2, moraCtcScore: 0.2, localLmScore: 1.0 },
  ];
  assert.equal(selectObservedCandidate(candidates).selected.id, 0);
  assert.equal(selectNormalizedCandidate(candidates, candidates[0].text).selected.id, 1);
});

test("rank-only validation rejects invented or duplicate candidates", () => {
  const candidates = [{ id: "a", text: "A" }, { id: "b", text: "B" }];
  assert.deepEqual(validateRankOnlyResult({ order: ["b", "a"] }, candidates).map(({ id }) => id), ["b", "a"]);
  assert.throws(() => validateRankOnlyResult({ order: ["a", "a"] }, candidates));
});
