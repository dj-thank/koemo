const SMALL_KANA = new Set([
  "ァ", "ィ", "ゥ", "ェ", "ォ",
  "ャ", "ュ", "ョ", "ヮ", "ヵ", "ヶ",
]);

const PUNCTUATION = /[\s、。,.!?！？・「」『』（）()［］\[\]【】…‥:：;；]/u;
const KATAKANA = /[\u30A0-\u30FF]/u;

export function toKatakana(value) {
  return String(value ?? "")
    .normalize("NFKC")
    .replace(/[\u3041-\u3096]/gu, (char) =>
      String.fromCharCode(char.charCodeAt(0) + 0x60),
    );
}

export function classifyMora(kana) {
  if (kana === "ン") return "moraic-nasal";
  if (kana === "ッ") return "geminate";
  if (kana === "ー") return "long-vowel";
  return "regular";
}

function confidenceOf(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : null;
}

function combineConfidence(left, right) {
  if (left == null) return right;
  if (right == null) return left;
  return Math.min(left, right);
}

function canAttachSmallKana(previous) {
  return previous && previous.type === "regular";
}

export function splitMora(value, { includeUnknown = false } = {}) {
  const normalized = toKatakana(value);
  const units = [];
  let offset = 0;

  for (const char of normalized) {
    const start = offset;
    offset += char.length;

    if (PUNCTUATION.test(char)) continue;

    if (SMALL_KANA.has(char) && canAttachSmallKana(units.at(-1))) {
      const previous = units.at(-1);
      previous.surface += char;
      previous.kana += char;
      previous.charEnd = offset;
      continue;
    }

    if (!KATAKANA.test(char) && !includeUnknown) continue;

    units.push({
      index: units.length,
      surface: char,
      kana: char,
      phones: null,
      type: classifyMora(char),
      charStart: start,
      charEnd: offset,
      startMs: null,
      endMs: null,
      confidence: null,
      source: "text",
    });
  }

  return units;
}

export function countMora(value) {
  return splitMora(value).length;
}

export function mergeCharacterAlignment(characterUnits) {
  if (!Array.isArray(characterUnits)) {
    throw new TypeError("characterUnits must be an array");
  }

  const moraUnits = [];

  for (const characterUnit of characterUnits) {
    const raw = characterUnit?.char ?? characterUnit?.surface ?? characterUnit?.text;
    const normalized = toKatakana(raw);
    if (!normalized) continue;

    for (const char of normalized) {
      if (PUNCTUATION.test(char)) continue;
      if (!KATAKANA.test(char)) continue;

      const startMs = Number.isFinite(Number(characterUnit.startMs))
        ? Number(characterUnit.startMs)
        : null;
      const endMs = Number.isFinite(Number(characterUnit.endMs))
        ? Number(characterUnit.endMs)
        : null;
      const confidence = confidenceOf(characterUnit.confidence);

      if (SMALL_KANA.has(char) && canAttachSmallKana(moraUnits.at(-1))) {
        const previous = moraUnits.at(-1);
        previous.surface += char;
        previous.kana += char;
        if (previous.startMs == null) previous.startMs = startMs;
        if (endMs != null) previous.endMs = endMs;
        previous.confidence = combineConfidence(previous.confidence, confidence);
        continue;
      }

      moraUnits.push({
        index: moraUnits.length,
        surface: char,
        kana: char,
        phones: null,
        type: classifyMora(char),
        charStart: null,
        charEnd: null,
        startMs,
        endMs,
        confidence,
        source: "char-merge",
      });
    }
  }

  return moraUnits;
}

export function moraPerSecond(moraUnits, durationMs) {
  const duration = Number(durationMs);
  if (!Array.isArray(moraUnits) || !Number.isFinite(duration) || duration <= 0) {
    return null;
  }
  return moraUnits.length / (duration / 1000);
}

export function validateMoraUnits(moraUnits) {
  if (!Array.isArray(moraUnits)) return false;

  let previousEnd = -Infinity;
  for (let index = 0; index < moraUnits.length; index += 1) {
    const unit = moraUnits[index];
    if (!unit || unit.index !== index || typeof unit.kana !== "string") return false;
    if (!["regular", "moraic-nasal", "geminate", "long-vowel"].includes(unit.type)) {
      return false;
    }
    if (unit.startMs != null && unit.endMs != null) {
      if (!Number.isFinite(unit.startMs) || !Number.isFinite(unit.endMs)) return false;
      if (unit.startMs > unit.endMs || unit.startMs < previousEnd) return false;
      previousEnd = unit.endMs;
    }
  }

  return true;
}
