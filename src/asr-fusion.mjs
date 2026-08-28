function finite(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function normalize(values) {
  const observed = values.filter((value) => value != null);
  if (observed.length === 0) return values.map(() => 0);
  const minimum = Math.min(...observed);
  const maximum = Math.max(...observed);
  if (minimum === maximum) return values.map((value) => (value == null ? 0 : 1));
  return values.map((value) => (value == null ? 0 : (value - minimum) / (maximum - minimum)));
}

function editDistance(left, right) {
  const a = [...String(left ?? "")];
  const b = [...String(right ?? "")];
  const row = Array.from({ length: b.length + 1 }, (_, index) => index);

  for (let i = 1; i <= a.length; i += 1) {
    let diagonal = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const previous = row[j];
      row[j] = Math.min(
        row[j] + 1,
        row[j - 1] + 1,
        diagonal + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
      diagonal = previous;
    }
  }
  return row.at(-1);
}

export function fuseCandidates(candidates, {
  whisper = 0.65,
  moraCtc = 0.25,
  localLm = 0,
  vocabulary = 0.1,
  editPenalty = 0,
  referenceText = null,
} = {}) {
  if (!Array.isArray(candidates) || candidates.length === 0) {
    throw new TypeError("candidates must be a non-empty array");
  }

  const whisperScores = normalize(candidates.map((item) => finite(item.whisperScore)));
  const moraScores = normalize(candidates.map((item) => finite(item.moraCtcScore)));
  const lmScores = normalize(candidates.map((item) => finite(item.localLmScore)));
  const vocabularyScores = normalize(candidates.map((item) => finite(item.vocabularyScore)));

  const ranked = candidates.map((candidate, index) => {
    const unsupportedEdits = referenceText == null
      ? 0
      : editDistance(candidate.text, referenceText) / Math.max(1, [...referenceText].length);
    const finalScore =
      whisper * whisperScores[index]
      + moraCtc * moraScores[index]
      + localLm * lmScores[index]
      + vocabulary * vocabularyScores[index]
      - editPenalty * unsupportedEdits;

    return {
      ...candidate,
      finalScore,
      scoreBreakdown: {
        whisper: whisperScores[index],
        moraCtc: moraScores[index],
        localLm: lmScores[index],
        vocabulary: vocabularyScores[index],
        unsupportedEdits,
      },
    };
  });

  return ranked.sort((left, right) =>
    right.finalScore - left.finalScore
      || String(left.id).localeCompare(String(right.id)),
  );
}

export function selectObservedCandidate(candidates, options = {}) {
  const ranked = fuseCandidates(candidates, {
    whisper: options.whisper ?? 0.7,
    moraCtc: options.moraCtc ?? 0.25,
    vocabulary: options.vocabulary ?? 0.05,
    localLm: 0,
    editPenalty: 0,
  });
  return {
    selected: ranked[0],
    ranked,
    margin: ranked.length > 1 ? ranked[0].finalScore - ranked[1].finalScore : 1,
  };
}

export function selectNormalizedCandidate(candidates, observedTranscript, options = {}) {
  // The normalized lane is derivative and may use a local-LM rank, while the
  // observed lane above remains strictly acoustic. These initialization weights
  // are intentionally separate and must be calibrated on held-out Japanese audio.
  const ranked = fuseCandidates(candidates, {
    whisper: options.whisper ?? 0.25,
    moraCtc: options.moraCtc ?? 0.15,
    localLm: options.localLm ?? 0.5,
    vocabulary: options.vocabulary ?? 0.1,
    editPenalty: options.editPenalty ?? 0.15,
    referenceText: observedTranscript,
  });
  return {
    selected: ranked[0],
    ranked,
    margin: ranked.length > 1 ? ranked[0].finalScore - ranked[1].finalScore : 1,
  };
}
