function assertLoopback(endpoint) {
  const url = new URL(endpoint);
  const allowed = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);
  if (!allowed.has(url.hostname)) {
    throw new Error("remote LLM endpoints are disabled by default");
  }
  return url;
}

function validateOrder(order, candidates) {
  if (!Array.isArray(order) || order.length !== candidates.length) {
    throw new Error("LLM ranking must contain every candidate exactly once");
  }
  const expected = new Set(candidates.map(({ id }) => String(id)));
  const actual = new Set(order.map(String));
  if (actual.size !== expected.size || [...expected].some((id) => !actual.has(id))) {
    throw new Error("LLM ranking contains an unknown, duplicate, or missing candidate id");
  }
  return order.map((id) => candidates.find((candidate) => String(candidate.id) === String(id)));
}

export async function rerankWithOllama({
  candidates,
  context = "",
  endpoint = "http://127.0.0.1:11434",
  model = "qwen2.5:3b-instruct",
  timeoutMs = 30_000,
  fetchImpl = globalThis.fetch,
}) {
  if (!Array.isArray(candidates) || candidates.length < 2) {
    throw new TypeError("at least two candidates are required");
  }
  if (typeof fetchImpl !== "function") throw new TypeError("fetch is unavailable");

  const base = assertLoopback(endpoint);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  const schema = {
    type: "object",
    additionalProperties: false,
    required: ["order"],
    properties: {
      order: {
        type: "array",
        minItems: candidates.length,
        maxItems: candidates.length,
        items: { type: ["string", "number"] },
      },
      rationale: { type: "string" },
    },
  };

  const prompt = [
    "You are a Japanese ASR N-best reranker.",
    "Rank only the supplied candidate IDs. Never write a new transcript.",
    "Prefer acoustic plausibility over grammatical correction.",
    context ? `Context: ${context}` : null,
    `Candidates: ${JSON.stringify(candidates.map(({ id, text }) => ({ id, text })))}`,
  ].filter(Boolean).join("\n");

  try {
    const response = await fetchImpl(new URL("/api/chat", base), {
      method: "POST",
      headers: { "content-type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        model,
        stream: false,
        format: schema,
        options: { temperature: 0 },
        messages: [{ role: "user", content: prompt }],
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama request failed: HTTP ${response.status}`);
    }

    const payload = await response.json();
    const content = payload?.message?.content;
    const parsed = typeof content === "string" ? JSON.parse(content) : content;
    const ranked = validateOrder(parsed?.order, candidates);

    return {
      model,
      mode: "rank-only",
      order: ranked.map(({ id }) => id),
      ranked,
      rationale: typeof parsed?.rationale === "string" ? parsed.rationale : null,
    };
  } finally {
    clearTimeout(timeout);
  }
}

export function validateRankOnlyResult(result, candidates) {
  return validateOrder(result?.order, candidates);
}
