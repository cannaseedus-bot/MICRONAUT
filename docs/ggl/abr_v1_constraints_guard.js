// ABR v1 constraints guard
// Provides static banlists, runtime guards, and verifier helpers.

export const ABR_BANNED_GLOBALS = Object.freeze([
  "Date",
  "performance",
  "Math.random",
  "fetch",
  "WebSocket",
  "caches",
  "indexedDB",
  "crypto",
  "setTimeout",
  "setInterval",
  "setImmediate",
  "Promise",
  "async function",
  "await",
]);

export const ABR_REQUIRED_CTX_FIELDS = Object.freeze([
  "tokens",
  "entropy",
  "tick",
  "policyHash",
  "clusterId",
]);

export const ABR_MASK_REASONS = Object.freeze({
  OK: "M0_OK",
  COLLAPSE_LOCKED: "M6_COLLAPSE_LOCKED",
});

export const ABR_PHASE_DOMAIN_ALLOW = Object.freeze({
  pre: ["control", "data", "auth"],
  compute: ["compute", "data"],
  post: ["proof", "audit"],
});

export const ABR_PHASE_LANE_ALLOW = Object.freeze({
  pre: ["DICT", "FIELD"],
  compute: ["LANE"],
  post: ["EDGE", "FIELD"],
});

const DEFAULT_NULLIFIERS = Object.freeze({
  Date: undefined,
  performance: undefined,
  fetch: undefined,
  WebSocket: undefined,
  caches: undefined,
  indexedDB: undefined,
  crypto: undefined,
  setTimeout: undefined,
  setInterval: undefined,
  setImmediate: undefined,
  Promise: undefined,
});

export function lintKernelSource(source, extraBans = []) {
  const bans = [...ABR_BANNED_GLOBALS, ...extraBans];
  const hits = [];
  for (const token of bans) {
    if (source.includes(token)) {
      hits.push(token);
    }
  }
  return hits;
}

export function assertKernelCtxShape(ctx) {
  const missing = ABR_REQUIRED_CTX_FIELDS.filter((field) => !(field in ctx));
  if (missing.length > 0) {
    throw new Error(`ABR kernel ctx missing fields: ${missing.join(", ")}`);
  }
}

export function withDeterminismGuard(fn, nullifiers = {}) {
  const overrides = { ...DEFAULT_NULLIFIERS, ...nullifiers };
  const original = {};
  for (const key of Object.keys(overrides)) {
    if (key in globalThis) {
      original[key] = globalThis[key];
    }
    globalThis[key] = overrides[key];
  }
  try {
    return fn();
  } finally {
    for (const key of Object.keys(overrides)) {
      if (key in original) {
        globalThis[key] = original[key];
      } else {
        delete globalThis[key];
      }
    }
  }
}

export function verifyPhaseBarrier({ phase, domain, lane, maskReason }) {
  const domainAllow = ABR_PHASE_DOMAIN_ALLOW[phase] || [];
  const laneAllow = ABR_PHASE_LANE_ALLOW[phase] || [];

  if (!domainAllow.includes(domain)) {
    return {
      ok: false,
      reason: `Domain ${domain} not allowed in phase ${phase}`,
    };
  }

  if (!laneAllow.includes(lane)) {
    return {
      ok: false,
      reason: `Lane ${lane} not allowed in phase ${phase}`,
    };
  }

  if (maskReason !== ABR_MASK_REASONS.OK) {
    return {
      ok: false,
      reason: `Mask reason not OK: ${maskReason}`,
    };
  }

  return { ok: true };
}

export function verifyCollapseLock({ postCollapse, maskReason }) {
  if (!postCollapse) {
    return { ok: true };
  }

  if (maskReason !== ABR_MASK_REASONS.COLLAPSE_LOCKED) {
    return {
      ok: false,
      reason: "Post-collapse barrier violated",
    };
  }

  return { ok: true };
}

export function verifyProofContract({ contract, expectedFields }) {
  const fields = Object.keys(contract).filter((key) => key !== "@type");
  const extras = fields.filter((field) => !expectedFields.includes(field));
  const missing = expectedFields.filter((field) => !(field in contract));

  if (extras.length > 0) {
    return { ok: false, reason: `Extra fields: ${extras.join(", ")}` };
  }

  if (missing.length > 0) {
    return { ok: false, reason: `Missing fields: ${missing.join(", ")}` };
  }

  return { ok: true };
}
