/**
 * Deterministic (code-based) checks for the FSM rephraser eval.
 *
 * Each function is a promptfoo `javascript` assertion:
 *   assert:
 *     - type: javascript
 *       value: file://checks.js:preserveTokens
 *
 * Signature: (output, context) => { pass, score, reason }
 *   - output       : the model's raw text output
 *   - context.vars : the test row vars (note, tone, type, preserve, ...)
 *
 * These catch the FSM-critical failures that DON'T need an LLM judge:
 * dropped numbers/dates/job-refs, US spelling, JSON leakage, length rules.
 */

'use strict';

function result(pass, reason) {
  return { pass, score: pass ? 1 : 0, reason };
}

function words(s) {
  return (s.match(/\b[\w'-]+\b/g) || []).length;
}

/**
 * Every string listed in vars.preserve must appear verbatim in the output.
 * Use for job numbers, prices, dates, proper names, references.
 */
function toTokenArray(raw) {
  if (Array.isArray(raw)) return raw;
  if (raw == null || raw === '') return [];
  if (typeof raw === 'string') {
    const s = raw.trim();
    // promptfoo may serialise a list var as a JSON string like '["a","b"]'
    if (s.startsWith('[')) {
      try {
        const parsed = JSON.parse(s);
        if (Array.isArray(parsed)) return parsed;
      } catch (_) {
        /* fall through to comma split */
      }
    }
    return s.split(',').map((x) => x.trim()).filter(Boolean);
  }
  return [raw];
}

function preserveTokens(output, context) {
  const tokens = toTokenArray(context.vars && context.vars.preserve);
  if (!tokens.length) return result(true, 'No tokens required to preserve.');
  const missing = tokens.filter((t) => !output.includes(String(t)));
  if (missing.length) {
    return result(false, `Dropped/altered required token(s): ${missing.join(', ')}`);
  }
  return result(true, `All ${tokens.length} required token(s) preserved.`);
}

/**
 * Flags common US spellings. Uses a curated map to avoid false positives
 * (e.g. "size", "prize" are NOT flagged).
 */
const US_TO_UK = {
  organize: 'organise', organized: 'organised', organizing: 'organising',
  organization: 'organisation',
  categorize: 'categorise', categorized: 'categorised',
  prioritize: 'prioritise', prioritized: 'prioritised',
  customize: 'customise', customized: 'customised',
  authorize: 'authorise', authorized: 'authorised',
  apologize: 'apologise', apologized: 'apologised',
  recognize: 'recognise', recognized: 'recognised',
  analyze: 'analyse', analyzed: 'analysed',
  minimize: 'minimise', maximize: 'maximise', optimize: 'optimise',
  finalize: 'finalise', finalized: 'finalised',
  utilize: 'utilise', utilized: 'utilised',
  color: 'colour', colour: null, favor: 'favour', favorite: 'favourite',
  behavior: 'behaviour', labor: 'labour', honor: 'honour',
  center: 'centre', centered: 'centred', meter: 'metre', liter: 'litre',
  fiber: 'fibre', theater: 'theatre',
  catalog: 'catalogue', dialog: 'dialogue',
  fulfill: 'fulfil', enrollment: 'enrolment', installment: 'instalment',
  traveling: 'travelling', traveled: 'travelled', canceled: 'cancelled',
  canceling: 'cancelling', labeled: 'labelled', modeling: 'modelling',
  program: null, // ambiguous, don't flag
  gray: 'grey', check: null, // "check" is valid in UK for inspection
};

function ukSpelling(output, context) {
  const lower = output.toLowerCase();
  const offenders = [];
  for (const [us, uk] of Object.entries(US_TO_UK)) {
    if (!uk) continue; // null = intentionally not flagged
    const re = new RegExp(`\\b${us}\\b`, 'i');
    if (re.test(lower)) offenders.push(`${us} -> ${uk}`);
  }
  if (offenders.length) {
    return result(false, `US spelling found: ${offenders.join('; ')}`);
  }
  return result(true, 'No US spelling detected.');
}

/**
 * Output must be plain text: not a JSON object, and must not echo the
 * tone/type control keys. (Rule 3 & 2 of the system prompt.)
 */
function plainTextNotJson(output, context) {
  const trimmed = output.trim();
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      JSON.parse(trimmed);
      return result(false, 'Output is JSON; expected plain text.');
    } catch (_) {
      /* not valid JSON, fall through */
    }
  }
  const leaked = /"?(tone|type)"?\s*[:=]/i.test(trimmed);
  if (leaked) {
    return result(false, 'Output appears to echo the tone/type control keys.');
  }
  return result(true, 'Plain-text output, no JSON or control-key leakage.');
}

/**
 * Length rule keyed off vars.type, compared against vars.note:
 *   Make Shorter -> must be shorter than input
 *   Make Longer  -> must be longer than input
 *   Default/other -> must NOT be a summary (>= 60% of input length)
 */
function lengthRule(output, context) {
  const note = (context.vars && context.vars.note) || '';
  const type = ((context.vars && context.vars.type) || '').toLowerCase();
  const inW = words(note);
  const outW = words(output);
  if (inW === 0) return result(true, 'Empty input; length rule skipped.');

  if (type.includes('shorter')) {
    return outW < inW
      ? result(true, `Shorter as required (${outW} < ${inW} words).`)
      : result(false, `Expected shorter, got ${outW} >= ${inW} words.`);
  }
  if (type.includes('longer')) {
    return outW > inW
      ? result(true, `Longer as required (${outW} > ${inW} words).`)
      : result(false, `Expected longer, got ${outW} <= ${inW} words.`);
  }
  // Default: must preserve, not summarise.
  const ratio = outW / inW;
  return ratio >= 0.6
    ? result(true, `Length preserved (ratio ${ratio.toFixed(2)}).`)
    : result(false, `Output too short for Default (ratio ${ratio.toFixed(2)} < 0.60) — looks summarised.`);
}

module.exports = { preserveTokens, ukSpelling, plainTextNotJson, lengthRule };
