// promptfoo prompt function.
// Reads the FSM rephrase system prompt once, and wraps each test's
// note/tone/type into the exact JSON input format the feature uses.
//
// It also suppresses chain-of-thought: gemma/qwen were dumping their
// "Thinking Process" into the output. We add an explicit no-reasoning
// directive (production-realistic), and for Qwen we prepend its native
// "/no_think" soft switch which disables the reasoning channel.
const fs = require('fs');
const path = require('path');

const SYSTEM_PROMPT = fs.readFileSync(
  path.join(__dirname, 'rephrase_system_prompt.txt'),
  'utf-8'
);

const NO_REASONING =
  '\n\nIMPORTANT: Respond with ONLY the final rephrased note as plain text. ' +
  'Do not include any reasoning, thinking, analysis, explanation, headings, ' +
  'or preamble — output the rephrased note and nothing else.';

module.exports = async function (context) {
  const vars = (context && context.vars) || {};
  const provider = (context && context.provider) || {};
  const pid = String(provider.id || provider.label || '').toLowerCase();

  // The user message must match production: {"note":..,"tone":..,"type":..}
  const userInput = JSON.stringify({
    note: vars.note,
    tone: vars.tone,
    type: vars.type,
  });

  let userContent = userInput + NO_REASONING;

  // Qwen3.x honours a "/no_think" soft switch to turn off its reasoning trace.
  if (pid.includes('qwen')) {
    userContent = '/no_think\n' + userContent;
  }

  return [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: userContent },
  ];
};
