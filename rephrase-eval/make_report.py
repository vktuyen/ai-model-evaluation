#!/usr/bin/env python3
"""
Build a self-contained HTML report from promptfoo's results.json.

Why: the server is shell-only, so instead of tunnelling promptfoo's web UI,
we generate a single report.html you can open locally by double-clicking.
No server, no internet, no dependencies.

Mirrors the key promptfoo views: summary stats, a per-check scoreboard,
a pass/fail matrix (cases x models), search + filter, and full per-case
detail (input, each model's output, per-check pass/fail with reasons,
latency), plus token usage and cost.

Usage:
  python3 make_report.py                       # results.json -> report.html
  python3 make_report.py results.json out.html
"""
import json, sys, statistics
from collections import defaultdict

CHECKS = ["preserve_details", "uk_spelling", "output_format",
          "length_rule", "semantic_quality", "latency_guard"]

def load(path):
    d = json.load(open(path))
    return d["results"]["results"], d.get("evalId", "")

def dedupe(rows):
    seen, out = set(), []
    for r in rows:
        k = (r["provider"]["label"], r["testCase"].get("description"))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out

def cm(comp):
    return comp.get("assertion", {}).get("metric") or comp.get("metric")

def pctile(vals, q):
    if not vals:
        return 0
    s = sorted(vals)
    i = min(len(s) - 1, int(q * len(s)))
    return int(s[i])

def build(rows, eval_id):
    rows = dedupe(rows)
    by_prov = defaultdict(list)
    for r in rows:
        by_prov[r["provider"]["label"]].append(r)
    providers = list(by_prov.keys())

    summary = {}
    for p, rs in by_prov.items():
        counts = {c: [0, 0] for c in CHECKS}
        overall = 0
        toks = 0
        cost = 0.0
        for r in rs:
            allpass = True
            for comp in r["gradingResult"]["componentResults"]:
                m = cm(comp)
                if m in counts:
                    counts[m][1] += 1
                    if comp["pass"]:
                        counts[m][0] += 1
                    else:
                        allpass = False
            if allpass:
                overall += 1
            tu = r.get("tokenUsage") or {}
            toks += tu.get("total", 0) or 0
            cost += r.get("cost", 0) or 0
        lat = [r["latencyMs"] for r in rs if r.get("latencyMs")]
        summary[p] = {
            "n": len(rs), "overall": overall, "checks": counts,
            "median_latency": int(statistics.median(lat)) if lat else 0,
            "p95_latency": pctile(lat, 0.95),
            "tokens": toks, "cost": round(cost, 4),
        }

    cases = {}
    order = []
    for r in rows:
        desc = r["testCase"].get("description")
        if desc not in cases:
            order.append(desc)
            cases[desc] = {"_meta": {
                "note": r["vars"].get("note", ""),
                "tone": r["vars"].get("tone", ""),
                "type": r["vars"].get("type", ""),
            }}
        checks = {}
        for comp in r["gradingResult"]["componentResults"]:
            m = cm(comp)
            if m:
                checks[m] = {"pass": comp["pass"], "reason": comp.get("reason", "")}
        passed = sum(1 for c in CHECKS if checks.get(c, {}).get("pass"))
        total = sum(1 for c in CHECKS if c in checks)
        cases[desc][r["provider"]["label"]] = {
            "output": r["response"]["output"],
            "latency": r.get("latencyMs", 0),
            "checks": checks, "passed": passed, "total": total,
            "allpass": passed == total and total > 0,
        }
    return {
        "evalId": eval_id, "providers": providers, "checks": CHECKS,
        "summary": summary,
        "cases": [{"desc": d, **cases[d]} for d in order],
    }

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FSM Rephrase Eval Report</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--line:#2a2f3a;--fg:#e6e8ec;--mut:#9aa3b2;
--pass:#2ecc71;--fail:#e74c3c;--warn:#f1c40f;--accent:#4f8cff;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:26px}
h1{font-size:22px;margin:0 0 4px}h3{margin:0 0 12px}
.sub{color:var(--mut);margin-bottom:20px;font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:20px}
.cards{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}
.stat{flex:1;min-width:160px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.stat .k{color:var(--mut);font-size:12px}.stat .v{font-size:24px;font-weight:700;margin-top:2px}
.stat .m{color:var(--mut);font-size:12px;margin-top:2px}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px 11px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:13px}
.pill{display:inline-block;min-width:50px;text-align:center;padding:3px 8px;border-radius:999px;font-size:12px;font-weight:600}
.bar{height:7px;border-radius:6px;background:#242833;overflow:hidden;margin-top:5px}.bar>i{display:block;height:100%}
.win{color:var(--accent);font-weight:700}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;white-space:pre-wrap;word-break:break-word}
.cell{cursor:pointer;text-align:center;font-weight:600;border-radius:8px;padding:6px 4px;font-size:12px}
.cok{background:rgba(46,204,113,.16);color:var(--pass)}.cno{background:rgba(231,76,60,.16);color:var(--fail)}
.case{border:1px solid var(--line);border-radius:10px;margin-bottom:10px;overflow:hidden}
.case>summary{cursor:pointer;padding:11px 14px;list-style:none;display:flex;gap:9px;align-items:center;flex-wrap:wrap}
.case>summary::-webkit-details-marker{display:none}
.tag{font-size:11px;color:var(--mut);border:1px solid var(--line);border-radius:6px;padding:2px 7px}
.grid{padding:0 14px 14px}.out{background:#12151b;border:1px solid var(--line);border-radius:8px;padding:10px;margin:8px 0}
.chk{display:inline-block;font-size:11px;padding:2px 7px;border-radius:6px;margin:2px 4px 2px 0}
.ok{background:rgba(46,204,113,.15);color:var(--pass)}.no{background:rgba(231,76,60,.15);color:var(--fail)}
.model{font-weight:600;margin-top:10px}.reason{color:var(--mut);font-size:12px;margin-top:4px}
input,select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:7px 10px;font-size:13px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
</style></head><body>
<h1>FSM Rephrase — Evaluation Report</h1>
<div class="sub" id="sub"></div>
<div class="cards" id="statcards"></div>
<div class="card"><h3>Scoreboard — pass rate per check</h3><div id="board"></div></div>
<div class="card"><h3>Matrix — cases x models (click a cell to jump to detail)</h3><div style="overflow-x:auto" id="matrix"></div></div>
<div class="card">
  <div class="row">
    <h3 style="margin:0">Per-case detail</h3>
    <input id="search" placeholder="search cases / text…" style="flex:1;min-width:180px">
    <select id="filter"><option value="all">All cases</option><option value="fail">Only cases a model failed</option></select>
  </div>
  <div id="cases"></div>
</div>
<script>
const DATA = __DATA__;
const P=DATA.providers, C=DATA.checks, S=DATA.summary;
const pct=(a,b)=>b?Math.round(100*a/b):0;
const color=v=>v>=80?'var(--pass)':v>=50?'var(--warn)':'var(--fail)';
const esc=s=>(s||'').replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));
function pill(a,b){const p=pct(a,b);return `<span class="pill" style="background:${color(p)}22;color:${color(p)}">${p}%</span>`}

// best model by overall
let best=null,bo=-1; P.forEach(p=>{if(S[p].overall>bo){bo=S[p].overall;best=p}});
document.getElementById('sub').textContent=`${DATA.cases.length} cases × ${P.length} models · ${C.length} checks · eval ${DATA.evalId}`;

// summary cards (one per model)
document.getElementById('statcards').innerHTML=P.map(p=>{
  const s=S[p];return `<div class="stat">
   <div class="k">${p===best?'<span class=win>'+p+' ★</span>':p}</div>
   <div class="v">${pct(s.overall,s.n)}%</div>
   <div class="m">${s.overall}/${s.n} pass · ${s.median_latency}ms med · p95 ${s.p95_latency}ms<br>${s.tokens.toLocaleString()} tok · $${s.cost}</div></div>`}).join('');

// scoreboard
(function(){
  let h='<table><tr><th>Check</th>'+P.map(p=>`<th>${p===best?'<span class=win>'+p+' ★</span>':p}</th>`).join('')+'</tr>';
  C.forEach(c=>{h+=`<tr><td>${c}</td>`+P.map(p=>{const[a,b]=S[p].checks[c];
    return `<td>${pill(a,b)}<div class="bar"><i style="width:${pct(a,b)}%;background:${color(pct(a,b))}"></i></div></td>`}).join('')+'</tr>';});
  h+=`<tr><td><b>Overall (all pass)</b></td>`+P.map(p=>`<td><b>${S[p].overall}/${S[p].n}</b> ${pill(S[p].overall,S[p].n)}</td>`).join('')+'</tr>';
  h+=`<tr><td>Median / p95 latency</td>`+P.map(p=>`<td>${S[p].median_latency} / ${S[p].p95_latency} ms</td>`).join('')+'</tr>';
  document.getElementById('board').innerHTML=h+'</table>';
})();

// matrix
(function(){
  let h='<table><tr><th>Case</th>'+P.map(p=>`<th>${p}</th>`).join('')+'</tr>';
  DATA.cases.forEach((cs,i)=>{
    h+=`<tr><td style="min-width:220px">${esc(cs.desc)}<div class="tag" style="margin-top:3px">${cs._meta.tone} · ${cs._meta.type}</div></td>`;
    P.forEach(p=>{const d=cs[p];if(!d){h+='<td>–</td>';return;}
      const ok=d.allpass;h+=`<td><div class="cell ${ok?'cok':'cno'}" onclick="openCase(${i})">${ok?'✓':'✗'} ${d.passed}/${d.total}</div></td>`;});
    h+='</tr>';
  });
  document.getElementById('matrix').innerHTML=h+'</table>';
})();

function openCase(i){const el=document.getElementById('case'+i);
  if(el){document.getElementById('filter').value='all';document.getElementById('search').value='';renderCases();
    const e2=document.getElementById('case'+i);e2.open=true;e2.scrollIntoView({behavior:'smooth',block:'center'});}}

function renderCases(){
  const mode=document.getElementById('filter').value;
  const q=document.getElementById('search').value.toLowerCase();
  const box=document.getElementById('cases');box.innerHTML='';
  DATA.cases.forEach((cs,i)=>{
    const anyFail=P.some(p=>cs[p]&&!cs[p].allpass);
    if(mode==='fail'&&!anyFail)return;
    if(q){const hay=(cs.desc+' '+cs._meta.note+' '+P.map(p=>cs[p]?cs[p].output:'').join(' ')).toLowerCase();
      if(!hay.includes(q))return;}
    const m=cs._meta;let inner='';
    P.forEach(p=>{const d=cs[p];if(!d)return;
      const chks=C.filter(c=>d.checks[c]).map(c=>{const x=d.checks[c];
        return `<span class="chk ${x.pass?'ok':'no'}">${c} ${x.pass?'✓':'✗'}</span>`}).join('');
      const fails=C.filter(c=>d.checks[c]&&!d.checks[c].pass).map(c=>`${c}: ${esc(d.checks[c].reason)}`);
      inner+=`<div class="model">${p} <span class="tag">${d.latency} ms</span> <span class="tag">${d.passed}/${d.total}</span></div>
        <div class="out mono">${esc((d.output||'').slice(0,1200))}</div><div>${chks}</div>
        ${fails.length?`<div class="reason">${fails.join('<br>')}</div>`:''}`;});
    box.insertAdjacentHTML('beforeend',
      `<details class="case" id="case${i}"><summary><b>${esc(cs.desc)}</b>
        <span class="tag">${m.tone}</span><span class="tag">${m.type}</span>
        ${anyFail?'<span class="chk no">has failures</span>':'<span class="chk ok">all pass</span>'}</summary>
        <div class="grid"><div class="out mono"><b>INPUT:</b> ${esc(m.note)}</div>${inner}</div></details>`);
  });
}
document.getElementById('filter').addEventListener('change',renderCases);
document.getElementById('search').addEventListener('input',renderCases);
renderCases();
</script></body></html>"""

def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    outp = sys.argv[2] if len(sys.argv) > 2 else "report.html"
    rows, eval_id = load(inp)
    data = build(rows, eval_id)
    open(outp, "w").write(TEMPLATE.replace("__DATA__", json.dumps(data)))
    s = data["summary"]
    print(f"Wrote {outp}  ({len(data['cases'])} cases, {len(data['providers'])} models)")
    for p in data["providers"]:
        print(f"  {p:24} overall {s[p]['overall']}/{s[p]['n']}  "
              f"median {s[p]['median_latency']}ms  p95 {s[p]['p95_latency']}ms  "
              f"{s[p]['tokens']} tok  ${s[p]['cost']}")

if __name__ == "__main__":
    main()
