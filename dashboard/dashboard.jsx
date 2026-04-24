import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, BarChart, Bar, Cell
} from "recharts";

// ─── Config ───────────────────────────────────────────────────────────────────
const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ─── Colour palette ───────────────────────────────────────────────────────────
const C = {
  bg:       "#0a0a0f",
  surface:  "#111118",
  border:   "#1e1e2e",
  accent:   "#7c6aff",
  accent2:  "#ff6a6a",
  accent3:  "#6affd4",
  muted:    "#4a4a6a",
  text:     "#e8e6ff",
  textDim:  "#7a7a9a",
  exp: ["#7c6aff","#6affd4","#ff6a6a","#ffd96a"],
};

// ─── Hardcoded experiment data (matches API /experiments fallback) ─────────────
const EXPERIMENTS = [
  {
    id:"exp1", label:"Baseline 0.79M", color: C.exp[0],
    params:"0.79M", perplexity:561.06, train_time:"~3 min",
    key_finding:"Stable baseline at this scale",
    curve:[],
  },
  {
    id:"exp2", label:"Larger 4.72M", color: C.exp[1],
    params:"4.72M", perplexity:1624.76, train_time:"~6 min",
    key_finding:"Larger model overfit under current budget",
    curve:[],
  },
  {
    id:"exp3", label:"No Grad Clip", color: C.exp[2],
    params:"0.79M", perplexity:579.88, train_time:"~3 min",
    key_finding:"Similar to clipped run in this setup",
    curve:[],
  },
];

const LR_CURVES = {
  "1e-3":  [{step:5000,val:7.4326}],
  "3e-4":  [{step:5000,val:6.3949}],
  "1e-4":  [{step:5000,val:6.5666}],
};

const ARCH_PARAMS = [
  { name:"tok_emb",   params:1024000, pct:56.0, color:C.accent },
  { name:"blocks[0]", params:197632,  pct:10.8, color:C.accent2 },
  { name:"blocks[1]", params:197632,  pct:10.8, color:"#ff9f6a" },
  { name:"blocks[2]", params:197632,  pct:10.8, color:C.accent3 },
  { name:"blocks[3]", params:197632,  pct:10.8, color:"#6ab8ff" },
  { name:"norm_final",params:128,     pct:0.007,color:C.muted   },
];

const LORA_DATA = {
  base_params:1827968, lora_params:16384, lora_pct:0.8963,
  reduction_factor:111.57, adapter_kb:64.0, rank:4,
};

const EVAL_DATA = {
  baseline: {
    aggregate: { mean_perplexity: 990.2874, std_perplexity: 4.3012, mean_val_loss: 6.8980, std_val_loss: 0.0043 },
    per_seed: [
      { seed: 42, perplexity: 986.5365 },
      { seed: 123, perplexity: 996.3099 },
      { seed: 999, perplexity: 988.0159 },
    ],
  },
  lora: {
    aggregate: { mean_perplexity: 962.5481, std_perplexity: 1.3378, mean_val_loss: 6.8696, std_val_loss: 0.0014 },
    per_seed: [
      { seed: 42, perplexity: 961.6133 },
      { seed: 123, perplexity: 961.5910 },
      { seed: 999, perplexity: 964.4401 },
    ],
  },
  delta: { perplexity_mean_delta: -27.7393, val_loss_mean_delta: -0.0284 },
};

const RANK_SWEEP_DATA = {
  best_rank: { rank: 16, mean_perplexity: 923.0597, trainable_pct: 3.4913 },
  ranks: [
    { rank: 1, mean_perplexity: 101026.3604, std_perplexity: 786.3998, trainable_pct: 0.2256, decision: "Very poor quality despite high efficiency." },
    { rank: 2, mean_perplexity: 4621.5574, std_perplexity: 6.4286, trainable_pct: 0.4502, decision: "Large quality drop; not practical." },
    { rank: 4, mean_perplexity: 962.5481, std_perplexity: 1.3378, trainable_pct: 0.8963, decision: "Strong baseline efficiency point." },
    { rank: 8, mean_perplexity: 925.6619, std_perplexity: 1.8594, trainable_pct: 1.7767, decision: "Near-best quality with better efficiency." },
    { rank: 16, mean_perplexity: 923.0597, std_perplexity: 2.5692, trainable_pct: 3.4913, decision: "Best quality in sweep." },
  ],
};

const PROMPT_BENCHMARK_DATA = {
  rank4_default: { label: "Rank-4 default", counts: { baseline_wins: 21, lora_wins: 1, ties: 8, lora_win_rate_on_decisive: 0.0455 } },
  rank8_t06_k20: { label: "Rank-8 t=0.6 k=20", counts: { baseline_wins: 19, lora_wins: 3, ties: 8, lora_win_rate_on_decisive: 0.1364 } },
  rank16_t06_k20: { label: "Rank-16 t=0.6 k=20", counts: { baseline_wins: 15, lora_wins: 9, ties: 6, lora_win_rate_on_decisive: 0.3750 } },
};

function formatParams(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function buildExperimentsView(payload) {
  if (!payload || !payload.data) {
    return {
      experiments: EXPERIMENTS,
      lrCurves: LR_CURVES,
      lora: LORA_DATA,
      evaluation: EVAL_DATA,
      rankSweep: RANK_SWEEP_DATA,
      promptBenchmarks: PROMPT_BENCHMARK_DATA,
      summaryRows: [
        {exp:"Baseline 0.79M", params:"0.79M", ppl:561.06,  time:"~3 min", finding:"Stable baseline at this scale"},
        {exp:"Larger 4.72M",   params:"4.72M", ppl:1624.76, time:"~6 min", finding:"Overfit under fixed budget"},
        {exp:"No Grad Clip",   params:"0.79M", ppl:579.88,  time:"~3 min", finding:"Similar to clipped run"},
        {exp:"Best LR",        params:"0.79M", ppl:598.81,  time:"~3 min", finding:"3e-4 best among tested LRs"},
        {exp:"LoRA FT",       params:"16K",  ppl:219.7, time:"~5 min", finding:"0.9% params, domain shift"},
      ],
    };
  }

  const data = payload.data;

  if (Array.isArray(data.experiments)) {
    const experiments = data.experiments.map((e, i) => ({
      id: e.id || `exp${i + 1}`,
      label: e.label || e.id || `exp${i + 1}`,
      color: C.exp[i % C.exp.length],
      params: e.params ? formatParams(e.params) : "-",
      perplexity: e.final_perplexity ?? "-",
      train_time: e.train_time_min ? `${e.train_time_min} min` : (e.train_time || "-"),
      key_finding: e.key_finding || "-",
      curve: (e.loss_curve || []).map((p) => ({ step: p.step, train: p.train_loss, val: p.val_loss })),
    }));
    const rawLrCurves = data.experiments.find((e) => e.id === "exp4_lr_sweep")?.lr_curves || LR_CURVES;
    const lrCurves = Object.fromEntries(
      Object.entries(rawLrCurves).map(([lr, curve]) => [
        lr,
        (curve || []).map((p) => ({ step: p.step, val: p.val ?? p.val_loss })),
      ])
    );
    const lora = data.lora || LORA_DATA;
    const evaluation = data.evaluation || EVAL_DATA;
    const rankSweep = data.lora_rank_sweep || RANK_SWEEP_DATA;
    const promptBenchmarks = data.prompt_benchmarks || PROMPT_BENCHMARK_DATA;
    const summaryRows = (data.summary_table || []).map((r) => ({
      exp: r.experiment, params: r.params, ppl: r.perplexity, time: r.train_time, finding: r.key_finding,
    }));
    return { experiments, lrCurves, lora, evaluation, rankSweep, promptBenchmarks, summaryRows };
  }

  return {
    experiments: EXPERIMENTS,
    lrCurves: LR_CURVES,
    lora: LORA_DATA,
    evaluation: EVAL_DATA,
    rankSweep: RANK_SWEEP_DATA,
    promptBenchmarks: PROMPT_BENCHMARK_DATA,
    summaryRows: [],
  };
}

// ─── Tiny components ──────────────────────────────────────────────────────────

function Pill({ children, active, onClick, color }) {
  return (
    <button onClick={onClick} style={{
      padding:"6px 14px", borderRadius:20, border:"1px solid",
      borderColor: active ? (color||C.accent) : C.border,
      background:  active ? `${color||C.accent}22` : "transparent",
      color:       active ? (color||C.accent) : C.textDim,
      cursor:"pointer", fontSize:12, fontFamily:"inherit",
      transition:"all 0.15s",
    }}>{children}</button>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{
      background:C.surface, border:`1px solid ${C.border}`, borderRadius:12,
      padding:"18px 22px",
    }}>
      <div style={{fontSize:11,color:C.textDim,textTransform:"uppercase",letterSpacing:2,marginBottom:6}}>
        {label}
      </div>
      <div style={{fontSize:28,fontWeight:700,color:color||C.text,fontFamily:"'DM Mono',monospace"}}>
        {value}
      </div>
      {sub && <div style={{fontSize:12,color:C.textDim,marginTop:4}}>{sub}</div>}
    </div>
  );
}

// ─── Section 1: Generation Playground ─────────────────────────────────────────

function Playground() {
  const [prompt,      setPrompt]      = useState("To be, or not to be, that is the question:");
  const [output,      setOutput]      = useState("");
  const [streaming,   setStreaming]   = useState(false);
  const [strategy,    setStrategy]    = useState("top_p");
  const [temperature, setTemperature] = useState(0.8);
  const [topK,        setTopK]        = useState(40);
  const [topP,        setTopP]        = useState(0.9);
  const [maxNew,      setMaxNew]      = useState(100);
  const [meta,        setMeta]        = useState(null);
  const [error,       setError]       = useState("");
  const abortRef = useRef(null);

  const strategies = ["greedy","temperature","top_k","top_p"];

  const generate = useCallback(async () => {
    setOutput(""); setError(""); setMeta(null); setStreaming(true);
    const params = new URLSearchParams({
      prompt, max_new:maxNew, strategy, temperature, top_k:topK, top_p:topP
    });
    const url = `${API}/generate/stream?${params}`;
    try {
      const es = new EventSource(url);
      abortRef.current = es;
      let full = "";
      es.onmessage = (e) => {
        const d = JSON.parse(e.data);
        if (d.error) { setError(d.error); es.close(); setStreaming(false); return; }
        if (d.done)  { setMeta(d.meta); es.close(); setStreaming(false); return; }
        full += d.token;
        setOutput(full);
      };
      es.onerror = () => {
        // SSE not available (CORS/no server) — fall back to REST
        es.close();
        fallbackRest();
      };
    } catch {
      fallbackRest();
    }
  }, [prompt, maxNew, strategy, temperature, topK, topP]);

  const fallbackRest = async () => {
    try {
      const r = await fetch(`${API}/generate`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({prompt, max_new:maxNew, strategy, temperature, top_k:topK, top_p:topP}),
      });
      if (!r.ok) throw new Error(await r.text());
      const d = await r.json();
      setOutput(d.generated_text);
      setMeta({ tokens_generated:d.tokens_generated, tokens_per_second:d.tokens_per_second, elapsed_seconds:d.elapsed_seconds });
    } catch(e) {
      setError(`API error: ${e.message}. Make sure the FastAPI server is running on :8000`);
    } finally {
      setStreaming(false);
    }
  };

  const stop = () => { abortRef.current?.close(); setStreaming(false); };

  return (
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
      {/* Left: controls */}
      <div style={{display:"flex",flexDirection:"column",gap:16}}>
        <label style={{color:C.textDim,fontSize:12,textTransform:"uppercase",letterSpacing:1.5}}>
          Prompt
        </label>
        <textarea
          value={prompt} onChange={e=>setPrompt(e.target.value)}
          rows={5}
          style={{
            background:C.surface, border:`1px solid ${C.border}`, borderRadius:10,
            color:C.text, padding:"14px 16px", fontSize:14, resize:"vertical",
            fontFamily:"'DM Mono',monospace", lineHeight:1.6,
            outline:"none",
          }}
        />

        {/* Strategy pills */}
        <div>
          <div style={{color:C.textDim,fontSize:11,textTransform:"uppercase",letterSpacing:1.5,marginBottom:8}}>
            Decoding Strategy
          </div>
          <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
            {strategies.map(s=>(
              <Pill key={s} active={strategy===s} onClick={()=>setStrategy(s)}>
                {s}
              </Pill>
            ))}
          </div>
        </div>

        {/* Sliders */}
        {[
          {label:"Temperature", key:"temperature", value:temperature, setter:setTemperature, min:0.1, max:2.0, step:0.05, show:strategy!=="greedy"},
          {label:`Top-k  (k=${topK})`, key:"topk", value:topK, setter:setTopK, min:1, max:200, step:1, show:strategy==="top_k"},
          {label:`Top-p  (p=${topP})`, key:"topp", value:topP, setter:setTopP, min:0.1, max:1.0, step:0.05, show:strategy==="top_p"},
          {label:`Max tokens  (${maxNew})`, key:"maxnew", value:maxNew, setter:setMaxNew, min:20, max:400, step:10, show:true},
        ].filter(s=>s.show).map(s=>(
          <div key={s.key}>
            <div style={{color:C.textDim,fontSize:11,marginBottom:6}}>{s.label}</div>
            <input type="range" min={s.min} max={s.max} step={s.step}
              value={s.value} onChange={e=>s.setter(Number(e.target.value))}
              style={{width:"100%",accentColor:C.accent}}
            />
          </div>
        ))}

        <div style={{display:"flex",gap:10}}>
          <button onClick={generate} disabled={streaming} style={{
            flex:1, padding:"12px 0", borderRadius:10,
            background: streaming ? C.muted : C.accent,
            color:"#fff", border:"none", cursor: streaming ? "not-allowed":"pointer",
            fontSize:14, fontWeight:600, fontFamily:"inherit",
            transition:"background 0.2s",
          }}>
            {streaming ? "Generating…" : "Generate →"}
          </button>
          {streaming && (
            <button onClick={stop} style={{
              padding:"12px 18px", borderRadius:10, background:"transparent",
              border:`1px solid ${C.accent2}`, color:C.accent2,
              cursor:"pointer", fontSize:14, fontFamily:"inherit",
            }}>Stop</button>
          )}
        </div>
      </div>

      {/* Right: output */}
      <div style={{display:"flex",flexDirection:"column",gap:12}}>
        <div style={{color:C.textDim,fontSize:12,textTransform:"uppercase",letterSpacing:1.5}}>
          Model Output
        </div>
        <div style={{
          background:C.surface, border:`1px solid ${streaming ? C.accent : C.border}`,
          borderRadius:10, padding:"16px 18px", minHeight:220,
          fontFamily:"'DM Mono',monospace", fontSize:13.5, color:C.text,
          lineHeight:1.75, whiteSpace:"pre-wrap", wordBreak:"break-word",
          transition:"border-color 0.3s", flex:1,
          overflowY:"auto",
        }}>
          {output || <span style={{color:C.muted}}>Generated text will appear here…</span>}
          {streaming && <span style={{
            display:"inline-block", width:2, height:"1em",
            background:C.accent, marginLeft:2, verticalAlign:"middle",
            animation:"blink 0.7s step-end infinite",
          }} />}
        </div>

        {error && (
          <div style={{
            background:`${C.accent2}18`, border:`1px solid ${C.accent2}44`,
            borderRadius:8, padding:"10px 14px", color:C.accent2, fontSize:12,
          }}>{error}</div>
        )}

        {meta && (
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:10}}>
            {[
              {label:"Tokens",  value:meta.tokens_generated},
              {label:"Tok/sec", value:meta.tokens_per_second?.toFixed(1)},
              {label:"Elapsed", value:`${meta.elapsed_seconds?.toFixed(2)}s`},
            ].map(m=>(
              <div key={m.label} style={{
                background:`${C.accent}11`, border:`1px solid ${C.accent}33`,
                borderRadius:8, padding:"10px 12px", textAlign:"center",
              }}>
                <div style={{fontSize:18,fontWeight:700,color:C.accent,fontFamily:"'DM Mono',monospace"}}>
                  {m.value}
                </div>
                <div style={{fontSize:10,color:C.textDim,textTransform:"uppercase",letterSpacing:1}}>
                  {m.label}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <style>{`@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}`}</style>
    </div>
  );
}

// ─── Section 2: Experiment Results ────────────────────────────────────────────

const TOOLTIP_STYLE = {
  background:C.surface, border:`1px solid ${C.border}`,
  borderRadius:8, fontSize:12, color:C.text,
};

function ExperimentResults() {
  const [view,    setView]    = useState("loss_curves");
  const [activeExp, setActiveExp] = useState(new Set(["exp1","exp2","exp3"]));
  const [expPayload, setExpPayload] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API}/experiments`);
        const d = await r.json();
        setExpPayload(d);
      } catch {
        setExpPayload(null);
      }
    })();
  }, []);

  const viewData = buildExperimentsView(expPayload);
  const experiments = viewData.experiments;
  const lrCurvesSource = viewData.lrCurves;
  const loraData = viewData.lora;
  const evalData = viewData.evaluation || EVAL_DATA;
  const rankSweep = viewData.rankSweep || RANK_SWEEP_DATA;
  const promptBenchmarks = viewData.promptBenchmarks || PROMPT_BENCHMARK_DATA;
  const summaryRows = viewData.summaryRows.length
    ? viewData.summaryRows
    : [
      {exp:"Baseline 0.79M", params:"0.79M", ppl:561.06,  time:"~3 min", finding:"Stable baseline at this scale"},
      {exp:"Larger 4.72M",   params:"4.72M", ppl:1624.76, time:"~6 min", finding:"Overfit under fixed budget"},
      {exp:"No Grad Clip",   params:"0.79M", ppl:579.88,  time:"~3 min", finding:"Similar to clipped run"},
      {exp:"Best LR",        params:"0.79M", ppl:598.81,  time:"~3 min", finding:"3e-4 best among tested LRs"},
      {exp:"LoRA FT",       params:"16K",  ppl:219.7, time:"~5 min", finding:"0.9% params, domain shift"},
    ];

  const baselineAgg = evalData?.baseline?.aggregate || EVAL_DATA.baseline.aggregate;
  const loraAgg = evalData?.lora?.aggregate || EVAL_DATA.lora.aggregate;
  const evalDelta = evalData?.delta || EVAL_DATA.delta;

  const seedRows = (() => {
    const base = evalData?.baseline?.per_seed || [];
    const lora = evalData?.lora?.per_seed || [];
    const seeds = Array.from(new Set([...base.map((s) => s.seed), ...lora.map((s) => s.seed)])).sort((a, b) => a - b);
    return seeds.map((seed) => {
      const b = base.find((x) => x.seed === seed);
      const l = lora.find((x) => x.seed === seed);
      return {
        seed: `seed ${seed}`,
        baseline_ppl: b?.perplexity ?? null,
        lora_ppl: l?.perplexity ?? null,
      };
    });
  })();

  const rankRows = (rankSweep?.ranks || []).map((r) => ({
    rank: `r=${r.rank}`,
    rankValue: r.rank,
    mean_ppl: r.mean_perplexity,
    std_ppl: r.std_perplexity,
    trainable_pct: r.trainable_pct,
    decision: r.decision,
  }));

  const bestRank = rankSweep?.best_rank || RANK_SWEEP_DATA.best_rank;
  const promptRows = Object.entries(promptBenchmarks).map(([key, value]) => ({
    key,
    label: value.label || key,
    baseline_wins: value.counts?.baseline_wins ?? 0,
    lora_wins: value.counts?.lora_wins ?? 0,
    ties: value.counts?.ties ?? 0,
    win_rate: value.counts?.lora_win_rate_on_decisive ?? 0,
  }));
  const bestPromptRun = [...promptRows].sort((a, b) => b.win_rate - a.win_rate)[0];

  const toggleExp = (id) => {
    setActiveExp(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  // Merge all curves onto common step axis for multi-line chart
  const mergedCurves = (() => {
    const steps = new Set();
    experiments.forEach(e => e.curve?.forEach(p => steps.add(p.step)));
    return [...steps].sort((a,b)=>a-b).map(step => {
      const row = {step};
      experiments.forEach(e => {
        const pt = e.curve?.find(p=>p.step===step);
        if (pt) {
          row[`${e.id}_train`] = pt.train;
          row[`${e.id}_val`]   = pt.val;
        }
      });
      return row;
    });
  })();

  const finalExperimentRows = experiments
    .filter((e) => Number.isFinite(Number(e.perplexity)))
    .map((e) => ({
      experiment: e.label,
      perplexity: Number(e.perplexity),
    }));

  const hasLossCurvePoints = mergedCurves.length > 1;

  const lrMerged = (() => {
    const steps = new Set();
    Object.values(lrCurvesSource).forEach(c => c.forEach(p => steps.add(p.step)));
    return [...steps].sort((a,b)=>a-b).map(step => {
      const row = {step};
      Object.entries(lrCurvesSource).forEach(([lr, curve]) => {
        const pt = curve.find(p=>p.step===step);
        if (pt) row[lr] = pt.val;
      });
      return row;
    });
  })();

  const lrBarData = Object.entries(lrCurvesSource).map(([lr, curve]) => {
    const last = [...curve].sort((a, b) => (a.step ?? 0) - (b.step ?? 0)).slice(-1)[0];
    return { lr, val_loss: last?.val ?? null };
  }).filter((x) => Number.isFinite(Number(x.val_loss)));
  const hasLrCurveLines = Object.values(lrCurvesSource).some((curve) => (curve?.length || 0) > 1);

  return (
    <div style={{display:"flex",flexDirection:"column",gap:20}}>
      {/* View selector */}
      <div style={{display:"flex",gap:8}}>
        {[
          {id:"loss_curves", label:"Loss Curves"},
          {id:"lr_sweep",    label:"LR Sweep"},
          {id:"summary",     label:"Summary Table"},
          {id:"lora",        label:"LoRA Efficiency"},
          {id:"heldout_eval",label:"Held-out Eval"},
          {id:"rank_sweep",  label:"Rank Sweep"},
          {id:"prompt_benchmark", label:"Prompt Benchmark"},
        ].map(v=>(
          <Pill key={v.id} active={view===v.id} onClick={()=>setView(v.id)}>
            {v.label}
          </Pill>
        ))}
      </div>

      {/* Loss curves */}
      {view==="loss_curves" && (
        <div style={{display:"flex",flexDirection:"column",gap:14}}>
          <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
            {experiments.filter(e => (e.curve?.length || 0) > 0).map(e=>(
              <Pill key={e.id} active={activeExp.has(e.id)}
                onClick={()=>toggleExp(e.id)} color={e.color}>
                {e.label}
              </Pill>
            ))}
          </div>
          {hasLossCurvePoints ? (
            <>
              <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"20px 10px 10px"}}>
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={mergedCurves} margin={{right:20}}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                    <XAxis dataKey="step" stroke={C.muted} tick={{fontSize:11}} />
                    <YAxis stroke={C.muted} tick={{fontSize:11}} domain={[0,"auto"]} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Legend wrapperStyle={{fontSize:11,color:C.textDim}} />
                    {experiments.filter(e=>activeExp.has(e.id)).map(e=>[
                      <Line key={`${e.id}_val`} type="monotone" dataKey={`${e.id}_val`}
                        name={`${e.label} val`} stroke={e.color}
                        strokeWidth={2} dot={false} connectNulls={false} />,
                      <Line key={`${e.id}_train`} type="monotone" dataKey={`${e.id}_train`}
                        name={`${e.label} train`} stroke={e.color}
                        strokeWidth={1.5} strokeDasharray="4 3" dot={false} connectNulls={false} />,
                    ])}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div style={{fontSize:11,color:C.textDim,textAlign:"center"}}>
                Solid = validation loss · Dashed = training loss
              </div>
            </>
          ) : (
            <>
              <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"20px 10px 10px"}}>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={finalExperimentRows}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                    <XAxis dataKey="experiment" stroke={C.muted} tick={{fontSize:11}} />
                    <YAxis stroke={C.muted} tick={{fontSize:11}} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Legend wrapperStyle={{fontSize:11,color:C.textDim}} />
                    <Bar dataKey="perplexity" name="Final Perplexity" fill={C.accent3} radius={[5,5,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div style={{fontSize:11,color:C.textDim,textAlign:"center"}}>
                Curve data is unavailable in current artifact set; showing final perplexity by experiment.
              </div>
            </>
          )}
        </div>
      )}

      {/* LR sweep */}
      {view==="lr_sweep" && (
        <div style={{display:"flex",flexDirection:"column",gap:14}}>
          <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"20px 10px 10px"}}>
            <ResponsiveContainer width="100%" height={320}>
              {hasLrCurveLines ? (
                <LineChart data={lrMerged}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="step" stroke={C.muted} tick={{fontSize:11}} />
                  <YAxis stroke={C.muted} tick={{fontSize:11}} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{fontSize:11,color:C.textDim}} />
                  <Line type="monotone" dataKey="1e-3" name="lr=1e-3 (too aggressive)" stroke={C.accent2} strokeWidth={2} dot={false} connectNulls={false}/>
                  <Line type="monotone" dataKey="3e-4" name="lr=3e-4 (best in sweep)" stroke={C.accent3} strokeWidth={2.5} dot={false}/>
                  <Line type="monotone" dataKey="1e-4" name="lr=1e-4 (too slow)" stroke={C.muted} strokeWidth={2} dot={false}/>
                </LineChart>
              ) : (
                <BarChart data={lrBarData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="lr" stroke={C.muted} tick={{fontSize:11}} />
                  <YAxis stroke={C.muted} tick={{fontSize:11}} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{fontSize:11,color:C.textDim}} />
                  <Bar dataKey="val_loss" name="Final Val Loss" fill={C.accent} radius={[5,5,0,0]} />
                </BarChart>
              )}
            </ResponsiveContainer>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12}}>
            {[{lr:"1e-3",c:C.accent2,note:"Too aggressive — ppl 1690.17 at step 5000"},{lr:"3e-4",c:C.accent3,note:"Best in sweep — ppl 598.81 at step 5000"},{lr:"1e-4",c:C.muted,note:"Too slow — ppl 710.95 at step 5000"}].map(r=>(
              <div key={r.lr} style={{background:C.surface,border:`1px solid ${r.c}44`,borderRadius:10,padding:"14px 16px"}}>
                <div style={{fontSize:20,fontWeight:700,color:r.c,fontFamily:"'DM Mono',monospace"}}>lr = {r.lr}</div>
                <div style={{fontSize:12,color:C.textDim,marginTop:4}}>{r.note}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary table */}
      {view==="summary" && (
        <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,overflow:"hidden"}}>
          <table style={{width:"100%",borderCollapse:"collapse",fontFamily:"'DM Mono',monospace"}}>
            <thead>
              <tr style={{background:`${C.accent}18`,borderBottom:`1px solid ${C.border}`}}>
                {["Experiment","Params","Perplexity","Train Time","Key Finding"].map(h=>(
                  <th key={h} style={{padding:"14px 18px",textAlign:"left",fontSize:11,color:C.accent,textTransform:"uppercase",letterSpacing:1.5,fontWeight:600}}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {summaryRows.map((r,i)=>(
                <tr key={i} style={{borderBottom:`1px solid ${C.border}`,background:i%2===0?"transparent":`${C.accent}06`}}>
                  <td style={{padding:"13px 18px",color:C.text,fontWeight:600}}>{r.exp}</td>
                  <td style={{padding:"13px 18px",color:C.accent3}}>{r.params}</td>
                  <td style={{padding:"13px 18px",color:r.ppl==="∞"?C.accent2:C.text}}>{r.ppl}</td>
                  <td style={{padding:"13px 18px",color:C.textDim}}>{r.time}</td>
                  <td style={{padding:"13px 18px",color:C.textDim,fontSize:12}}>{r.finding}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* LoRA efficiency */}
      {view==="lora" && (
        <div style={{display:"flex",flexDirection:"column",gap:16}}>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:14}}>
            <StatCard label="Full Fine-tune"  value={formatParams(loraData.base_params || 1827968)} sub="100% of parameters" color={C.text}/>
            <StatCard label="LoRA Fine-tune"  value={(loraData.lora_params || 16384).toLocaleString()} sub={`${(loraData.lora_pct || 0.8963).toFixed(4)}% of parameters`} color={C.accent3}/>
            <StatCard label="Reduction"       value={`${(loraData.reduction_factor || 111.57).toFixed(2)}×`} sub="Fewer trainable params" color={C.accent}/>
          </div>
          <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:20}}>
            <div style={{fontSize:11,color:C.textDim,marginBottom:16,textTransform:"uppercase",letterSpacing:1.5}}>
              Trainable parameter breakdown
            </div>
            <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:12}}>
              <div style={{flex:1,height:28,background:C.border,borderRadius:6,overflow:"hidden",position:"relative"}}>
                <div style={{
                  position:"absolute",left:0,top:0,bottom:0,
                  width:`${loraData.lora_pct || 0.8963}%`,background:C.accent3,borderRadius:6,
                  display:"flex",alignItems:"center",paddingLeft:8,
                }}><span style={{fontSize:10,fontWeight:700,color:C.bg,whiteSpace:"nowrap"}}>LoRA 0.9%</span></div>
              </div>
              <div style={{fontSize:12,color:C.textDim,whiteSpace:"nowrap"}}>Base: 99.1% frozen</div>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14,marginTop:16}}>
              {[
                {label:"Rank (r)", value:loraData.rank ?? 4},{label:"Alpha (α)", value:loraData.alpha ?? 4},
                {label:"Scale (α/r)", value:((loraData.alpha ?? 4)/(loraData.rank ?? 4)).toFixed(1)},
                {label:"Adapter size", value:`${loraData.adapter_kb ?? 64} KB`},
                {label:"Target layers", value:(loraData.target_modules || ["W_q","W_k","W_v","W_o"]).join(",")},
                {label:"Fine-tune steps", value:loraData.fine_tune_steps ?? 1000},
              ].map(r=>(
                <div key={r.label} style={{display:"flex",justifyContent:"space-between",padding:"10px 14px",background:`${C.accent}0a`,borderRadius:8}}>
                  <span style={{fontSize:12,color:C.textDim}}>{r.label}</span>
                  <span style={{fontSize:13,color:C.text,fontFamily:"'DM Mono',monospace",fontWeight:600}}>{r.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Held-out evaluation */}
      {view==="heldout_eval" && (
        <div style={{display:"flex",flexDirection:"column",gap:16}}>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:14}}>
            <StatCard
              label="Baseline Mean PPL"
              value={baselineAgg.mean_perplexity?.toFixed ? baselineAgg.mean_perplexity.toFixed(2) : baselineAgg.mean_perplexity}
              sub={`std=${baselineAgg.std_perplexity?.toFixed ? baselineAgg.std_perplexity.toFixed(2) : baselineAgg.std_perplexity}`}
              color={C.text}
            />
            <StatCard
              label="LoRA Mean PPL"
              value={loraAgg.mean_perplexity?.toFixed ? loraAgg.mean_perplexity.toFixed(2) : loraAgg.mean_perplexity}
              sub={`std=${loraAgg.std_perplexity?.toFixed ? loraAgg.std_perplexity.toFixed(2) : loraAgg.std_perplexity}`}
              color={C.accent3}
            />
            <StatCard
              label="PPL Delta (LoRA-Baseline)"
              value={evalDelta.perplexity_mean_delta?.toFixed ? evalDelta.perplexity_mean_delta.toFixed(2) : evalDelta.perplexity_mean_delta}
              sub="negative is better for LoRA"
              color={evalDelta.perplexity_mean_delta < 0 ? C.accent3 : C.accent2}
            />
          </div>
          <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"20px 10px 10px"}}>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={seedRows}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="seed" stroke={C.muted} tick={{fontSize:11}} />
                <YAxis stroke={C.muted} tick={{fontSize:11}} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Legend wrapperStyle={{fontSize:11,color:C.textDim}} />
                <Bar dataKey="baseline_ppl" name="Baseline PPL" fill={C.accent2} radius={[5,5,0,0]} />
                <Bar dataKey="lora_ppl" name="LoRA PPL" fill={C.accent3} radius={[5,5,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{fontSize:12,color:C.textDim}}>
            Multi-seed held-out evaluation reduces single-run noise and makes model claims reproducible.
          </div>
        </div>
      )}

      {/* Rank sweep */}
      {view==="rank_sweep" && (
        <div style={{display:"flex",flexDirection:"column",gap:16}}>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:14}}>
            <StatCard label="Best Rank" value={`r=${bestRank.rank}`} sub="Lowest mean perplexity" color={C.accent} />
            <StatCard
              label="Best Mean PPL"
              value={bestRank.mean_perplexity?.toFixed ? bestRank.mean_perplexity.toFixed(2) : bestRank.mean_perplexity}
              sub="held-out mean over seeds"
              color={C.accent3}
            />
            <StatCard
              label="Best Trainable %"
              value={`${bestRank.trainable_pct?.toFixed ? bestRank.trainable_pct.toFixed(3) : bestRank.trainable_pct}%`}
              sub="parameter efficiency at best rank"
              color={C.text}
            />
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
            <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"20px 10px 10px"}}>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={rankRows}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="rank" stroke={C.muted} tick={{fontSize:11}} />
                  <YAxis stroke={C.muted} tick={{fontSize:11}} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Line type="monotone" dataKey="mean_ppl" name="Mean PPL" stroke={C.accent3} strokeWidth={2.5} dot />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"20px 10px 10px"}}>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={rankRows}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="rank" stroke={C.muted} tick={{fontSize:11}} />
                  <YAxis stroke={C.muted} tick={{fontSize:11}} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Bar dataKey="trainable_pct" name="Trainable %" fill={C.accent} radius={[5,5,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr",gap:10}}>
            {rankRows.map((r) => (
              <div key={r.rank} style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:10,padding:"10px 14px"}}>
                <div style={{fontSize:12,color:C.text}}>
                  <span style={{color:C.accent3,fontFamily:"'DM Mono',monospace"}}>{r.rank}</span>{" "}
                  · mean ppl {r.mean_ppl?.toFixed ? r.mean_ppl.toFixed(2) : r.mean_ppl} · trainable {r.trainable_pct?.toFixed ? r.trainable_pct.toFixed(3) : r.trainable_pct}%
                </div>
                <div style={{fontSize:11,color:C.textDim,marginTop:4}}>{r.decision}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Prompt benchmark */}
      {view==="prompt_benchmark" && (
        <div style={{display:"flex",flexDirection:"column",gap:16}}>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:14}}>
            <StatCard label="Best Run" value={bestPromptRun?.label || "n/a"} sub="Highest LoRA decisive win-rate" color={C.accent}/>
            <StatCard label="Best LoRA Win-rate" value={bestPromptRun ? `${(bestPromptRun.win_rate * 100).toFixed(2)}%` : "n/a"} sub="Decisive prompts only" color={C.accent3}/>
            <StatCard label="Benchmark Variants" value={promptRows.length} sub="evaluated prompt benchmark runs" color={C.text}/>
          </div>
          <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"20px 10px 10px"}}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={promptRows}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="label" stroke={C.muted} tick={{fontSize:11}} />
                <YAxis stroke={C.muted} tick={{fontSize:11}} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Legend wrapperStyle={{fontSize:11,color:C.textDim}} />
                <Bar dataKey="baseline_wins" name="Baseline wins" fill={C.accent2} radius={[5,5,0,0]} />
                <Bar dataKey="lora_wins" name="LoRA wins" fill={C.accent3} radius={[5,5,0,0]} />
                <Bar dataKey="ties" name="Ties" fill={C.muted} radius={[5,5,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Section 3: Architecture Explorer ─────────────────────────────────────────

function ArchExplorer() {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [view,      setView]      = useState("bars");

  const fetchInfo = async () => {
    setLoading(true);
    try {
      const r = await fetch(`${API}/model/info`);
      const d = await r.json();
      setModelInfo(d);
    } catch {
      setModelInfo(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchInfo(); }, []);

  const totalParams = modelInfo?.total_parameters ?? ARCH_PARAMS.reduce((s,r)=>s+r.params,0);

  return (
    <div style={{display:"flex",flexDirection:"column",gap:20}}>
      {/* Stat row */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:14}}>
        <StatCard label="Total Params"  value={formatParams(totalParams)} sub={loading ? "loading..." : `checkpoint: ${modelInfo?.checkpoint_path || "fallback"}`} color={C.accent}/>
        <StatCard label="d_model"       value={modelInfo?.d_model ?? 128}   sub="Embedding dimension"       color={C.accent3}/>
        <StatCard label="Attention"     value={`${modelInfo?.n_heads ?? 4} heads`} sub={`n_layers=${modelInfo?.n_layers ?? 4}`}         color={C.text}/>
        <StatCard label="Context"       value={modelInfo?.context_len ?? 256}   sub={`device=${modelInfo?.device ?? "unknown"}`}        color={C.text}/>
      </div>

      {/* View toggle */}
      <div style={{display:"flex",gap:8}}>
        <Pill active={view==="bars"}   onClick={()=>setView("bars")}>Bar Chart</Pill>
        <Pill active={view==="tree"}   onClick={()=>setView("tree")}>Layer Tree</Pill>
        <Pill active={view==="flow"}   onClick={()=>setView("flow")}>Data Flow</Pill>
      </div>

      {/* Bar chart */}
      {view==="bars" && (
        <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"20px 10px 10px"}}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={ARCH_PARAMS} margin={{left:10,right:20}}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="name" stroke={C.muted} tick={{fontSize:11}} />
              <YAxis stroke={C.muted} tick={{fontSize:11}}
                tickFormatter={v=>v>=1e6?`${(v/1e6).toFixed(1)}M`:v>=1e3?`${(v/1e3).toFixed(0)}K`:v}/>
              <Tooltip contentStyle={TOOLTIP_STYLE}
                formatter={(v,n)=>[v.toLocaleString(),n]}/>
              <Bar dataKey="params" name="Parameters" radius={[6,6,0,0]}>
                {ARCH_PARAMS.map((e,i)=><Cell key={i} fill={e.color}/>)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Layer tree */}
      {view==="tree" && (
        <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:20}}>
          {[
            {name:"NanoGPT",      indent:0, params:"1,827,968", color:C.accent},
            {name:"tok_emb",      indent:1, params:"1,024,000  (V×d)", color:C.accent2},
            {name:"blocks[0..3]", indent:1, params:"197,632 each", color:C.text},
            {name:"  attn",       indent:2, params:"W_q,W_k,W_v,W_o  4×128²", color:C.accent3},
            {name:"    W_q",      indent:3, params:"16,384", color:C.muted},
            {name:"    W_k",      indent:3, params:"16,384", color:C.muted},
            {name:"    W_v",      indent:3, params:"16,384", color:C.muted},
            {name:"    W_o",      indent:3, params:"16,384", color:C.muted},
            {name:"  ffn",        indent:2, params:"W1,W2  2×(128×512)", color:C.accent3},
            {name:"    W1",       indent:3, params:"65,536", color:C.muted},
            {name:"    W2",       indent:3, params:"65,536", color:C.muted},
            {name:"  norm_attn",  indent:2, params:"128 (RMSNorm γ)", color:C.muted},
            {name:"  norm_ffn",   indent:2, params:"128 (RMSNorm γ)", color:C.muted},
            {name:"norm_final",   indent:1, params:"128", color:C.textDim},
            {name:"lm_head",      indent:1, params:"weight-tied → 0 extra", color:C.textDim},
          ].map((r,i)=>(
            <div key={i} style={{
              display:"flex", alignItems:"center", justifyContent:"space-between",
              padding:"8px 0", paddingLeft: r.indent*24,
              borderBottom:`1px solid ${C.border}`,
            }}>
              <span style={{fontSize:13,color:r.color,fontFamily:"'DM Mono',monospace"}}>
                {r.name}
              </span>
              <span style={{fontSize:12,color:C.textDim,fontFamily:"'DM Mono',monospace"}}>
                {r.params}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Data flow */}
      {view==="flow" && (
        <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:28}}>
          {[
            {label:"Input tokens",     shape:"(B, T)",            note:"integer ids ∈ [0, V)",         color:C.textDim},
            {label:"tok_emb lookup",   shape:"(B, T, 128)",       note:"V×d table lookup",              color:C.accent2},
            {label:"+ RoPE",           shape:"(B, T, 128)",       note:"position encoded in Q,K",       color:C.textDim},
            {label:"Block 0",          shape:"(B, T, 128)",       note:"Attn + FFN + residuals",        color:C.accent},
            {label:"Block 1",          shape:"(B, T, 128)",       note:"same structure",                color:C.accent},
            {label:"Block 2",          shape:"(B, T, 128)",       note:"same structure",                color:C.accent},
            {label:"Block 3",          shape:"(B, T, 128)",       note:"same structure",                color:C.accent},
            {label:"RMSNorm final",    shape:"(B, T, 128)",       note:"stabilise output scale",        color:C.accent3},
            {label:"lm_head (tied)",   shape:"(B, T, 8000)",      note:"d → vocab, weight = tok_emb.W", color:C.accent2},
            {label:"softmax → probs",  shape:"(B, T, 8000)",      note:"next-token distribution",       color:C.accent3},
          ].map((r,i,arr)=>(
            <div key={i} style={{display:"flex",flexDirection:"column",alignItems:"center"}}>
              <div style={{
                background:`${r.color}18`, border:`1px solid ${r.color}44`,
                borderRadius:10, padding:"12px 20px", width:"100%",
                display:"flex", justifyContent:"space-between", alignItems:"center",
              }}>
                <span style={{fontSize:13,color:r.color,fontWeight:600}}>{r.label}</span>
                <div style={{textAlign:"right"}}>
                  <div style={{fontSize:12,color:C.text,fontFamily:"'DM Mono',monospace"}}>{r.shape}</div>
                  <div style={{fontSize:11,color:C.textDim}}>{r.note}</div>
                </div>
              </div>
              {i<arr.length-1 && (
                <div style={{width:2,height:18,background:C.border,margin:"2px 0"}}/>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function InferenceComparison() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchData = async () => {
    setLoading(true);
    setError("");
    try {
      const r = await fetch(`${API}/inference/compare`);
      if (!r.ok) throw new Error(await r.text());
      setData(await r.json());
    } catch (e) {
      setError(`Failed to load comparison: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const nano = data?.nanogpt || {};
  const ext = data?.llm_inference_project || {};
  const speedup = (nano.tokens_per_second && ext.tokens_per_second)
    ? (nano.tokens_per_second / ext.tokens_per_second).toFixed(2)
    : null;

  return (
    <div style={{display:"flex",flexDirection:"column",gap:16}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
        <div style={{fontSize:14,color:C.textDim}}>Prompt: <span style={{color:C.text}}>{data?.prompt || "-"}</span></div>
        <button onClick={fetchData} style={{padding:"8px 12px", borderRadius:8, border:`1px solid ${C.border}`, background:C.surface, color:C.text, cursor:"pointer"}}>
          {loading ? "Refreshing..." : "Refresh Benchmark"}
        </button>
      </div>

      {error && <div style={{color:C.accent2,fontSize:12}}>{error}</div>}

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
        <StatCard label="NanoGPT tok/sec" value={nano.tokens_per_second ?? "-"} sub={nano.device || "unknown"} color={C.accent3} />
        <StatCard label="TinyLlama tok/sec" value={ext.tokens_per_second ?? "-"} sub={ext.device || "unknown"} color={C.accent2} />
        <StatCard label="NanoGPT latency (s)" value={nano.elapsed_seconds ?? "-"} sub={`tokens=${nano.tokens_generated ?? "-"}`} color={C.text} />
        <StatCard label="TinyLlama latency (s)" value={ext.elapsed_seconds ?? "-"} sub={`tokens=${ext.tokens_generated ?? "-"}`} color={C.text} />
      </div>

      <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:16}}>
        <div style={{fontSize:13,color:C.text,marginBottom:8}}>Comparison Summary</div>
        <div style={{fontSize:12,color:C.textDim,lineHeight:1.8}}>
          {speedup
            ? `Relative speed (NanoGPT / TinyLlama): ${speedup}x`
            : "Relative speed unavailable (missing metrics)."}
          <br/>
          TinyLlama memory (reported): {ext.vram_gb ?? "n/a"} GB VRAM.
        </div>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
        <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:16}}>
          <div style={{fontSize:12,color:C.textDim,marginBottom:8}}>NanoGPT Output Sample</div>
          <div style={{fontFamily:"'DM Mono',monospace",fontSize:12,color:C.text,whiteSpace:"pre-wrap"}}>{nano.output_sample || "-"}</div>
        </div>
        <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:16}}>
          <div style={{fontSize:12,color:C.textDim,marginBottom:8}}>TinyLlama Output Sample</div>
          <div style={{fontFamily:"'DM Mono',monospace",fontSize:12,color:C.text,whiteSpace:"pre-wrap"}}>{ext.output_sample || "-"}</div>
        </div>
      </div>
    </div>
  );
}

function DemoHighlights() {
  const [expPayload, setExpPayload] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const [expRes, modelRes] = await Promise.all([
          fetch(`${API}/experiments`),
          fetch(`${API}/model/info`),
        ]);
        if (expRes.ok) setExpPayload(await expRes.json());
        if (modelRes.ok) setModelInfo(await modelRes.json());
      } catch {
        setExpPayload(null);
      }
    })();
  }, []);

  const viewData = buildExperimentsView(expPayload);
  const evalData = viewData.evaluation || EVAL_DATA;
  const rankSweep = viewData.rankSweep || RANK_SWEEP_DATA;
  const promptBenchmarks = viewData.promptBenchmarks || PROMPT_BENCHMARK_DATA;
  const experiments = viewData.experiments || EXPERIMENTS;

  const baselineAgg = evalData?.baseline?.aggregate || EVAL_DATA.baseline.aggregate;
  const loraAgg = evalData?.lora?.aggregate || EVAL_DATA.lora.aggregate;
  const evalDelta = evalData?.delta || EVAL_DATA.delta;
  const bestRank = rankSweep?.best_rank || RANK_SWEEP_DATA.best_rank;

  const promptRows = Object.entries(promptBenchmarks).map(([key, value]) => ({
    key,
    label: value.label || key,
    baseline_wins: value.counts?.baseline_wins ?? 0,
    lora_wins: value.counts?.lora_wins ?? 0,
    ties: value.counts?.ties ?? 0,
    win_rate: value.counts?.lora_win_rate_on_decisive ?? 0,
  }));
  const bestPromptRun = [...promptRows].sort((a, b) => b.win_rate - a.win_rate)[0];

  const rankRows = (rankSweep?.ranks || []).map((r) => ({
    rank: `r=${r.rank}`,
    mean_ppl: r.mean_perplexity,
  }));

  const expRows = experiments
    .filter((e) => Number.isFinite(Number(e.perplexity)))
    .map((e) => ({ experiment: e.label, perplexity: Number(e.perplexity) }));

  return (
    <div style={{display:"flex",flexDirection:"column",gap:18}}>
      <div style={{fontSize:13,color:C.textDim}}>
        One-screen summary for demos: reproducible metrics, rank tradeoffs, and benchmark outcomes.
      </div>

      <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:14}}>
        <StatCard
          label="Model Size"
          value={modelInfo?.total_parameters ? formatParams(modelInfo.total_parameters) : "0.79M"}
          sub="from /model/info"
          color={C.accent}
        />
        <StatCard
          label="Held-out PPL Delta"
          value={evalDelta.perplexity_mean_delta?.toFixed ? evalDelta.perplexity_mean_delta.toFixed(2) : evalDelta.perplexity_mean_delta}
          sub={`baseline ${baselineAgg.mean_perplexity?.toFixed ? baselineAgg.mean_perplexity.toFixed(2) : baselineAgg.mean_perplexity} → lora ${loraAgg.mean_perplexity?.toFixed ? loraAgg.mean_perplexity.toFixed(2) : loraAgg.mean_perplexity}`}
          color={evalDelta.perplexity_mean_delta < 0 ? C.accent3 : C.accent2}
        />
        <StatCard
          label="Best Rank (Held-out)"
          value={`r=${bestRank.rank}`}
          sub={`ppl ${bestRank.mean_perplexity?.toFixed ? bestRank.mean_perplexity.toFixed(2) : bestRank.mean_perplexity}`}
          color={C.accent3}
        />
        <StatCard
          label="Best Prompt Win-rate"
          value={bestPromptRun ? `${(bestPromptRun.win_rate * 100).toFixed(2)}%` : "n/a"}
          sub={bestPromptRun?.label || "no benchmark loaded"}
          color={C.text}
        />
      </div>

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
        <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"18px 10px 10px"}}>
          <div style={{fontSize:12,color:C.textDim,marginBottom:8,paddingLeft:8}}>LoRA Rank Sweep (mean perplexity)</div>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={rankRows}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="rank" stroke={C.muted} tick={{fontSize:11}} />
              <YAxis stroke={C.muted} tick={{fontSize:11}} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Line type="monotone" dataKey="mean_ppl" name="Mean PPL" stroke={C.accent3} strokeWidth={2.5} dot />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"18px 10px 10px"}}>
          <div style={{fontSize:12,color:C.textDim,marginBottom:8,paddingLeft:8}}>Prompt Benchmark Wins</div>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={promptRows}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="label" stroke={C.muted} tick={{fontSize:11}} />
              <YAxis stroke={C.muted} tick={{fontSize:11}} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Legend wrapperStyle={{fontSize:11,color:C.textDim}} />
              <Bar dataKey="baseline_wins" name="Baseline wins" fill={C.accent2} radius={[5,5,0,0]} />
              <Bar dataKey="lora_wins" name="LoRA wins" fill={C.accent3} radius={[5,5,0,0]} />
              <Bar dataKey="ties" name="Ties" fill={C.muted} radius={[5,5,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"18px 10px 10px"}}>
        <div style={{fontSize:12,color:C.textDim,marginBottom:8,paddingLeft:8}}>Final Perplexity by Experiment</div>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={expRows}>
            <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
            <XAxis dataKey="experiment" stroke={C.muted} tick={{fontSize:11}} />
            <YAxis stroke={C.muted} tick={{fontSize:11}} />
            <Tooltip contentStyle={TOOLTIP_STYLE} />
            <Bar dataKey="perplexity" name="Perplexity" fill={C.accent} radius={[5,5,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── Root App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState("playground");

  const tabs = [
    {id:"highlights", label:"⭐ Demo Highlights"},
    {id:"playground", label:"⚡ Generation Playground"},
    {id:"experiments", label:"📊 Experiment Results"},
    {id:"arch",        label:"🔬 Architecture Explorer"},
    {id:"compare",     label:"🧪 Inference Compare"},
  ];

  return (
    <div style={{
      minHeight:"100vh", background:C.bg, color:C.text,
      fontFamily:"'DM Sans', system-ui, sans-serif",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 3px; }
        input[type=range] { height: 4px; }
      `}</style>

      {/* Header */}
      <header style={{
        borderBottom:`1px solid ${C.border}`, padding:"0 40px",
        display:"flex", alignItems:"center", justifyContent:"space-between",
        height:62,
      }}>
        <div style={{display:"flex",alignItems:"center",gap:14}}>
          <div style={{
            width:34, height:34, borderRadius:9,
            background:`linear-gradient(135deg, ${C.accent}, ${C.accent2})`,
            display:"flex", alignItems:"center", justifyContent:"center",
            fontSize:16,
          }}>⬡</div>
          <div>
            <div style={{fontSize:16,fontWeight:700,letterSpacing:"-0.3px"}}>NanoGPT Lab</div>
          </div>
        </div>
        <div style={{display:"flex",gap:6}}>
          {tabs.map(t=>(
            <button key={t.id} onClick={()=>setTab(t.id)} style={{
              padding:"8px 18px", borderRadius:8,
              background: tab===t.id ? `${C.accent}22` : "transparent",
              border: `1px solid ${tab===t.id ? C.accent : C.border}`,
              color: tab===t.id ? C.accent : C.textDim,
              cursor:"pointer", fontSize:13, fontFamily:"inherit",
              transition:"all 0.15s", fontWeight: tab===t.id ? 600 : 400,
            }}>{t.label}</button>
          ))}
        </div>
      </header>

      {/* Content */}
      <main style={{maxWidth:1200, margin:"0 auto", padding:"36px 40px"}}>
        {tab==="highlights"  && <DemoHighlights/>}
        {tab==="playground"  && <Playground/>}
        {tab==="experiments" && <ExperimentResults/>}
        {tab==="arch"        && <ArchExplorer/>}
        {tab==="compare"     && <InferenceComparison/>}
      </main>
    </div>
  );
}
