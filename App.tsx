import { useState, useEffect, useRef } from “react”;

// ── Seeded RNG ────────────────────────────────────────────────────────────────
function seededRng(seed) {
let s = seed >>> 0;
return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0xffffffff; };
}

// ── Generate 80-point valence history ────────────────────────────────────────
function genHistory() {
const rng = seededRng(42);
const h = []; let p=0.2,a=0.1,c=0.5,f=0.15,conf=0.55;
for (let i=0;i<80;i++) {
p    = Math.max(-1,Math.min(1, p    +(rng()-0.48)*0.08));
a    = Math.max(-1,Math.min(1, a    +(rng()-0.48)*0.07));
c    = Math.max( 0,Math.min(1, c    +(rng()-0.45)*0.05));
f    = Math.max( 0,Math.min(1, f    +(rng()-0.52)*0.06));
conf = Math.max( 0,Math.min(1, conf +(rng()-0.47)*0.04));
h.push({i, p:+p.toFixed(3), a:+a.toFixed(3), c:+c.toFixed(3), f:+f.toFixed(3), conf:+conf.toFixed(3)});
}
return h;
}

const INIT_HIST = genHistory();
const PHASES = [“OBSERVE”,“PLAN”,“PROPOSE”,“VERIFY”,“APPLY”];
const PHASE_COLOR = {OBSERVE:”#60a5fa”,PLAN:”#a78bfa”,PROPOSE:”#fb923c”,VERIFY:”#34d399”,APPLY:”#10b981”};

const GOALS = [
{ id:“g1”, title:“Build REST API server”,    status:“active”,   progress:0.62, children:[“g1a”,“g1b”,“g1c”,“g1d”] },
{ id:“g1a”,title:“Define requirements”,      status:“complete”, progress:1.0,  children:[], parent:“g1” },
{ id:“g1b”,title:“Design architecture”,      status:“complete”, progress:1.0,  children:[], parent:“g1” },
{ id:“g1c”,title:“Implement core”,           status:“active”,   progress:0.48, children:[], parent:“g1” },
{ id:“g1d”,title:“Write tests”,              status:“pending”,  progress:0.0,  children:[], parent:“g1” },
{ id:“g2”, title:“Calibrate world model”,    status:“stalled”,  progress:0.31, children:[] },
{ id:“g3”, title:“Social sim refinement”,    status:“pending”,  progress:0.0,  children:[] },
];

const EPISODES = [
{ ts:“07:14:32”, summary:“Workspace — 1 person, laptop. Wake word detected. Task discussion started.”, importance:0.91, entities:[“person”,“laptop”], dur:42 },
{ ts:“07:09:11”, summary:“Scene shift outdoor→workspace. Motion detected, 2 objects tracked.”, importance:0.73, entities:[“bicycle”,“bag”], dur:28 },
{ ts:“07:01:58”, summary:“Voice dialogue 4 turns. Goal ‘REST API’ created, 6 sub-goals decomposed.”, importance:0.88, entities:[“person”], dur:61 },
{ ts:“06:55:40”, summary:“Quiet scene. Memory consolidation: 3 episodes compressed to 1 abstract.”, importance:0.52, entities:[], dur:18 },
{ ts:“06:44:12”, summary:“Code exec: autonomous_ide created project janus-api-v2.”, importance:0.79, entities:[“code”,“file”], dur:35 },
{ ts:“06:31:05”, summary:“Person left view. Frustration bump +0.2 from stalled goal g2.”, importance:0.67, entities:[“person”], dur:12 },
];

const TOOLS = [
{ name:“memory_query”,  desc:“Search episodic + semantic memory”,   risk:“low”,    calls:203 },
{ name:“valence_query”, desc:“Read homeostasis valence state”,       risk:“low”,    calls:142 },
{ name:“file_read”,     desc:“Read a local file”,                    risk:“low”,    calls:89  },
{ name:“file_write”,    desc:“Write content to local filesystem”,    risk:“medium”, calls:37  },
{ name:“shell_cmd”,     desc:“Whitelisted shell command runner”,     risk:“medium”, calls:21  },
{ name:“code_exec”,     desc:“Execute Python in sandbox”,            risk:“high”,   calls:14  },
{ name:“web_fetch”,     desc:“Fetch URL (offline stub)”,             risk:“high”,   calls:6   },
{ name:“calendar_add”,  desc:“Add event to local calendar store”,    risk:“medium”, calls:3   },
];

const INIT_EVENTS = [
{ ts:“07:14:51”, kind:“cycle_start”,       data:“cycle=47 phase=OBSERVE” },
{ ts:“07:14:50”, kind:“perception_event”,  data:“source=audio type=wake_word” },
{ ts:“07:14:49”, kind:“observe”,           data:“entities=[person,laptop] scene=workspace” },
{ ts:“07:14:48”, kind:“plan”,              data:“intent=maintain_homeostasis sub_goals=3” },
{ ts:“07:14:47”, kind:“propose”,           data:“action_count=2” },
{ ts:“07:14:46”, kind:“verify”,            data:“safe=2 blocked=0” },
{ ts:“07:14:45”, kind:“apply_ok”,          data:“tool=memory_query → 3 hits” },
{ ts:“07:14:44”, kind:“apply_ok”,          data:“tool=valence_query → {pleasure:0.71}” },
{ ts:“07:14:43”, kind:“apply_fail”,        data:“tool=web_fetch error=offline” },
{ ts:“07:14:42”, kind:“perception_event”,  data:“source=vision type=person_entered” },
];

const EVENT_COLOR = {
apply_ok:”#10b981”,apply_fail:”#ef4444”,perception_event:”#60a5fa”,
cycle_start:”#6366f1”,observe:”#94a3b8”,plan:”#a78bfa”,
propose:”#fb923c”,verify:”#34d399”,loop_error:”#f87171”,
};

const STATUS = {
complete:{ bg:”#10b981”,label:“✓ done” },
active:  { bg:”#3b82f6”,label:“● active” },
pending: { bg:”#475569”,label:“○ pending” },
stalled: { bg:”#f59e0b”,label:“⚠ stalled” },
failed:  { bg:”#ef4444”,label:“✗ failed” },
};

const RISK_COL = { low:”#10b981”,medium:”#f59e0b”,high:”#ef4444” };

// ── Gauge ──────────────────────────────────────────────────────────────────────
function Gauge({ value, color, label, sz=76 }) {
const r=sz/2-7, cx=sz/2, cy=sz/2, circ=2*Math.PI*r;
const norm = Math.max(0, Math.min(1, (value+1)/2));
return (
<div style={{display:“flex”,flexDirection:“column”,alignItems:“center”,gap:2}}>
<svg width={sz} height={sz}>
<circle cx={cx} cy={cy} r={r} fill=“none” stroke=”#0f172a” strokeWidth={6}
strokeDasharray={`${circ*0.75} ${circ}`} strokeLinecap=“round”
transform={`rotate(135 ${cx} ${cy})`} />
<circle cx={cx} cy={cy} r={r} fill=“none” stroke={color} strokeWidth={6}
strokeDasharray={`${norm*circ*0.75} ${circ}`} strokeLinecap=“round”
transform={`rotate(135 ${cx} ${cy})`}
style={{transition:“stroke-dasharray 0.8s ease”}} />
<text x={cx} y={cy+4} textAnchor="middle" fontSize="11" fontWeight="700" fill={color}>
{value>0?”+”:””}{value.toFixed(2)}
</text>
</svg>
<span style={{fontSize:8,color:”#475569”,letterSpacing:“0.1em”,textTransform:“uppercase”}}>{label}</span>
</div>
);
}

// ── SVG valence chart ─────────────────────────────────────────────────────────
function ValChart({ hist }) {
const W=520, H=130, PL=30, PR=8, PT=6, PB=16;
const cw=W-PL-PR, ch=H-PT-PB;
const sl = hist.slice(-60);
const sx = i => PL + (i/(sl.length-1||1))*cw;
const sy = v => PT + ((1-(v+1)/2)*ch);
const dims=[
{k:“p”,col:”#10b981”},{k:“a”,col:”#f59e0b”},{k:“c”,col:”#60a5fa”},
{k:“f”,col:”#f87171”},{k:“conf”,col:”#a78bfa”},
];
const labels=[“Pleasure”,“Arousal”,“Curiosity”,“Frustration”,“Confidence”];
return (
<svg viewBox={`0 0 ${W} ${H}`} style={{width:“100%”,height:H}}>
{[-1,-0.5,0,0.5,1].map(v=>(
<g key={v}>
<line x1={PL} x2={W-PR} y1={sy(v)} y2={sy(v)} stroke="#1e293b" strokeWidth="0.8"/>
<text x={PL-3} y={sy(v)+3} textAnchor="end" fontSize="7" fill="#334155">{v}</text>
</g>
))}
{dims.map(d=>{
const pts=sl.map((pt,i)=>`${sx(i).toFixed(1)},${sy(pt[d.k]).toFixed(1)}`).join(” “);
return <polyline key={d.k} points={pts} fill="none" stroke={d.col}
strokeWidth="1.8" strokeLinejoin="round" opacity="0.9"/>;
})}
{dims.map((d,i)=>(
<g key={d.k} transform={`translate(${PL+i*100},${H-4})`}>
<line x1="0" x2="10" y1="0" y2="0" stroke={d.col} strokeWidth="2"/>
<text x="13" y="3" fontSize="8" fill={d.col}>{labels[i]}</text>
</g>
))}
</svg>
);
}

// ── Goal node ─────────────────────────────────────────────────────────────────
function GoalNode({ goal, map, depth=0 }) {
const [open,setOpen]=useState(depth<1);
const children=(goal.children||[]).map(id=>map[id]).filter(Boolean);
const s=STATUS[goal.status]||STATUS.pending;
const pct=Math.round(goal.progress*100);
return (
<div style={{marginLeft:depth*14,marginTop:5}}>
<div onClick={()=>setOpen(o=>!o)} style={{
display:“flex”,alignItems:“center”,gap:7,cursor:“pointer”,padding:“5px 8px”,
borderRadius:6,background:depth===0?”#0a1628”:“transparent”,
border:depth===0?“1px solid #1e293b”:“none”
}}>
<span style={{fontSize:9,padding:“1px 5px”,borderRadius:3,background:s.bg+“22”,
color:s.bg,fontFamily:“monospace”,whiteSpace:“nowrap”}}>{s.label}</span>
<span style={{flex:1,fontSize:11,color:”#cbd5e1”,fontWeight:depth===0?700:400}}>{goal.title}</span>
<div style={{width:40,height:3,background:”#1e293b”,borderRadius:2}}>
<div style={{width:`${pct}%`,height:“100%”,background:s.bg,borderRadius:2,
transition:“width 0.6s”}}/>
</div>
<span style={{fontSize:9,color:”#475569”,width:24,textAlign:“right”}}>{pct}%</span>
{children.length>0&&<span style={{fontSize:9,color:”#475569”}}>{open?“▾”:“▸”}</span>}
</div>
{open&&children.map(c=><GoalNode key={c.id} goal={c} map={map} depth={depth+1}/>)}
</div>
);
}

// ── Card ──────────────────────────────────────────────────────────────────────
function Card({ title, badge, accent=”#1e293b”, children }) {
return (
<div style={{
background:“linear-gradient(135deg,#0d1929 0%,#0a1222 100%)”,
border:`1px solid ${accent}44`,borderRadius:12,padding:“14px 16px”,
position:“relative”,overflow:“hidden”
}}>
<div style={{position:“absolute”,top:0,left:0,right:0,height:2,
background:`linear-gradient(90deg,${accent}88,transparent)`}}/>
<div style={{display:“flex”,alignItems:“center”,gap:8,marginBottom:11}}>
<h2 style={{margin:0,fontSize:11,fontWeight:700,color:”#94a3b8”,
letterSpacing:“0.12em”,textTransform:“uppercase”}}>{title}</h2>
{badge!=null&&<span style={{fontSize:10,color:”#334155”,marginLeft:“auto”,
fontFamily:“monospace”}}>{badge}</span>}
</div>
{children}
</div>
);
}

// ── Dot ───────────────────────────────────────────────────────────────────────
function Dot({ color }) {
return (
<span style={{position:“relative”,display:“inline-flex”,width:9,height:9}}>
<span style={{position:“absolute”,inset:0,borderRadius:“50%”,background:color,
opacity:0.5,animation:“ping 1.4s ease infinite”}}/>
<span style={{borderRadius:“50%”,width:9,height:9,background:color}}/>
</span>
);
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
const [hist,setHist]=useState(INIT_HIST);
const [live,setLive]=useState(INIT_HIST[INIT_HIST.length-1]);
const [cycle,setCycle]=useState(47);
const [phaseIdx,setPhaseIdx]=useState(0);
const [events,setEvents]=useState(INIT_EVENTS);
const logRef=useRef(null);
const phase=PHASES[phaseIdx];
const pCol=PHASE_COLOR[phase];

useEffect(()=>{
const id=setInterval(()=>{
const bump=(v,lo,hi,d)=>Math.max(lo,Math.min(hi,v+(Math.random()-d)*0.04));
setLive(prev=>{
const n={
…prev,
p:   bump(prev.p,   -1, 1, 0.48),
a:   bump(prev.a,   -1, 1, 0.48),
c:   bump(prev.c,    0, 1, 0.45),
f:   bump(prev.f,    0, 1, 0.52),
conf:bump(prev.conf, 0, 1, 0.47),
i:prev.i+1,
};
setHist(h=>[…h.slice(-79),n]);
return n;
});
setCycle(c=>c+1);
setPhaseIdx(i=>(i+1)%5);
setEvents(prev=>{
const kinds=[“observe”,“apply_ok”,“perception_event”,“verify”,“plan”,“cycle_start”];
const dataMap={observe:“entities=[person] scene=workspace”,apply_ok:“tool=memory_query → 2 hits”,
perception_event:“source=vision type=scene_change”,verify:“safe=2 blocked=0”,
plan:“sub_goals=3 intent=homeostasis”,cycle_start:`cycle=${cycle+1}`};
const k=kinds[Math.floor(Math.random()*kinds.length)];
const now=new Date();
const ts=`${String(now.getHours()).padStart(2,"0")}:${String(now.getMinutes()).padStart(2,"0")}:${String(now.getSeconds()).padStart(2,"0")}`;
return [{ts,kind:k,data:dataMap[k]},…prev.slice(0,24)];
});
},1300);
return ()=>clearInterval(id);
},[]);

useEffect(()=>{if(logRef.current)logRef.current.scrollTop=0;},[events.length]);

const goalMap=Object.fromEntries(GOALS.map(g=>[g.id,g]));
const roots=GOALS.filter(g=>!g.parent);

return (
<div style={{minHeight:“100vh”,background:”#060d17”,color:”#e2e8f0”,
fontFamily:”‘Space Mono’,‘Courier New’,monospace”,paddingBottom:40}}>
<style>{`@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap'); *{box-sizing:border-box;} ::-webkit-scrollbar{width:3px;} ::-webkit-scrollbar-track{background:#0a1222;} ::-webkit-scrollbar-thumb{background:#1e293b;border-radius:2px;} @keyframes ping{75%,100%{transform:scale(2.2);opacity:0;}}`}</style>

```
  {/* ── Header ── */}
  <div style={{
    background:"linear-gradient(180deg,#0a1628 0%,#060d17 100%)",
    borderBottom:"1px solid #0f172a",
    padding:"18px 20px 14px",position:"sticky",top:0,zIndex:50,
  }}>
    <div style={{display:"flex",alignItems:"flex-start",justifyContent:"space-between",
                 maxWidth:880,margin:"0 auto"}}>
      <div style={{display:"flex",alignItems:"center",gap:10}}>
        <Dot color={pCol}/>
        <div>
          <div style={{fontSize:20,fontWeight:800,lineHeight:1.1,
                       fontFamily:"'Syne',sans-serif",letterSpacing:"-0.01em"}}>
            <span style={{color:pCol}}>JANUS</span>
            <span style={{color:"#e2e8f0",fontWeight:700}}> COGNITION</span>
          </div>
          <div style={{fontSize:9,color:"#334155",letterSpacing:"0.14em",textTransform:"uppercase",marginTop:2}}>
            Autonomous Cognitive Monitor · v0.9
          </div>
        </div>
      </div>
      <div style={{textAlign:"right"}}>
        <div style={{display:"inline-flex",alignItems:"center",gap:6,
                     background:pCol+"15",border:`1px solid ${pCol}33`,
                     borderRadius:6,padding:"3px 10px",marginBottom:4}}>
          <span style={{fontSize:10,color:pCol,letterSpacing:"0.12em",fontWeight:700}}>
            ◈ {phase}
          </span>
        </div>
        <div style={{fontSize:9,color:"#334155",letterSpacing:"0.06em"}}>
          CYCLE #{cycle} &nbsp;·&nbsp; {new Date().toLocaleTimeString()}
        </div>
      </div>
    </div>
  </div>

  <div style={{padding:"18px 16px",maxWidth:880,margin:"0 auto"}}>

    {/* ── Gauges ── */}
    <div style={{display:"flex",gap:12,justifyContent:"center",
                 flexWrap:"wrap",marginBottom:14}}>
      {[
        {k:"p",   col:"#10b981",lbl:"PLEASURE"},
        {k:"a",   col:"#f59e0b",lbl:"AROUSAL"},
        {k:"c",   col:"#60a5fa",lbl:"CURIOSITY"},
        {k:"f",   col:"#f87171",lbl:"FRUSTRATE"},
        {k:"conf",col:"#a78bfa",lbl:"CONFIDENCE"},
      ].map(d=><Gauge key={d.k} value={+(live[d.k]||0).toFixed(2)} color={d.col} label={d.lbl}/>)}
    </div>

    {/* ── Valence chart ── */}
    <Card title="Homeostasis Valence" badge={`${hist.length} samples · live`} accent="#3b82f6">
      <ValChart hist={hist}/>
    </Card>

    <div style={{height:12}}/>

    {/* ── 2-col grid ── */}
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>

      {/* Perception Feed */}
      <Card title="Perception Feed" accent="#10b981">
        <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:9}}>
          <Dot color="#10b981"/>
          <span style={{fontSize:9,color:"#10b981",letterSpacing:"0.1em"}}>LIVE · 9.8 FPS</span>
        </div>
        {[
          {lbl:"VISUAL",  val:"Workspace · 1 person, laptop",  col:"#60a5fa"},
          {lbl:"AUDIO",   val:"Listening for wake word…",       col:"#f59e0b"},
          {lbl:"SCENE",   val:"workspace · bright",             col:"#94a3b8"},
          {lbl:"PEOPLE",  val:"1 detected",                     col:"#a78bfa"},
        ].map(r=>(
          <div key={r.lbl} style={{display:"grid",gridTemplateColumns:"50px 1fr",
                                   gap:6,marginBottom:6,alignItems:"start"}}>
            <span style={{fontSize:8,color:"#334155",letterSpacing:"0.1em",
                          textAlign:"right",paddingTop:1}}>{r.lbl}</span>
            <span style={{fontSize:11,color:r.col,lineHeight:1.4}}>{r.val}</span>
          </div>
        ))}
        <div style={{marginTop:10,height:3,background:"#0a1222",borderRadius:2,overflow:"hidden"}}>
          <div style={{
            height:"100%",width:"60%",borderRadius:2,
            background:"linear-gradient(90deg,#10b981,#3b82f6)",
            transition:"width 1.2s ease",
            width:`${50+30*Math.sin(Date.now()/2000)}%`
          }}/>
        </div>
      </Card>

      {/* Goal / Task Graph */}
      <Card title="Goal / Task Graph" badge={`${GOALS.length} nodes`} accent="#a78bfa">
        <div style={{maxHeight:210,overflowY:"auto"}}>
          {roots.map(g=><GoalNode key={g.id} goal={g} map={goalMap}/>)}
        </div>
      </Card>

      {/* Episodic Traces */}
      <Card title="Recent Episodic Traces" badge={`${EPISODES.length} episodes`} accent="#f59e0b">
        <div style={{maxHeight:210,overflowY:"auto"}}>
          {EPISODES.map((ep,i)=>(
            <div key={i} style={{
              borderLeft:`2px solid ${ep.importance>0.8?"#f59e0b":"#1e293b"}`,
              paddingLeft:10,marginBottom:9
            }}>
              <div style={{fontSize:11,color:"#cbd5e1",lineHeight:1.4}}>{ep.summary}</div>
              <div style={{display:"flex",gap:7,marginTop:3,flexWrap:"wrap",alignItems:"center"}}>
                <span style={{fontSize:9,color:"#334155"}}>{ep.ts}</span>
                <span style={{fontSize:9,color:"#f59e0b"}}>imp {ep.importance.toFixed(2)}</span>
                <span style={{fontSize:9,color:"#334155"}}>{ep.dur}s</span>
                {ep.entities.map(e=>(
                  <span key={e} style={{fontSize:8,color:"#60a5fa",
                                        background:"#60a5fa15",padding:"0 4px",borderRadius:3}}>{e}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Capability Registry */}
      <Card title="Capability Registry" badge={`${TOOLS.length} tools`} accent="#fb923c">
        <div style={{maxHeight:210,overflowY:"auto",
                     display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>
          {TOOLS.map(t=>(
            <div key={t.name} style={{
              background:"#0a1222",border:"1px solid #0f172a",
              borderRadius:7,padding:"7px 8px",
            }}>
              <div style={{display:"flex",alignItems:"center",gap:4,marginBottom:2}}>
                <span style={{width:5,height:5,borderRadius:"50%",
                              background:RISK_COL[t.risk],flexShrink:0}}/>
                <span style={{fontSize:9,fontWeight:700,color:"#cbd5e1",
                              overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>
                  {t.name}
                </span>
              </div>
              <div style={{fontSize:8,color:"#334155",marginBottom:4,lineHeight:1.3}}>{t.desc}</div>
              <div style={{display:"flex",justifyContent:"space-between"}}>
                <span style={{fontSize:8,color:RISK_COL[t.risk]}}>{t.risk}</span>
                <span style={{fontSize:8,color:"#334155"}}>{t.calls}</span>
              </div>
              <div style={{marginTop:4,height:2,background:"#0f172a",borderRadius:1}}>
                <div style={{width:`${Math.min(100,(t.calls/203)*100)}%`,height:"100%",
                             background:RISK_COL[t.risk],borderRadius:1,opacity:0.55}}/>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>

    <div style={{height:12}}/>

    {/* ── Event Log ── */}
    <Card title="Event Log" badge="live · auto-scroll" accent="#6366f1">
      <div ref={logRef} style={{maxHeight:190,overflowY:"auto",fontFamily:"monospace",fontSize:10}}>
        {events.map((e,i)=>(
          <div key={i} style={{
            display:"grid",gridTemplateColumns:"56px 126px 1fr",
            gap:8,padding:"2px 0",borderBottom:"1px solid #0a1222"
          }}>
            <span style={{color:"#1e293b"}}>{e.ts}</span>
            <span style={{color:EVENT_COLOR[e.kind]||"#475569",fontWeight:600}}>{e.kind}</span>
            <span style={{color:"#334155",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{e.data}</span>
          </div>
        ))}
      </div>
    </Card>

    <div style={{height:12}}/>

    {/* ── Stats strip ── */}
    <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10}}>
      {[
        {lbl:"MEMORIES",  val:"204", sub:"11 compressed",  col:"#60a5fa"},
        {lbl:"GOALS",     val:"7",   sub:"3 active",       col:"#a78bfa"},
        {lbl:"TOOL CALLS",val:"515", sub:"since boot",     col:"#fb923c"},
        {lbl:"UPTIME",    val:"52m", sub:`cycle #${cycle}`,col:"#10b981"},
      ].map(s=>(
        <div key={s.lbl} style={{
          background:"#0d1929",border:"1px solid #0f172a",
          borderRadius:10,padding:"11px 13px",
          borderTop:`2px solid ${s.col}55`,
        }}>
          <div style={{fontSize:8,color:"#334155",letterSpacing:"0.12em",marginBottom:4}}>{s.lbl}</div>
          <div style={{fontSize:22,fontWeight:800,color:s.col,
                       fontFamily:"'Syne',sans-serif",lineHeight:1}}>{s.val}</div>
          <div style={{fontSize:9,color:"#1e293b",marginTop:3}}>{s.sub}</div>
        </div>
      ))}
    </div>
  </div>
</div>
```

);
}