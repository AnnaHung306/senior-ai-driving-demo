import {PredictionClient} from './data-client.js';
import {RISK_LABELS,validateLabTraces,labProjectPoint} from './dynamics.js';
const $=id=>document.getElementById(id),client=new PredictionClient();
const names={following:['穩定跟車','兩車保持等速時，先累積觀測再估計地面速度。未來虛線只代表固定速度假設。'],
  turning:['轉彎','物件逐步改變行進方向；模型需要新的量測才能修正，較長預測會偏離彎路。'],
  cut_in:['側向切入','物件在第三秒開始側向靠近車道。變化發生前，模型不會預知切入意圖。'],
  sudden_brake:['突然煞車','前車第三秒後減速。速度估計需要時間追上變化；固定速度預測可能低估接近風險。']};
let traces=[],current=null,index=0,playing=false,start=0,startIndex=0;
const fmt=x=>Number.isFinite(x)?x.toFixed(1):'—';
const svg=(tag,attrs={},text=null)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));if(text!==null)e.textContent=text;return e;};
const projectPoint=labProjectPoint;
function pause(){playing=false;$('labPlay').textContent='▶';$('labPlay').setAttribute('aria-label','播放診斷情境');}
function fail(e){pause();$('labStatus').textContent='資料未套用 · '+e.message;$('labStatus').classList.add('lab-error');}
function draw(i){
  if(!current)return;index=Math.max(0,Math.min(i,current.frames.length-1));const f=current.frames[index],scene=$('labScene');scene.replaceChildren();
  scene.dataset.scenario=current.scenario;scene.dataset.frame=String(index);
  const mobile=scene.clientWidth<500;scene.setAttribute('viewBox',mobile?'140 0 420 530':'0 0 700 530');
  scene.append(svg('rect',{x:285,y:0,width:130,height:530,fill:'#2d4254'}));
  for(const x of [285,415])scene.append(svg('line',{x1:x,y1:0,x2:x,y2:530,stroke:'#b6b291','stroke-width':2}));
  scene.append(svg('line',{x1:350,y1:0,x2:350,y2:530,stroke:'#b6b291','stroke-width':2,'stroke-dasharray':'14 14'}));
  for(const m of [0,10,20,30])scene.append(svg('text',{x:mobile?158:20,y:438-m*12,fill:'#698298','font-size':11},m+' m'));
  function path(points,color,dashed=false){if(points.length<2)return;scene.append(svg('polyline',{points:points.map(projectPoint).map(p=>p.join(',')).join(' '),fill:'none',stroke:color,'stroke-width':2,'stroke-dasharray':dashed?'6 5':'none'}));}
  if(f.ego.status==='estimated')path(f.ego.positions_ego_xy_m,'#a2b4c555',true);
  const risk=f.risk.status,color=['high','watch'].includes(risk)?'#efad91':risk==='potential'?'#d8bd8f':'#7ed8bf';
  if(f.forecast.status==='available'){
    path(f.forecast.positions_ego_xy_m,color,true);
    for(const k of [5,10,15]){
      const [x,y]=projectPoint(f.forecast.positions_ego_xy_m[k]);scene.append(svg('circle',{cx:x,cy:y,r:4,fill:color}));
      if($('labUncertainty').checked)scene.append(svg('circle',{cx:x,cy:y,r:f.forecast.uncertainty_radius_m[k]*12,fill:'none',stroke:color,'stroke-opacity':.17}));
    }
  }
  const [x,y]=projectPoint(f.measured_xy);scene.append(svg('rect',{x:x-11,y:y-26,width:22,height:52,rx:5,fill:color,stroke:'#cde4e0','stroke-width':1}));
  scene.append(svg('rect',{x:x-8,y:y-16,width:16,height:10,fill:'#2b5261',rx:2}));
  scene.append(svg('rect',{x:338,y:412,width:24,height:54,rx:5,fill:'#dbe7ed'}));scene.append(svg('rect',{x:342,y:419,width:16,height:12,rx:2,fill:'#7b9aaf'}));
  scene.append(svg('text',{x:350,y:490,'text-anchor':'middle',fill:'#c2d7e4','font-size':12},'自車 · 合成量測'));
  scene.append(svg('text',{x,y:y-38,'text-anchor':'middle',fill:color,'font-size':12},'物件量測 · '+(f.motion.velocity_available?fmt(f.motion.speed_mps*3.6)+' km/h':'累積觀測')));
  if(y>500||y<30)scene.append(svg('text',{x:350,y:510,'text-anchor':'middle',fill:'#d8bd8f','font-size':12},y>500?'物件位於後方視窗之外':'物件位於前方視窗之外'));
  $('labSlider').value=String(index);$('labTime').textContent=fmt(f.time_s)+' / 10 s';$('labRisk').textContent=RISK_LABELS[risk];$('labRisk').className='risk-badge '+risk;
  $('labSpeed').textContent=f.motion.velocity_available?fmt(f.motion.speed_mps*3.6)+' km/h':'累積觀測';$('labTTC').textContent=f.risk.TTC_s!==null?fmt(f.risk.TTC_s)+' s':'未提供';
  $('labForecast').textContent=f.forecast.status==='available'?'固定速度 · 3 秒':'尚未可用';
  $('labDiagnostic').value=JSON.stringify({source_mode:'synthetic_geometry_not_AI_or_CARLA',scenario:current.scenario,frame:index,time_s:f.time_s,
    measured_xy:f.measured_xy,forecast:f.forecast,motion:f.motion,risk:f.risk,projected_measured_xy:[x,y]},null,2);
}
function choose(name){pause();current=traces.find(t=>t.scenario===name);if(!current)throw Error('未知診斷情境');
  $('scenarioTitle').textContent=names[name][0];$('scenarioDescription').textContent=names[name][1];
  for(const b of $('scenarioTabs').children){b.classList.toggle('active',b.dataset.scenario===name);b.setAttribute('aria-pressed',String(b.dataset.scenario===name));}draw(0);
}
function tick(now){if(playing&&current){const i=Math.min(current.frames.length-1,startIndex+Math.floor((now-start)/200));if(i!==index)draw(i);if(i===current.frames.length-1)pause();}requestAnimationFrame(tick);}
async function startLab(){
  try{
    const manifest=await client.initialize(),ref=manifest.synthetic_diagnostics;if(!ref||ref.scope!=='synthetic_geometry_not_AI_or_CARLA')throw Error('此資料集未提供合成診斷');
    traces=JSON.parse(new TextDecoder().decode(await client.verified(ref.file,ref.sha256)));
    validateLabTraces(traces);
    for(const t of traces){const b=document.createElement('button');b.dataset.scenario=t.scenario;b.textContent=names[t.scenario][0];b.addEventListener('click',()=>choose(t.scenario));$('scenarioTabs').append(b);}
    choose('following');$('labStatus').textContent='來源已驗證 · 合成診斷，非 RGB 模型成績';requestAnimationFrame(tick);
  }catch(e){fail(e);}
}
$('labPrev').addEventListener('click',()=>{pause();draw(index-1);});$('labNext').addEventListener('click',()=>{pause();draw(index+1);});
$('labSlider').addEventListener('input',()=>{pause();draw(Number($('labSlider').value));});$('labUncertainty').addEventListener('change',()=>draw(index));
$('labPlay').addEventListener('click',()=>{if(playing){pause();return;}if(!current)return;if(index===50)draw(0);startIndex=index;start=performance.now();playing=true;$('labPlay').textContent='Ⅱ';$('labPlay').setAttribute('aria-label','暫停診斷情境');});
$('labRetry').addEventListener('click',()=>location.reload());document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});startLab();
new ResizeObserver(()=>draw(index)).observe($('labScene'));
