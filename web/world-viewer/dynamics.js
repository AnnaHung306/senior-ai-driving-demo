export const MOTION_LABELS={warming_up:'累積觀測',moving:'移動中',stationary:'近乎靜止',uncertain:'運動待確認'};
export const RISK_LABELS={high:'可能快速接近',watch:'預測路徑接近',potential:'不確定範圍重疊',unknown:'資訊尚不足',not_triggered:'假設內未觸發'};
export const RISK_COLORS={high:0xf28c7d,watch:0xf1bd77,potential:0xcbb78d,unknown:0x8393a5,not_triggered:0x7ed8bf};
export const labProjectPoint=([x,y])=>[350+y*12,438-x*12];
const vector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
const truthForbidden=v=>{
  if(Array.isArray(v))return v.some(truthForbidden);
  if(v&&typeof v==='object')return Object.keys(v).some(k=>['actor_id','world_pose','world_vertices_m','GT_future','true_velocity','true_trajectory','depth'].includes(k))||Object.values(v).some(truthForbidden);
  return false;
};
export function validateDynamics(frame){
  if(!frame.dynamics)return false;
  const d=frame.dynamics;
  if(truthForbidden(d)||frame.future_observations_used!==false||d.GT_identity_or_future_used!==false||d.held_tracks_used_for_occupancy!==false
    ||d.forecast_coordinate_frame!=='fixed_current_ego'||!/^[a-f0-9]{64}$/.test(frame.motion_prediction_source_sha256))throw Error('動態推論來源不合法');
  const ids=new Set(frame.objects.map(o=>o.track_id));
  for(const o of frame.objects){
    const m=o.motion,f=o.forecast,r=o.risk;
    if(truthForbidden([m,f,r])||!MOTION_LABELS[m?.status]||typeof m.velocity_available!=='boolean'||m.confidence_calibrated!==false
      ||!Array.isArray(m.velocity_covariance)||m.velocity_covariance.length!==2||!m.velocity_covariance.every(v=>vector(v,2))
      ||!Number.isFinite(m.velocity_sigma_max_mps)||m.velocity_sigma_max_mps<0)throw Error('速度估計資料錯誤');
    const P=m.velocity_covariance;
    if(Math.abs(P[0][1]-P[1][0])>1e-7||P[0][0]<0||P[1][1]<0||P[0][0]*P[1][1]-P[0][1]*P[1][0]<-1e-7)throw Error('速度共變異數錯誤');
    if(m.velocity_available){
      if(!vector(m.velocity_ground_ego_mps,2)||!Number.isFinite(m.speed_mps)||m.speed_mps<0||m.speed_mps>35
        ||Math.abs(Math.hypot(...m.velocity_ground_ego_mps)-m.speed_mps)>1e-7||m.heading_rad!==null&&!Number.isFinite(m.heading_rad))throw Error('速度或方向不一致');
    }else if(m.velocity_ground_ego_mps!==null||m.speed_mps!==null||m.heading_rad!==null)throw Error('尚未可用的速度不得顯示數值');
    if(!Array.isArray(o.history_ego_xy_m)||o.history_ego_xy_m.length>20||!o.history_ego_xy_m.every(p=>vector(p,2)))throw Error('觀測歷史不合法');
    if(!['tentative','confirmed'].includes(o.lifecycle)||f?.coordinate_frame!=='fixed_current_ego'||f.probability_calibrated!==false||!['available','unavailable'].includes(f.status))throw Error('預測座標／狀態錯誤');
    if(f.status==='available'){
      if(!m.velocity_available||f.times_s.length!==16||!f.times_s.every((t,i)=>Number.isFinite(t)&&Math.abs(t-i*.2)<1e-8)
        ||f.positions_ego_xy_m.length!==16||!f.positions_ego_xy_m.every(p=>vector(p,2))
        ||f.braking_sensitivity_ego_xy_m.length!==16||!f.braking_sensitivity_ego_xy_m.every(p=>vector(p,2))
        ||f.uncertainty_radius_m.length!==16||!f.uncertainty_radius_m.every(x=>Number.isFinite(x)&&x>=0))throw Error('三秒預測陣列錯誤');
      for(let i=0;i<16;i++)for(let axis=0;axis<2;axis++)if(Math.abs(f.positions_ego_xy_m[i][axis]-o.center_ego_m[axis]-f.times_s[i]*m.velocity_ground_ego_mps[axis])>1e-6)throw Error('固定速度預測與來源不一致');
    }else if(m.velocity_available||[f.times_s,f.positions_ego_xy_m,f.uncertainty_radius_m,f.braking_sensitivity_ego_xy_m].some(a=>!Array.isArray(a)||a.length))throw Error('不可用預測含有虛構軌跡');
    if(!RISK_LABELS[r?.status]||r.probability_calibrated!==false||r.TTC_s!==null&&(!Number.isFinite(r.TTC_s)||r.TTC_s<0||r.TTC_s>3)
      ||f.status==='unavailable'&&r.status!=='unknown')throw Error('風險輸出錯誤');
  }
  if(!Array.isArray(d.held_unobserved_tracks)||d.held_unobserved_tracks.some(h=>ids.has(h.track_id)||h.measurement_current!==false||h.used_for_occupancy!==false
    ||!vector(h.center_ego_m,3)||!vector(h.size_lwh_m,3)||h.size_lwh_m.some(x=>x<=0)||!Number.isFinite(h.age_since_measurement_s)||h.age_since_measurement_s<0||h.age_since_measurement_s>.65+1e-6))throw Error('暫失軌跡混入當前量測');
  const ego=d.ego_motion;
  if(!['estimated','unavailable'].includes(ego?.status)||ego.source!=='RGB_ground_SE2_only')throw Error('自車運動來源錯誤');
  if(ego.status==='estimated'&&(!vector(ego.velocity_ego_mps,2)||!Number.isFinite(ego.yaw_rate_radps)||ego.positions_ego_xy_m.length!==16||!ego.positions_ego_xy_m.every(p=>vector(p,2))))throw Error('自車未來路徑錯誤');
  return true;
}

export function dynamicsSummary(frame){
  const objects=frame.objects,rank=['high','watch','potential','unknown','not_triggered'];
  const coasting=frame.dynamics?.held_unobserved_tracks.length??0;
  let risk=rank.find(s=>objects.some(o=>o.risk?.status===s))??'unknown';
  if(risk==='not_triggered'&&coasting)risk='unknown';
  return {available:objects.filter(o=>o.motion?.velocity_available).length,total:objects.length,
    risk,
    nominal_alerts:objects.filter(o=>['high','watch'].includes(o.risk?.status)).length,
    possible_alerts:objects.filter(o=>o.risk?.status==='potential').length,coasting};
}

export function validateLabTraces(traces){
  const names=new Set(['following','turning','cut_in','sudden_brake']);
  if(!Array.isArray(traces)||traces.length!==4||new Set(traces.map(t=>t.scenario)).size!==4||truthForbidden(traces))throw Error('診斷情境來源錯誤');
  for(const t of traces){
    if(!names.has(t.scenario)||t.scope!=='synthetic_geometry_not_AI_or_CARLA'||t.frames.length!==51)throw Error('診斷情境格式錯誤');
    for(const [i,r] of t.frames.entries()){
      if(!Number.isFinite(r.time_s)||Math.abs(r.time_s-i*.2)>1e-7||!vector(r.measured_xy,2)||!MOTION_LABELS[r.motion?.status]||!RISK_LABELS[r.risk?.status])throw Error('診斷時間／量測錯誤');
      const f=r.forecast;
      if(f?.coordinate_frame!=='fixed_current_ego'||f.probability_calibrated!==false||!['available','unavailable'].includes(f.status))throw Error('診斷預測來源錯誤');
      if(f.status==='available'){
        if(!vector(r.motion.velocity_ground_ego_mps,2)||f.times_s.length!==16||!f.times_s.every((x,j)=>Number.isFinite(x)&&Math.abs(x-j*.2)<1e-8)
          ||f.positions_ego_xy_m.length!==16||!f.positions_ego_xy_m.every(p=>vector(p,2))||f.uncertainty_radius_m.length!==16||!f.uncertainty_radius_m.every(x=>Number.isFinite(x)&&x>=0))throw Error('診斷軌跡格式錯誤');
      }else if(f.positions_ego_xy_m.length)throw Error('診斷不可用預測不得繪圖');
    }
  }
  return true;
}
