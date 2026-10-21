export function validateLiveRecord(r){
  const v=(x,n)=>Array.isArray(x)&&x.length===n&&x.every(Number.isFinite);
  if(r.schema_version!==1||r.other_actor_GT_in_planning!==false||r.formal_quality_passed!==false||!Number.isInteger(r.frame)||!Number.isFinite(r.speed_mps)||r.speed_mps<0||!v(r.ego_pose_navigation_prior,3))throw Error('導航來源或時間錯誤');
  if(!Array.isArray(r.perception?.objects)||r.perception.GT_other_actors_used!==false||r.plan?.actor_GT_used!==false||!Number.isFinite(r.navigation?.remaining_m))throw Error('缺少可追溯感知／規劃');
  for(const o of r.perception.objects)if(!v(o.center_ego_m,3)||!v(o.size_lwh_m,3)||o.size_lwh_m.some(x=>x<=0)||'actor_id'in o||'world_pose'in o)throw Error('物件幾何或GT污染');
  for(const p of r.plan.local_path_xy_m)if(!v(p,2))throw Error('無效路徑');
  for(const key of ['road_cells_xy_m','lane_cells_xy_m'])if(!Array.isArray(r.perception[key])||r.perception[key].some(p=>!v(p,2)))throw Error('無效道路幾何');
  const ids=new Set();for(const o of r.perception.objects){
    if(typeof o.track_id!=='string'||ids.has(o.track_id)||!Number.isFinite(o.yaw_rad))throw Error('無效ID或朝向');ids.add(o.track_id);
    if(o.forecast?.status==='available'){
      const t=o.forecast.times_s,p=o.forecast.positions_ego_xy_m;
      if(!Array.isArray(t)||!t.length||!t.every((x,i)=>Number.isFinite(x)&&x>=0&&(i===0||x>t[i-1]))||!Array.isArray(p)||p.length!==t.length||p.some(x=>!v(x,2)))throw Error('無效預測軌跡');
    }
  }
  if(!Array.isArray(r.perception.dynamics?.held_unobserved_tracks))throw Error('缺少暫失物件狀態');
  for(const o of r.perception.dynamics.held_unobserved_tracks)if(!v(o.center_ego_m,3)||!v(o.size_lwh_m,3)||o.size_lwh_m.some(x=>x<=0)||!Number.isFinite(o.age_since_measurement_s)||o.age_since_measurement_s<0||o.measurement_current!==false||o.used_for_occupancy!==false||'actor_id'in o)throw Error('暫失物件幾何或GT污染');
  for(const k of ['throttle','brake','steer'])if(!Number.isFinite(r.control[k])||Math.abs(r.control[k])>1)throw Error('無效控制');
  if(r.control.throttle>0&&r.control.brake>0)throw Error('油門煞車同時啟動');
  return true;
}
export function mapPoint(xy,bounds,width,height){const s=Math.min((width-32)/(bounds.maxX-bounds.minX),(height-32)/(bounds.maxY-bounds.minY));return [16+(xy[0]-bounds.minX)*s,16+(xy[1]-bounds.minY)*s];}
export function inverseMapPoint(p,bounds,width,height){const s=Math.min((width-32)/(bounds.maxX-bounds.minX),(height-32)/(bounds.maxY-bounds.minY));return [bounds.minX+(p[0]-16)/s,bounds.minY+(p[1]-16)/s];}
export function acceptSnapshot(previous,next){return !previous||previous.instance_id!==next.instance_id||next.revision>=previous.revision;}
