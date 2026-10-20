import {validateDynamics} from './dynamics.js';
export const CLASS_NAMES = {1:'汽車',2:'行人',3:'自行車',4:'機車',5:'卡車',6:'公車'};
export const COLORS = {1:0x69a9ed,2:0xf0a0bc,3:0x71d7b5,4:0xefbd76,5:0xb4a0ed,6:0x79d1da};
export const toScene = ([x,y,z=0]) => [y,z,-x];
export const fromScene = ([X,Y,Z]) => [-Z,X,Y];
export const sceneDimensions = ([length,width,height]) => [width,height,length];
export const sceneYaw = yaw => -yaw;

export function egoCorners(object) {
  const [L,W,H]=object.size_lwh_m, [x,y,z]=object.center_ego_m;
  const c=Math.cos(object.yaw_rad),s=Math.sin(object.yaw_rad),out=[];
  for (const a of [-1,1]) for (const b of [-1,1]) for (const d of [-1,1])
    out.push([x+c*a*L/2-s*b*W/2,y+s*a*L/2+c*b*W/2,z+d*H/2]);
  return out;
}

export function gridPoint(grid,row,col,layer=null) {
  const r=grid.resolution_m;
  return [grid.x_max-(row+.5)*r,grid.y_min+(col+.5)*r,layer===null?0:grid.z_min+(layer+.5)*r];
}

export function playbackIndex(timestamps,startIndex,elapsedSeconds) {
  if (!timestamps.length || startIndex<0 || startIndex>=timestamps.length || !Number.isFinite(elapsedSeconds) || elapsedSeconds<0)
    throw Error('Invalid playback clock');
  const target=timestamps[startIndex]+elapsedSeconds;
  let index=startIndex;
  while (index+1<timestamps.length && timestamps[index+1]<=target+1e-9) index++;
  return {index,ended:target>=timestamps.at(-1)-1e-9};
}

export function validateDisplayFrame(frame) {
  const vector=(a,n)=>Array.isArray(a)&&a.length===n&&a.every(Number.isFinite);
  const safe=p=>typeof p==='string'&&/^(frames|rgb)\//.test(p)&&!p.includes('..')&&!/[\\:%?#]/.test(p);
  if(frame.schema_version!==1||frame.source_mode!=='recorded_inference_replay'||frame.coordinate_frame!=='current_ego'||frame.axes!=='x_forward_y_right_z_up'
    ||frame.GT_used_in_inference!==false||frame.control_issued!==false||!Number.isFinite(frame.timestamp_s)||!Number.isInteger(frame.frame)) throw Error('不支援或不完整的 WorldFrame');
  if(frame.units?.position!=='meters'||frame.units?.size!=='full_LWH_meters'||frame.units?.angle!=='radians') throw Error('WorldFrame 單位不一致');
  const ids=new Set();
  if(!Array.isArray(frame.objects)) throw Error('缺少物件輸出');
  for(const object of frame.objects) {
    if(!CLASS_NAMES[object.class_id]||!vector(object.center_ego_m,3)||!vector(object.size_lwh_m,3)||object.size_lwh_m.some(x=>x<=0)
      ||!Number.isFinite(object.yaw_rad)||Math.abs(object.yaw_rad)>Math.PI+1e-9||!Number.isFinite(object.score_uncalibrated)||object.score_uncalibrated<0||object.score_uncalibrated>1
      ||typeof object.track_id!=='string'||!object.track_id.startsWith(frame.scene_id+':T')||ids.has(object.track_id)||object.measurement_current!==true) throw Error('三維物件資料不合法');
    if('actor_id' in object||'world_pose' in object||'world_vertices_m' in object) throw Error('主場景禁止 GT 物件');
    ids.add(object.track_id);
  }
  const ground=frame.ground;
  if(!vector(ground?.shape,2)||!ground.shape.every(x=>Number.isInteger(x)&&x>0)||ground.shape[0]*ground.shape[1]>250000
     ||ground.labels_row_major?.length!==ground.shape[0]*ground.shape[1]||!ground.labels_row_major.every(x=>[1,2,9,255].includes(x))) throw Error('道路格點尺寸或語意錯誤');
  if(!Number.isFinite(ground.grid.resolution_m)||ground.grid.resolution_m<=0) throw Error('道路尺度錯誤');
  if(!vector(frame.occupancy?.shape,3)||!frame.occupancy.shape.every(x=>Number.isInteger(x)&&x>0)||frame.occupancy.shape.reduce((a,b)=>a*b,1)>1000000
    ||!safe(frame.occupancy.file)||!/^[a-f0-9]{64}$/.test(frame.occupancy.sha256)) throw Error('Occupancy 格式錯誤');
  for(const [grid,shape] of [[ground.grid,ground.shape],[frame.occupancy_grid,frame.occupancy.shape]]) {
    const r=grid?.resolution_m;
    if(!Number.isFinite(r)||r<=0)throw Error('網格解析度錯誤');
    for(const [i,axis] of ['x','y',...(shape.length===3?['z']:[])].entries()) {
      const lo=grid[axis+'_min'],hi=grid[axis+'_max'];
      if(!Number.isFinite(lo)||!Number.isFinite(hi)||hi<=lo||Math.abs((hi-lo)/r-shape[i])>1e-7)throw Error('網格範圍與尺寸不一致');
    }
  }
  if(frame.occupancy.dtype!=='uint8'||frame.occupancy.order!=='C'||ground.height_measured!==false||ground.plane_z_m!==0
    ||frame.road_sidewalk_interfaces?.curb_height_measured!==false||frame.lane_paint?.cell_size_m!==.2
    ||!Number.isFinite(frame.occupancy_unknown_fraction)||frame.occupancy_unknown_fraction<0||frame.occupancy_unknown_fraction>1
    ||frame.quality?.formal_quality_passed!==false)throw Error('顯示語意或品質聲明錯誤');
  if(!Array.isArray(frame.lane_paint?.centers_ego_xy_m)||!frame.lane_paint.centers_ego_xy_m.every(p=>vector(p,2))) throw Error('標線包含非有限座標');
  if(!Array.isArray(frame.road_sidewalk_interfaces?.segments_ego_xy_m)||!frame.road_sidewalk_interfaces.segments_ego_xy_m.every(s=>Array.isArray(s)&&s.length===2&&s.every(p=>vector(p,2)))) throw Error('道路界面錯誤');
  if(!['candidate','unavailable'].includes(frame.local_path?.status)||frame.local_path.control_issued!==false
    ||!Array.isArray(frame.local_path.points_ego_m)||!frame.local_path.points_ego_m.every(p=>vector(p,2))
    ||(frame.local_path.status==='unavailable'&&frame.local_path.points_ego_m.length)) throw Error('候選路徑錯誤');
  if(!frame.cameras||Object.keys(frame.cameras).length!==8||!Object.values(frame.cameras).every(c=>safe(c.file)&&/^[a-f0-9]{64}$/.test(c.sha256))) throw Error('八鏡頭來源不完整');
  if(!Object.values(frame.timing_ms).every(x=>Number.isFinite(x)&&x>=0)) throw Error('延遲資料錯誤');
  validateDynamics(frame);
  return true;
}
