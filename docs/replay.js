/** Pure replay contract shared by the browser and the dependency-free tests.
 * Hashes detect accidental corruption; the unsigned manifest is not an
 * authenticity signature. Geometry remains subject to model error.
 */
import {validateLiveRecord} from './record-contract.js';

export const CAMERA_LABELS = Object.freeze({front:'前方',front_right:'右前',right:'右側',rear_right:'右後',rear:'後方',rear_left:'左後',left:'左側',front_left:'左前'});
export const BEHAVIOR_LABELS = Object.freeze({CRUISE:'循線行駛',FOLLOW:'減速避讓',YIELD_OFFSET:'同車道避讓',SIGNAL_STOP:'號誌停等',OBSTACLE_STOP:'停車等待',FAILSAFE:'失效停止',GOAL_BRAKE:'終點煞停',ARRIVED:'已抵達'});

export function safeAssetPath(path) {
  if (typeof path!=='string' || !/^(?:images|media|data|vendor)\/[A-Za-z0-9_./-]+$/.test(path) || path.split('/').includes('..')) throw Error('資料內含不支援的資產路徑');
  return path;
}

export function validateBundle(bundle) {
  if(bundle.mode==='recorded_perception')return validatePerception(bundle);
  if (bundle.schema_version!==1 || bundle.mode!=='recorded_replay' || !bundle.provenance?.not_live_inference || bundle.provenance?.other_actor_GT_in_main_view!==false) throw Error('回放來源不符');
  if (!Array.isArray(bundle.records) || !bundle.records.length || bundle.records.length!==bundle.frame_assets?.length || bundle.summary?.frames!==bundle.records.length) throw Error('回放影格數不一致');
  if (bundle.summary.status!=='arrived' || bundle.summary.engineering_route_passed!==true || bundle.summary.formal_quality_passed!==false) throw Error('成功路線標記不符');
  let previous=-Infinity;
  for (let i=0;i<bundle.records.length;i++) {
    const record=bundle.records[i], assets=bundle.frame_assets[i];
    validateLiveRecord(record);
    if (record.run_id!==bundle.run_id || !Number.isFinite(record.simulation_s) || record.simulation_s<=previous || record.frame!==assets.frame) throw Error('回放時間或影格順序錯誤');
    previous=record.simulation_s;
    for (const camera of Object.keys(CAMERA_LABELS)) {
      const image=assets.images[camera];
      if (record.RGB[camera]?.frame!==record.frame || image?.source_PNG_sha256!==record.RGB[camera]?.sha256 || !/^[a-f0-9]{64}$/.test(image?.web_JPEG_sha256)) throw Error('RGB來源不一致');
      safeAssetPath(image.url);
      if (image.url!==`images/${record.frame}/${camera}.jpg`) throw Error('RGB影格路徑不符');
    }
  }
  return bundle;
}

function validatePerception(bundle) {
  if(bundle.schema_version!==1 || bundle.kind!=='perception' || bundle.summary?.status!=='perception_only' ||
    bundle.summary.formal_quality_passed!==false || bundle.provenance?.other_actor_GT_in_main_view!==false ||
    bundle.provenance?.control_issued!==false || !bundle.provenance.not_live_inference || bundle.memory_max_coast_s!==.65 ||
    !bundle.records?.length || bundle.records.length!==bundle.frame_assets?.length || bundle.summary.frames!==bundle.records.length)throw Error('感知片段來源不符');
  let previous=-Infinity;
  const vector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
  for(const [i,r] of bundle.records.entries()) {
    const assets=bundle.frame_assets[i];
    if(r.source_mode!=='recorded_inference_replay' || r.scene_id!==bundle.scene_id || r.GT_used_in_inference!==false ||
      r.control_issued!==false || r.future_observations_used!==false || !Number.isFinite(r.timestamp_s) || r.timestamp_s<=previous ||
      r.frame!==assets.frame || r.dynamics?.held_tracks_used_for_occupancy!==false || r.dynamics.GT_identity_or_future_used!==false ||
      r.coordinate_frame!=='current_ego' || r.axes!=='x_forward_y_right_z_up')throw Error('感知影格或因果來源不符');
    previous=r.timestamp_s;
    const ids=new Set();
    for(const o of r.objects) {
      if(!vector(o.center_ego_m,3)||!vector(o.size_lwh_m,3)||o.size_lwh_m.some(x=>x<=0)||!Number.isFinite(o.yaw_rad)||
        !o.track_id||ids.has(o.track_id)||o.measurement_current!==true||'actor_id' in o||'world_pose' in o)throw Error('物件幾何或ID錯誤');
      ids.add(o.track_id);
    }
    for(const h of r.dynamics.held_unobserved_tracks) {
      if(ids.has(h.track_id)||h.measurement_current!==false||h.used_for_occupancy!==false||h.lifecycle!=='coasting'||
        !(h.age_since_measurement_s>0&&h.age_since_measurement_s<=bundle.memory_max_coast_s)||
        !vector(h.center_ego_m,3)||!vector(h.size_lwh_m,3))throw Error('暫失物件與當前觀測混淆');
      ids.add(h.track_id);
    }
    const g=r.ground;
    if(g.labels_row_major.length!==g.shape[0]*g.shape[1]||g.grid.resolution_m<=0||
      !r.lane_paint.centers_ego_xy_m.every(p=>vector(p,2))||!r.local_path.points_ego_m.every(p=>vector(p,2)))throw Error('道路幾何錯誤');
    for(const c of Object.keys(CAMERA_LABELS)) {
      const image=assets.images[c];
      if(image?.source_PNG_sha256!==r.cameras[c]?.sha256||!/^[a-f0-9]{64}$/.test(image?.web_JPEG_sha256)||
        safeAssetPath(image.url)!==`images/${r.frame}/${c}.jpg`)throw Error('感知RGB來源不符');
    }
  }
  return bundle;
}

/** Display adapter only. Original perception JSON has no navigation/control.
 * Ground row zero is x_max-r/2 (the frozen BEV grid convention).
 * Never fill missing observations from a later frame or from simulator actors.
 */
export function displayRecords(bundle) {
  if(bundle.kind!=='perception')return bundle.records;
  const start=bundle.records[0].timestamp_s;
  return bundle.records.map(r=>{
    const g=r.ground.grid,cols=r.ground.shape[1],road=[];
    r.ground.labels_row_major.forEach((label,i)=>{
      if(label===1)road.push([g.x_max-(Math.floor(i/cols)+.5)*g.resolution_m,g.y_min+(i%cols+.5)*g.resolution_m]);
    });
    return {frame:r.frame,simulation_s:r.timestamp_s-start,speed_mps:null,navigation:null,display_only:true,
      plan:{state:'PERCEPTION_ONLY',local_path_xy_m:r.local_path.points_ego_m},
      perception:{objects:r.objects,dynamics:r.dynamics,road_cells_xy_m:road,lane_cells_xy_m:r.lane_paint.centers_ego_xy_m}};
  });
}

/** Return the latest captured observation at a simulation time, never a future frame. */
export function frameAtTime(records,time) {
  if (!Number.isFinite(time)) throw Error('無效回放時間');
  let low=0, high=records.length-1;
  while (low<high) { const mid=Math.ceil((low+high)/2); if (records[mid].simulation_s<=time) low=mid; else high=mid-1; }
  return low;
}

export async function checkedBytes(response,expectedHash) {
  if (!response.ok) throw Error(`資產載入失敗（HTTP ${response.status}）`);
  const bytes=await response.arrayBuffer();
  if (!globalThis.crypto?.subtle) throw Error('來源驗證需要HTTPS或localhost，請依README開啟');
  const digest=await crypto.subtle.digest('SHA-256',bytes);
  const actual=Array.from(new Uint8Array(digest),byte=>byte.toString(16).padStart(2,'0')).join('');
  if (actual!==expectedHash) throw Error('檔案SHA256不一致，已停止回放；請重新載入');
  return bytes;
}
