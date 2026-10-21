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
