import {validateDisplayFrame} from './math.js';

export async function hashHex(buffer) {
  const hash=await crypto.subtle.digest('SHA-256',buffer);
  return Array.from(new Uint8Array(hash),n=>n.toString(16).padStart(2,'0')).join('');
}

export class PredictionClient {
  constructor(){this.manifest=null;this.cache=new Map();this.inflight=new Map();}
  async initialize(){
    const response=await fetch('/api/manifest',{cache:'no-store'});
    if(!response.ok)throw Error('無法讀取已驗證的資料清單');
    const m=await response.json();
    if(m.schema_version!==1||m.GT_geometry_exported!==false||m.source_mode!=='recorded_inference_replay'||!m.frames?.length)throw Error('資料來源不受支援');
    const keys=new Set();const previous=new Map();
    for(const frame of m.frames){
      if(keys.has(frame.key)||m.file_inventory[frame.file]!==frame.sha256||!Number.isFinite(frame.timestamp_s)||frame.timestamp_s<=(previous.get(frame.scene_id)??-Infinity))throw Error('資料清單重複、倒序或雜湊未綁定');
      keys.add(frame.key);previous.set(frame.scene_id,frame.timestamp_s);
    }
    this.manifest=m;return m;
  }
  async verified(relative,digest){
    if(!/^(frames|rgb)\//.test(relative)||relative.includes('..')||/[\\:%?#]/.test(relative)||this.manifest.file_inventory[relative]!==digest)throw Error('拒絕未綁定的顯示檔案');
    const response=await fetch('/data/'+relative,{cache:'no-store'});
    if(!response.ok)throw Error('檔案無法取得或來源遭改動');
    const bytes=await response.arrayBuffer();
    if(await hashHex(bytes)!==digest)throw Error('顯示檔案 SHA256 不符');
    return bytes;
  }
  async load(key){
    if(this.cache.has(key))return this.cache.get(key);
    if(this.inflight.has(key))return this.inflight.get(key);
    const promise=this.loadActual(key).finally(()=>this.inflight.delete(key));this.inflight.set(key,promise);return promise;
  }
  async loadActual(key){
    const row=this.manifest.frames.find(r=>r.key===key);if(!row)throw Error('未知影格');
    const data=await this.verified(row.file,row.sha256);const frame=JSON.parse(new TextDecoder().decode(data));
    validateDisplayFrame(frame);
    if(frame.scene_id!==row.scene_id||frame.frame!==row.frame||frame.timestamp_s!==row.timestamp_s||frame.objects.length!==row.objects)throw Error('影格身分與索引不一致');
    if(!/^[a-f0-9]{64}$/.test(frame.world_source_sha256)||!/^[a-f0-9]{64}$/.test(frame.parent_bev_sha256))throw Error('缺少上游來源雜湊');
    const binary=await this.verified(frame.occupancy.file,frame.occupancy.sha256);
    if(binary.byteLength!==frame.occupancy.shape.reduce((a,b)=>a*b,1))throw Error('Occupancy 位元組不完整');
    const occupancy=new Uint8Array(binary);
    if(!occupancy.every(x=>[0,1,2,3,4,5,6,7,8,9,255].includes(x)))throw Error('未知的 Occupancy 類別');
    if(Math.abs(occupancy.filter(x=>x===255).length/occupancy.length-frame.occupancy_unknown_fraction)>1e-12)throw Error('未知空間比例不一致');
    const result={frame,occupancy,source:row};this.cache.set(key,result);
    if(this.cache.size>16)this.cache.delete(this.cache.keys().next().value);
    return result;
  }
  async camera(frame,name){
    const source=frame.cameras[name];if(!source)throw Error('未知鏡頭');
    return URL.createObjectURL(new Blob([await this.verified(source.file,source.sha256)],{type:'image/png'}));
  }
}
