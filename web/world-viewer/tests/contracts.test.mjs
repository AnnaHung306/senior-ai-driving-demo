import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../vendor/three.module.js';
import {toScene,fromScene,sceneDimensions,sceneYaw,egoCorners,gridPoint,playbackIndex,validateDisplayFrame} from '../math.js';
import {PredictionClient,hashHex} from '../data-client.js';

const near=(a,b)=>a.forEach((x,i)=>assert.ok(Math.abs(x-b[i])<1e-6,`${a} != ${b}`));
const base=()=>({schema_version:1,source_mode:'recorded_inference_replay',coordinate_frame:'current_ego',axes:'x_forward_y_right_z_up',
  GT_used_in_inference:false,control_issued:false,timestamp_s:1,frame:1,scene_id:'s',units:{angle:'radians',position:'meters',size:'full_LWH_meters'},
  objects:[{class_id:1,center_ego_m:[8,3,1],size_lwh_m:[4,2,2],yaw_rad:0,score_uncalibrated:.8,track_id:'s:T1',measurement_current:true}],
  ground:{shape:[2,2],grid:{x_min:0,x_max:2,y_min:0,y_max:2,resolution_m:1},labels_row_major:[1,2,255,9],height_measured:false,plane_z_m:0},
  occupancy:{shape:[2,2,2],file:'frames/a.bin',sha256:'a'.repeat(64),dtype:'uint8',order:'C'},
  occupancy_grid:{x_min:0,x_max:2,y_min:0,y_max:2,z_min:0,z_max:2,resolution_m:1},occupancy_unknown_fraction:.5,
  lane_paint:{cell_size_m:.2,centers_ego_xy_m:[]},road_sidewalk_interfaces:{curb_height_measured:false,segments_ego_xy_m:[]},
  local_path:{status:'unavailable',points_ego_m:[],control_issued:false},quality:{formal_quality_passed:false},
  cameras:Object.fromEntries(Array.from({length:8},(_,i)=>['c'+i,{file:'rgb/a.png',sha256:'b'.repeat(64)}])),timing_ms:{compute:1}});

test('ego axes round trip, right stays right, forward becomes -Z',()=>{near(toScene([5,3,2]),[3,2,-5]);near(fromScene(toScene([5,-3,2])),[5,-3,2]);});
test('actual Three box corners agree with ego analytic corners at nontrivial yaw',()=>{
  for(const yaw of [0,Math.PI/2,-Math.PI/2,.73,Math.PI]){
    const o={center_ego_m:[8,-3,1.2],size_lwh_m:[4.6,1.7,1.5],yaw_rad:yaw};
    const m=new THREE.Mesh(new THREE.BoxGeometry(...sceneDimensions(o.size_lwh_m)));m.position.set(...toScene(o.center_ego_m));m.rotation.y=sceneYaw(yaw);m.updateMatrixWorld();
    const expected=egoCorners(o).map(toScene),p=m.geometry.attributes.position;
    for(let i=0;i<p.count;i++){const v=new THREE.Vector3().fromBufferAttribute(p,i).applyMatrix4(m.matrixWorld).toArray();assert.ok(expected.some(e=>e.every((x,j)=>Math.abs(x-v[j])<1e-6)));}
    m.geometry.dispose();m.material.dispose();
  }
});
test('grid row points backwards, column right, layer up with half cell offset',()=>{
  const g={x_max:35,y_min:-20,z_min:-1,resolution_m:.5};near(gridPoint(g,0,0,0),[34.75,-19.75,-.75]);near(gridPoint(g,2,3,4),[33.75,-18.25,1.25]);
});
test('timestamp playback respects missing intervals and ends without extrapolation',()=>{
  const t=[10,10.2,10.6,11.4];assert.deepEqual(playbackIndex(t,0,.59),{index:1,ended:false});assert.deepEqual(playbackIndex(t,1,.4),{index:2,ended:false});
  assert.deepEqual(playbackIndex(t,0,100),{index:3,ended:true});assert.throws(()=>playbackIndex(t,0,-1));
});
test('valid units are order independent and empty objects are supported',()=>{const f=base();assert.equal(validateDisplayFrame(f),true);f.objects=[];assert.equal(validateDisplayFrame(f),true);});
for(const [name,mutate] of [
  ['NaN position',f=>f.objects[0].center_ego_m[0]=NaN],['negative full size',f=>f.objects[0].size_lwh_m[1]=-2],
  ['degree yaw',f=>f.objects[0].yaw_rad=90],['duplicate IDs',f=>f.objects.push({...f.objects[0]})],
  ['GT geometry',f=>f.objects[0].actor_id=7],['held track as current',f=>f.objects[0].measurement_current=false],
  ['unknown filled label',f=>f.ground.labels_row_major[2]=0],['grid distortion',f=>f.ground.grid.x_max=9],
  ['bad voxel shape',f=>f.occupancy.shape[2]=3],['fake height',f=>f.ground.height_measured=true],
  ['invented path',f=>f.local_path.points_ego_m=[[0,0],[10,0]]],['unsafe source',f=>f.cameras.c0.file='rgb/../labels/a.png'],
  ['fake quality pass',f=>f.quality.formal_quality_passed=true],['negative compute time',f=>f.timing_ms.compute=-1]
])test('reject '+name,()=>{const f=base();mutate(f);assert.throws(()=>validateDisplayFrame(f));});
test('client verifies SHA bytes, refuses unbound files and poisoned RGB',async()=>{
  const bytes=new TextEncoder().encode('actual pixels'),digest=await hashHex(bytes);const c=new PredictionClient();
  c.manifest={file_inventory:{'rgb/a.png':digest}};const original=globalThis.fetch;
  try{globalThis.fetch=async()=>new Response(bytes);assert.deepEqual(new Uint8Array(await c.verified('rgb/a.png',digest)),bytes);
    await assert.rejects(c.verified('rgb/unknown.png',digest));await assert.rejects(c.verified('../labels/a',digest));
    globalThis.fetch=async()=>new Response('changed');await assert.rejects(c.verified('rgb/a.png',digest),/SHA256/);
  }finally{globalThis.fetch=original;}
});
