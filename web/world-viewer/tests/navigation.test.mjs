import test from 'node:test';import assert from 'node:assert/strict';
import {validateLiveRecord,mapPoint,inverseMapPoint,acceptSnapshot} from '../navigation-contract.js';
const record=()=>({schema_version:1,frame:2,speed_mps:2,ego_pose_navigation_prior:[0,0,0],other_actor_GT_in_planning:false,formal_quality_passed:false,
 perception:{objects:[],GT_other_actors_used:false},navigation:{remaining_m:10},plan:{actor_GT_used:false,local_path_xy_m:[[1,0],[2,0]]},control:{throttle:.2,brake:0,steer:.1}});
test('navigation projection round trip',()=>{const b={minX:-50,maxX:400,minY:0,maxY:350};const p=[123,45];const v=inverseMapPoint(mapPoint(p,b,420,360),b,420,360);assert.ok(Math.hypot(...v.map((x,i)=>x-p[i]))<1e-10);});
test('valid map RGB navigation record',()=>assert.equal(validateLiveRecord(record()),true));
test('late poll cannot overwrite newer seek or start state',()=>{assert.equal(acceptSnapshot({instance_id:'a',revision:4},{instance_id:'a',revision:3}),false);assert.equal(acceptSnapshot({instance_id:'a',revision:4},{instance_id:'a',revision:5}),true);assert.equal(acceptSnapshot({instance_id:'a',revision:4},{instance_id:'new',revision:0}),true);});
for(const[name,change]of[['GT',r=>r.other_actor_GT_in_planning=true],['false quality',r=>r.formal_quality_passed=true],['nonfinite speed',r=>r.speed_mps=NaN],['double pedals',r=>r.control.brake=.5],['bad path',r=>r.plan.local_path_xy_m=[[1,NaN]]]])test('reject navigation '+name,()=>{const r=record();change(r);assert.throws(()=>validateLiveRecord(r));});
