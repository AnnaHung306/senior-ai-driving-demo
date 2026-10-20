import test from 'node:test';
import assert from 'node:assert/strict';
import {validateDynamics,dynamicsSummary,labProjectPoint,validateLabTraces} from '../dynamics.js';
import {createDynamicsLayer} from '../dynamics-view.js';
const base=()=>{
  const times=Array.from({length:16},(_,i)=>i*.2),v=[2,1],p=[8,2],positions=times.map(t=>p.map((x,j)=>x+v[j]*t));
  return {future_observations_used:false,motion_prediction_source_sha256:'a'.repeat(64),objects:[{track_id:'s:T1',center_ego_m:[...p,1],size_lwh_m:[4,2,2],class_id:1,
    motion:{status:'moving',velocity_available:true,velocity_ground_ego_mps:v,speed_mps:Math.sqrt(5),heading_rad:Math.atan2(1,2),velocity_covariance:[[.04,0],[0,.04]],velocity_sigma_max_mps:.2,confidence_calibrated:false},
    history_ego_xy_m:[[7,1.5],p],lifecycle:'confirmed',forecast:{status:'available',coordinate_frame:'fixed_current_ego',probability_calibrated:false,times_s:times,
      positions_ego_xy_m:positions,uncertainty_radius_m:times.map(t=>.2+t*.3),braking_sensitivity_ego_xy_m:positions},risk:{status:'high',TTC_s:1.2,probability_calibrated:false}}],
    dynamics:{GT_identity_or_future_used:false,held_tracks_used_for_occupancy:false,forecast_coordinate_frame:'fixed_current_ego',held_unobserved_tracks:[],
      ego_motion:{status:'estimated',source:'RGB_ground_SE2_only',velocity_ego_mps:[4,0],yaw_rate_radps:0,positions_ego_xy_m:times.map(t=>[4*t,0])}}};
};
test('accept causal fixed-current-ego motion and compute visible counts',()=>{
  const f=base();assert.equal(validateDynamics(f),true);assert.deepEqual(dynamicsSummary(f),{available:1,total:1,risk:'high',nominal_alerts:1,possible_alerts:0,coasting:0});
  assert.equal(validateDynamics({objects:[]}),false);
});
test('actual Three forecast buffers use y,z,-x and do not move measured boxes',()=>{
  const f=base(),before=structuredClone(f),g=createDynamicsLayer(f,'s:T1',{prediction:true,history:true,velocity:true,uncertainty:true,coasting:true});
  const r=g.userData.renderedForecasts[0];assert.equal(r.track_id,'s:T1');assert.equal(r.opacity,1);assert.equal(r.positions_three.length,16);
  f.objects[0].forecast.positions_ego_xy_m.forEach(([x,y],i)=>{assert.ok(Math.abs(r.positions_three[i][0]-y)<1e-5);assert.ok(Math.abs(r.positions_three[i][1]-.11)<1e-5);assert.ok(Math.abs(r.positions_three[i][2]+x)<1e-5);});
  assert.deepEqual(f,before);const off=createDynamicsLayer(f,null,{prediction:false,history:false,velocity:false,uncertainty:false,coasting:false});assert.equal(off.children.length,0);
  const dispose=root=>root.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});dispose(g);dispose(off);
});
test('unobserved tracks prevent an all-clear interpretation',()=>{
  const f=base();f.objects[0].risk.status='not_triggered';f.objects[0].risk.TTC_s=null;
  f.dynamics.held_unobserved_tracks=[{track_id:'s:T2',measurement_current:false,used_for_occupancy:false,center_ego_m:[10,2,1],size_lwh_m:[4,2,1],age_since_measurement_s:.4}];
  assert.equal(validateDynamics(f),true);assert.equal(dynamicsSummary(f).risk,'unknown');
});
for(const [name,mutate] of [
 ['missing source hash',f=>f.motion_prediction_source_sha256='wrong'],['future observation used',f=>f.future_observations_used=true],
 ['nonfinite speed',f=>f.objects[0].motion.speed_mps=NaN],['inconsistent speed',f=>f.objects[0].motion.speed_mps=20],
 ['negative covariance',f=>f.objects[0].motion.velocity_covariance[0][0]=-1],['nonPSD covariance',f=>f.objects[0].motion.velocity_covariance=[[1,2],[2,1]]],
 ['fake calibration',f=>f.objects[0].motion.confidence_calibrated=true],['future GT',f=>f.objects[0].forecast.GT_future=[[1,2]]],
 ['forecast yaw coordinate confusion',f=>f.objects[0].forecast.coordinate_frame='future_ego'],['false motion model',f=>f.objects[0].forecast.positions_ego_xy_m[4]=[999,999]],
 ['unordered future time',f=>f.objects[0].forecast.times_s[5]=.1],['infinite uncertainty',f=>f.objects[0].forecast.uncertainty_radius_m[8]=Infinity],
 ['unavailable velocity has numbers',f=>f.objects[0].motion.velocity_available=false],['unknown risk code',f=>f.objects[0].risk.status='safe'],
 ['out of horizon TTC',f=>f.objects[0].risk.TTC_s=10],['ghost in current measurements',f=>f.dynamics.held_unobserved_tracks=[{track_id:'s:T1',measurement_current:false,used_for_occupancy:false,center_ego_m:[1,2,1],size_lwh_m:[4,2,1],age_since_measurement_s:.2}]]
])test('reject '+name,()=>{const f=base();mutate(f);assert.throws(()=>validateDynamics(f));});
test('diagnostic projection preserves forward and right and rejects invalid synthetic traces',()=>{
  assert.deepEqual(labProjectPoint([10,2]),[374,318]);assert.throws(()=>validateLabTraces([]));
  assert.throws(()=>validateLabTraces([{scenario:'following',scope:'real_RGB',frames:[]} ]));
});
