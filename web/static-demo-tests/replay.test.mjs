import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {validateBundle,displayRecords,frameAtTime,safeAssetPath,checkedBytes} from '../../docs/replay.js';

const bundle=JSON.parse(await readFile(new URL('../../docs/data/route.json',import.meta.url),'utf8'));
const perception=JSON.parse(await readFile(new URL('../../docs/data/perception.json',import.meta.url),'utf8'));
const signal=JSON.parse(await readFile(new URL('../../docs/data/signal-turn.json',import.meta.url),'utf8'));

test('new complete perception and signal/turn recordings satisfy distinct contracts',()=>{
  assert.equal(validateBundle(perception),perception);assert.equal(validateBundle(signal),signal);
  const views=displayRecords(perception);assert.equal(views.length,60);assert.equal(views[0].navigation,null);
  assert.equal(views[0].simulation_s,0);assert.equal(views[0].display_only,true);assert.equal(views[0].control,undefined);
  assert.ok(views[0].perception.road_cells_xy_m.every(([x,y])=>x<=35&&x>=-15&&y<=20&&y>=-20));
});
test('known memory episode retains original IDs, reappears and expires without future filling',()=>{
  const frames=displayRecords(perception),id='town01-00:T000003',reappear='town01-00:T000005';
  assert.ok(frames[0].perception.objects.some(o=>o.track_id===id));
  for(const i of [2,3,4])assert.ok(frames[i].perception.dynamics.held_unobserved_tracks.some(o=>o.track_id===id));
  assert.ok(frames[4].perception.objects.some(o=>o.track_id===reappear));
  assert.ok(!frames[5].perception.objects.some(o=>o.track_id===id));
  assert.ok(!frames[5].perception.dynamics.held_unobserved_tracks.some(o=>o.track_id===id));
});
test('a held object cannot be marketed as a current observation or permanent memory',()=>{
  for(const field of ['measurement_current','age_since_measurement_s','used_for_occupancy']) {
    const bad=structuredClone(perception),h=bad.records[2].dynamics.held_unobserved_tracks[0];
    h[field]=field==='age_since_measurement_s'?1:true;assert.throws(()=>validateBundle(bad));
  }
  const bad=structuredClone(perception);bad.records[0].control_issued=true;assert.throws(()=>validateBundle(bad));
});
test('actual signal run contains stationary RGB red, green, then a real left turn and arrival',()=>{
  assert.ok(signal.records.some(r=>r.plan.state==='SIGNAL_STOP'&&r.speed_mps<.01&&r.relevant_RGB_signals.some(s=>s.phase==='red')));
  assert.ok(signal.records[97].relevant_RGB_signals.some(s=>s.phase==='green'));
  assert.equal(signal.records[97].plan.state,'SIGNAL_STOP'); // Preserve the real release delay.
  assert.ok(signal.records.at(-1).ego_pose_navigation_prior[2]<-1.3);
  assert.equal(signal.records.at(-1).plan.state,'ARRIVED');
});
test('all 67 committed real records satisfy the replay and live geometry contracts',()=>assert.equal(validateBundle(bundle),bundle));
test('time lookup cannot reveal future frames and clamps at boundaries',()=>{
  const records=[{simulation_s:.1},{simulation_s:.3},{simulation_s:.5}];
  assert.equal(frameAtTime(records,0),0);assert.equal(frameAtTime(records,.29),0);
  assert.equal(frameAtTime(records,.3),1);assert.equal(frameAtTime(records,10),2);
  assert.throws(()=>frameAtTime(records,NaN));
});
test('a cross-frame RGB swap is rejected',()=>{
  const bad=structuredClone(bundle);bad.frame_assets[0]=bad.frame_assets[1];assert.throws(()=>validateBundle(bad));
});
test('GT contamination cannot enter the main model scene',()=>{
  const bad=structuredClone(bundle);bad.records[0].other_actor_GT_in_planning=true;assert.throws(()=>validateBundle(bad));
});
test('duplicate simulation timestamps cannot enter playback',()=>{
  const bad=structuredClone(bundle);bad.records[1].simulation_s=bad.records[0].simulation_s;assert.throws(()=>validateBundle(bad));
});
test('paths cannot escape the site or load an external endpoint',()=>{
  for(const path of ['../a','images/../a','https://example.com/a','//example.com/a','images/a?token=x'])assert.throws(()=>safeAssetPath(path));
  assert.equal(safeAssetPath('images/69210/front.jpg'),'images/69210/front.jpg');
});
test('corrupt byte content or HTTP error stops replay instead of accepting stale data',async()=>{
  const good=new Response(new TextEncoder().encode('abc'));
  assert.equal((await checkedBytes(good,'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')).byteLength,3);
  await assert.rejects(()=>checkedBytes(new Response('abd'),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'));
  await assert.rejects(()=>checkedBytes(new Response('',{status:404}),'0'.repeat(64)));
});
