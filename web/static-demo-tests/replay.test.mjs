import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {validateBundle,frameAtTime,safeAssetPath,checkedBytes} from '../../docs/replay.js';

const bundle=JSON.parse(await readFile(new URL('../../docs/data/route.json',import.meta.url),'utf8'));
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
