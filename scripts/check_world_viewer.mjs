// Node reads only the export and independent Python corner receipt, never browser state.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import path from 'node:path';
import * as THREE from '../web/world-viewer/vendor/three.module.js';
import {PredictionClient} from '../web/world-viewer/data-client.js';
import {toScene,sceneDimensions,sceneYaw} from '../web/world-viewer/math.js';
const [bundle,proofPath]=process.argv.slice(2),read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const proof=await read(proofPath),manifest=await read(path.join(bundle,'bundle.json'));
globalThis.fetch=async url=>{
  if(url==='/api/manifest')return Response.json(manifest);
  if(!url.startsWith('/data/'))return new Response('',{status:404});
  return new Response(await fs.readFile(path.join(bundle,url.slice(6))));
};
const client=new PredictionClient();await client.initialize();let count=0;
for(const row of proof.geometry_for_node){
  const {frame}=await client.load(row.key);
  for(const [i,o] of frame.objects.entries()){
    const mesh=new THREE.Mesh(new THREE.BoxGeometry(...sceneDimensions(o.size_lwh_m)));mesh.position.set(...toScene(o.center_ego_m));mesh.rotation.y=sceneYaw(o.yaw_rad);mesh.updateMatrixWorld();
    const expected=row.objects[i];assert.equal(o.track_id,expected.track_id);
    for(let j=0;j<mesh.geometry.attributes.position.count;j++){
      const v=new THREE.Vector3().fromBufferAttribute(mesh.geometry.attributes.position,j).applyMatrix4(mesh.matrixWorld).toArray();
      assert.ok(expected.corners_three.some(c=>c.every((x,k)=>Math.abs(x-v[k])<1e-6)),`${row.key} ${o.track_id} corner mismatch`);
    }
    mesh.geometry.dispose();mesh.material.dispose();count++;
  }
  for(const name of Object.keys(frame.cameras)){const url=await client.camera(frame,name);URL.revokeObjectURL(url);}
}
assert.ok(client.cache.size<=16);assert.equal(client.inflight.size,0);
console.log(JSON.stringify({status:'passed_real_bundle_client_and_Three_corners',frames:proof.frames,objects:count,cameras:proof.frames*8,cache_size:client.cache.size,GT_geometry_rendered:false}));
