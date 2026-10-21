import fs from 'node:fs';import path from 'node:path';import assert from 'node:assert/strict';
import {validateLiveRecord} from '../web/world-viewer/navigation-contract.js';
import {egoCorners} from '../web/world-viewer/math.js';
import * as THREE from '../web/world-viewer/vendor/three.module.js';
const root=path.resolve(import.meta.dirname,'..'),runs=path.join(root,'artifacts/runs');let frames=0,objects=0;
for(const name of fs.readdirSync(runs).filter(n=>n.startsWith('phase89-drive-'))){const folder=path.join(runs,name),reportPath=path.join(folder,'report.json');if(!fs.existsSync(reportPath))continue;const r=JSON.parse(fs.readFileSync(reportPath));if(!r.records)continue;
 for(const row of r.records){const record=JSON.parse(fs.readFileSync(path.join(folder,row.file)));assert.equal(validateLiveRecord(record),true);frames++;
  for(const o of record.perception.objects){const[L,W,H]=o.size_lwh_m,[x,y,z]=o.center_ego_m;const geometry=new THREE.BoxGeometry(W,H,L),mesh=new THREE.Mesh(geometry);mesh.position.set(y,z,-x);mesh.rotation.y=-o.yaw_rad;mesh.updateMatrixWorld(true);const points=geometry.getAttribute('position'),expected=egoCorners(o);
   for(let i=0;i<points.count;i++){const p=new THREE.Vector3().fromBufferAttribute(points,i).applyMatrix4(mesh.matrixWorld);assert.ok(expected.some(q=>Math.hypot(q[0]+p.z,q[1]-p.x,q[2]-p.y)<1e-5));}geometry.dispose();objects++;
  }
 }
}
console.log(JSON.stringify({status:'passed_actual_records_and_Three_cuboid_corners',frames,objects,GT_geometry_used:false}));
