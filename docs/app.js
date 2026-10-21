/** Read-only static demo. There is deliberately no simulator/control API.
 * Each update waits for the selected RGB bytes to pass SHA and decode before
 * painting the matching 3D record. A generation counter rejects late seeks.
 */
import * as THREE from './vendor/three.module.js';
import {OrbitControls} from './vendor/OrbitControls.js';
import {CAMERA_LABELS,BEHAVIOR_LABELS,validateBundle,frameAtTime,checkedBytes} from './replay.js';

const $=id=>document.getElementById(id);
let bundle=null,index=-1,playing=false,requestGeneration=0,visibleBlob=null,renderer=null;
let playStart=0,simStart=0,busy=false,contextValid=true,scene,camera,controls,roadMesh,laneMesh,ego;
let dynamic=new THREE.Group();
const geometry=new THREE.BoxGeometry(1,1,1);
const materials=new Map();
const coordinates=([x,y,z=0])=>[y,z,-x]; // CARLA ego forward/right/up -> Three right/up/back.
const material=(color,wire=false)=>{
  const key=`${color}/${wire}`;
  if (!materials.has(key)) materials.set(key,new THREE.MeshStandardMaterial({color,wireframe:wire,transparent:wire,opacity:wire ? .45 : 1}));
  return materials.get(key);
};

function notice(text,isError=false) {
  $('notice').textContent=text;$('notice').classList.toggle('error',isError);
}
function pause() {
  playing=false;$('play').textContent='▶';$('play').setAttribute('aria-label','播放回放');
}
function fail(error) {
  pause();notice(String(error.message||error),true);
  // Invalid or mismatched data must not leave stale geometry paired with new RGB.
  $('rgb').hidden=true;$('rgb-loading').hidden=false;$('rgb-loading').textContent='畫面未通過來源驗證';
  $('world').hidden=true;$('player').dataset.verified='false';
  if (dynamic) dynamic.visible=false;
}

function initializeWorld() {
  try { renderer=new THREE.WebGLRenderer({canvas:$('world'),antialias:true}); }
  catch { throw Error('此瀏覽器無法建立WebGL；可改看下方已錄製影片。'); }
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor(0x101c28);
  scene=new THREE.Scene();scene.fog=new THREE.Fog(0x101c28,38,85);
  camera=new THREE.PerspectiveCamera(48,1,.1,110);camera.position.set(0,16,23);
  controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,0,-10);
  controls.enableDamping=true;controls.minDistance=7;controls.maxDistance=60;controls.maxPolarAngle=Math.PI/2-.03;
  scene.add(new THREE.HemisphereLight(0xe7f5ff,0x263744,2.5));
  const sun=new THREE.DirectionalLight(0xffffff,2);sun.position.set(-10,20,5);scene.add(sun);
  scene.add(new THREE.GridHelper(90,90,0x2b4051,0x1b2a38));
  const createCells=(key,color,size)=>{
    const capacity=Math.max(1,...bundle.records.map(r=>r.perception[key].length));
    const mesh=new THREE.InstancedMesh(geometry,material(color),capacity);
    mesh.userData.cellSize=size;mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);scene.add(mesh);return mesh;
  };
  roadMesh=createCells('road_cells_xy_m',0x526b7e,.48);
  laneMesh=createCells('lane_cells_xy_m',0xe2c881,.16);
  ego=new THREE.Group();
  const body=new THREE.Mesh(geometry,material(0xe6edf2));body.scale.set(1.8,.65,4.4);body.position.set(0,.55,0);
  const roof=new THREE.Mesh(geometry,material(0x8aafc4));roof.scale.set(1.6,.55,2.1);roof.position.set(0,1.1,.25);
  ego.add(body,roof);scene.add(ego,dynamic);
  new ResizeObserver(()=>{
    const box=$('viewport').getBoundingClientRect();renderer.setSize(box.width,box.height,false);
    camera.aspect=box.width/box.height;camera.updateProjectionMatrix();
  }).observe($('viewport'));
  $('reset-view').onclick=()=>{camera.position.set(0,16,23);controls.target.set(0,0,-10);controls.update();};
  $('world').addEventListener('webglcontextlost',event=>{event.preventDefault();contextValid=false;fail(Error('3D繪圖已中斷，回放暫停。請重新載入，或觀看下方影片。'));});
}

function updateCells(mesh,points) {
  const matrix=new THREE.Matrix4(),rotation=new THREE.Quaternion(),scale=new THREE.Vector3(mesh.userData.cellSize,.025,mesh.userData.cellSize);
  mesh.count=points.length;
  for (let i=0;i<points.length;i++) {
    matrix.compose(new THREE.Vector3(...coordinates([...points[i],.035])),rotation,scale);mesh.setMatrixAt(i,matrix);
  }
  mesh.instanceMatrix.needsUpdate=true;mesh.computeBoundingSphere();
}

function paintWorld(record) {
  updateCells(roadMesh,record.perception.road_cells_xy_m);updateCells(laneMesh,record.perception.lane_cells_xy_m);
  scene.remove(dynamic);
  // Only per-frame line geometry/material is disposed. Cubes/materials are pooled.
  dynamic.traverse(object=>{if(object.isLine){object.geometry.dispose();object.material.dispose();}});
  dynamic=new THREE.Group();scene.add(dynamic);
  const line=(points,color,dashed=false)=>{
    if(points.length<2)return;
    const shape=new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(...coordinates([...p,.12]))));
    const colorMaterial=dashed?new THREE.LineDashedMaterial({color,dashSize:.35,gapSize:.2}):new THREE.LineBasicMaterial({color});
    const object=new THREE.Line(shape,colorMaterial);if(dashed)object.computeLineDistances();dynamic.add(object);
  };
  line(record.plan.local_path_xy_m,0xa2e6cd);
  const colors={1:0x7eb9f1,2:0xe9a6bd,3:0x8bd1b8,4:0xe5bf7b,5:0xaf9bd6,6:0x7fcbd1};
  for(const object of record.perception.objects) {
    const mesh=new THREE.Mesh(geometry,material(colors[object.class_id]||0xaaaaaa));
    const [length,width,height]=object.size_lwh_m;mesh.scale.set(width,height,length);
    mesh.position.set(...coordinates(object.center_ego_m));mesh.rotation.y=-object.yaw_rad;
    mesh.userData.track_id=object.track_id;dynamic.add(mesh);
    if(object.forecast.status==='available')line(object.forecast.positions_ego_xy_m,0x8babb0,true);
  }
  for(const object of record.perception.dynamics.held_unobserved_tracks) {
    const mesh=new THREE.Mesh(geometry,material(0x8c9ba8,true));
    const [length,width,height]=object.size_lwh_m;mesh.scale.set(width,height,length);
    mesh.position.set(...coordinates(object.center_ego_m));dynamic.add(mesh);
  }
  $('viewport').dataset.frame=String(record.frame);
}

function paintMap(record) {
  const svg=$('map'),ns='http://www.w3.org/2000/svg',route=bundle.route.points_xy_m;
  svg.replaceChildren();
  const xs=route.map(p=>p[0]),ys=route.map(p=>p[1]);
  const minX=Math.min(...xs)-5,maxX=Math.max(...xs)+5,minY=Math.min(...ys)-5,maxY=Math.max(...ys)+5;
  const scale=Math.min(380/(maxX-minX),120/(maxY-minY));
  const xy=p=>[(p[0]-(minX+maxX)/2)*scale+210,(p[1]-(minY+maxY)/2)*scale+75];
  const node=(tag,attrs)=>{const element=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))element.setAttribute(k,v);svg.append(element);return element;};
  const path=route.map((p,i)=>(i?'L':'M')+xy(p).join(',')).join(' ');
  node('path',{d:path,stroke:'#a2e6cd','stroke-width':3,fill:'none'});
  for(const[p,color]of [[route[0],'#a2e6cd'],[route.at(-1),'#e3bf8b']]){const q=xy(p);node('circle',{cx:q[0],cy:q[1],r:5,fill:color});}
  const q=xy(record.ego_pose_navigation_prior);node('circle',{cx:q[0],cy:q[1],r:4,fill:'#ffffff',stroke:'#101c28','stroke-width':1.5});
  $('progress-label').textContent=record.plan.state==='ARRIVED'?'抵達終點':`已行駛 ${Math.min(28,record.navigation.progress_m).toFixed(1)} m`;
}

async function display(nextIndex) {
  const generation=++requestGeneration;
  if(!Number.isInteger(nextIndex)||nextIndex<0||nextIndex>=bundle.records.length)throw Error('影格超出範圍');
  const selectedCamera=$('camera').value,record=bundle.records[nextIndex],asset=bundle.frame_assets[nextIndex].images[selectedCamera];
  const response=await fetch(new URL(asset.url,location.href));
  const bytes=await checkedBytes(response,asset.web_JPEG_sha256);
  const url=URL.createObjectURL(new Blob([bytes],{type:'image/jpeg'}));
  const decoded=new Image();decoded.src=url;
  try { await decoded.decode(); } catch(error) { URL.revokeObjectURL(url);throw error; }
  if(generation!==requestGeneration){URL.revokeObjectURL(url);return false;}
  // Commit 3D, RGB, controls and labels together after source verification.
  if(visibleBlob)URL.revokeObjectURL(visibleBlob);visibleBlob=url;
  $('rgb').src=url;$('rgb').hidden=false;$('rgb').alt=`${CAMERA_LABELS[selectedCamera]} RGB · F${record.frame}`;
  $('rgb').dataset.frame=String(record.frame);$('rgb').dataset.camera=selectedCamera;
  $('rgb-loading').hidden=true;
  paintWorld(record);paintMap(record);index=nextIndex;$('world').hidden=false;
  $('frame-label').textContent=`F${record.frame} · ${CAMERA_LABELS[selectedCamera]}`;
  $('speed').textContent=(record.speed_mps*3.6).toFixed(1);
  $('remaining').textContent=record.navigation.remaining_m.toFixed(1);
  $('behavior').textContent=BEHAVIOR_LABELS[record.plan.state]||record.plan.state;
  const names={1:'車',2:'行人',3:'自行車',4:'機車',5:'卡車',6:'公車'},counts={};
  for(const object of record.perception.objects)counts[object.class_id]=(counts[object.class_id]||0)+1;
  $('objects').textContent=Object.entries(counts).map(([id,n])=>`${names[id]||'物件'} ${n}`).join(' · ')||'無當前物件';
  $('seek').value=String(index);$('count').textContent=`${index+1} / ${bundle.records.length} 幀`;
  $('time').textContent=`${record.simulation_s.toFixed(1)} / ${bundle.summary.simulation_s.toFixed(1)} s`;
  const descriptions={CRUISE:'沿已規劃路線行駛，持續檢查RGB道路與物件。',FOLLOW:'前方有物件，降低車速並保留間距。',OBSTACLE_STOP:'前方走廊有障礙，停車等待。',GOAL_BRAKE:'接近終點，降低車速並準備煞停。',ARRIVED:'此段紀錄已抵達；後續清理煞停實測為0.0 km/h。'};
  $('decision-description').textContent=descriptions[record.plan.state]||'依當時保存的感知與規劃結果顯示。';
  $('previous').disabled=index===0;$('next').disabled=index===bundle.records.length-1;
  notice('已驗證成功路線 · 同幀RGB與AI輸出 · 晴天模擬環境');
  // Diagnostic DOM state is intentionally read-only and contains no GT actors.
  $('player').dataset.index=String(index);$('player').dataset.frame=String(record.frame);
  $('player').dataset.verified='true';return true;
}

async function seek(next) {
  pause();try { await display(next); } catch(error) { fail(error); }
}

function animate(now) {
  if(renderer&&contextValid){controls.update();renderer.render(scene,camera);}
  if(playing&&!busy) {
    const elapsed=(now-playStart)/1000*Number($('rate').value);
    const next=frameAtTime(bundle.records,simStart+elapsed);
    if(next!==index) {
      busy=true;
      // Keep OrbitControls/rendering responsive while fetch/decode is pending.
      display(next).catch(fail).finally(()=>{busy=false;if(index===bundle.records.length-1)pause();});
    }
    if(index===bundle.records.length-1)pause();
  }
  requestAnimationFrame(animate);
}

async function init() {
  const manifestResponse=await fetch('./manifest.json');
  if(!manifestResponse.ok)throw Error('manifest無法載入；請確認伺服器的根目錄為docs');
  const manifest=await manifestResponse.json();
  const bytes=await checkedBytes(await fetch('./data/route.json'),manifest.bundle_sha256);
  bundle=validateBundle(JSON.parse(new TextDecoder().decode(bytes)));
  $('camera').replaceChildren(...Object.entries(CAMERA_LABELS).map(([value,label])=>new Option(label,value)));
  initializeWorld();requestAnimationFrame(animate);
  $('seek').max=String(bundle.records.length-1);
  await display(0);
  for(const id of ['play','seek','camera'])$(id).disabled=false;
  $('play').onclick=async()=>{
    if(playing){pause();return;}
    if(index===bundle.records.length-1) {try {await display(0);}catch(error){fail(error);return;}}
    simStart=bundle.records[index].simulation_s;playStart=performance.now();playing=true;
    $('play').textContent='Ⅱ';$('play').setAttribute('aria-label','暫停回放');
  };
  $('previous').onclick=()=>seek(Math.max(0,index-1));$('next').onclick=()=>seek(Math.min(bundle.records.length-1,index+1));
  $('seek').oninput=()=>seek(Number($('seek').value));
  $('camera').onchange=()=>seek(Math.max(0,index));
  $('rate').onchange=()=>{if(playing){simStart=bundle.records[index].simulation_s;playStart=performance.now();}};
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  // Keyboard shortcuts are scoped to the player and never hijack form controls.
  $('player').addEventListener('keydown',event=>{
    if(/^(INPUT|SELECT|BUTTON|A)$/.test(event.target.tagName))return;
    if(event.code==='Space'){event.preventDefault();$('play').click();}
    if(event.code==='ArrowRight'){event.preventDefault();$('next').click();}
    if(event.code==='ArrowLeft'){event.preventDefault();$('previous').click();}
  });
}
init().catch(fail);
