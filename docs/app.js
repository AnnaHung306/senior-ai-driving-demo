/** Read-only static demo. There is deliberately no simulator/control API.
 * Each update waits for the selected RGB bytes to pass SHA and decode before
 * painting the matching 3D record. A generation counter rejects late seeks.
 */
import * as THREE from './vendor/three.module.js';
import {OrbitControls} from './vendor/OrbitControls.js';
import {CAMERA_LABELS,BEHAVIOR_LABELS,validateBundle,displayRecords,frameAtTime,checkedBytes,safeAssetPath} from './replay.js';

const $=id=>document.getElementById(id);
let bundle=null,index=-1,playing=false,requestGeneration=0,visibleBlob=null,renderer=null;
let playStart=0,simStart=0,busy=false,contextValid=true,scene,camera,controls,roadMesh,laneMesh,ego;
let dynamic=new THREE.Group();
let records=[],chapters=[],chapter=null,sceneGeneration=0,heldLabels=[],busyToken=0;
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
  const createCells=(color,size)=>{
    const mesh=new THREE.InstancedMesh(geometry,material(color),1);
    mesh.userData.cellSize=size;mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);scene.add(mesh);return mesh;
  };
  roadMesh=createCells(0x526b7e,.48);
  laneMesh=createCells(0xe2c881,.16);
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

function resizeCells() {
  const grow=(mesh,key)=>{
    const size=Math.max(1,...records.map(r=>r.perception[key].length));
    if(mesh.instanceMatrix.count>=size)return mesh;
    const replacement=new THREE.InstancedMesh(geometry,mesh.material,size);
    replacement.userData.cellSize=mesh.userData.cellSize;replacement.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    scene.remove(mesh);mesh.dispose();scene.add(replacement);return replacement;
  };
  roadMesh=grow(roadMesh,'road_cells_xy_m');laneMesh=grow(laneMesh,'lane_cells_xy_m');
}

function paintWorld(record) {
  for(const label of heldLabels)label.node.remove();heldLabels=[];
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
    if(object.forecast?.status==='available')line(object.forecast.positions_ego_xy_m,0x8babb0,true);
  }
  for(const object of record.perception.dynamics.held_unobserved_tracks) {
    const mesh=new THREE.Mesh(geometry,material(0xf0cc87,true));
    const [length,width,height]=object.size_lwh_m;mesh.scale.set(width,height,length);
    mesh.position.set(...coordinates(object.center_ego_m));dynamic.add(mesh);
    const node=document.createElement('span');node.className='track-label';
    node.textContent=`${object.track_id.split(':').at(-1)} · ${object.age_since_measurement_s.toFixed(1)}s`;
    $('viewport').append(node);
    const position=new THREE.Vector3(...coordinates(object.center_ego_m));position.y+=height/2+.4;
    heldLabels.push({node,position});
  }
  $('viewport').dataset.frame=String(record.frame);
}

function paintMap(record) {
  if(bundle.kind==='perception') {
    $('map').setAttribute('hidden','');$('progress-label').textContent='八鏡頭融合 · 局部候選線';
    $('map-source').textContent='感知片段，未設定導航目的地';return;
  }
  $('map').removeAttribute('hidden');$('map-source').textContent='地圖／自車定位先驗';
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
  $('progress-label').textContent=record.plan.state==='ARRIVED'?'抵達終點':`已行駛 ${Math.min(bundle.route.length_m,record.navigation.progress_m).toFixed(1)} m`;
}

async function display(nextIndex) {
  const generation=++requestGeneration;
  if(!Number.isInteger(nextIndex)||nextIndex<0||nextIndex>=records.length)throw Error('影格超出範圍');
  const selectedCamera=$('camera').value,record=records[nextIndex],asset=bundle.frame_assets[nextIndex].images[selectedCamera];
  let bytes;
  try { bytes=await checkedBytes(await fetch(new URL(asset.url,location.href)),asset.web_JPEG_sha256); }
  catch(error) { if(generation!==requestGeneration)return false;throw error; }
  const url=URL.createObjectURL(new Blob([bytes],{type:'image/jpeg'}));
  const decoded=new Image();decoded.src=url;
  try { await decoded.decode(); } catch(error) { URL.revokeObjectURL(url);if(generation!==requestGeneration)return false;throw error; }
  if(generation!==requestGeneration){URL.revokeObjectURL(url);return false;}
  // Commit 3D, RGB, controls and labels together after source verification.
  if(visibleBlob)URL.revokeObjectURL(visibleBlob);visibleBlob=url;
  $('rgb').src=url;$('rgb').hidden=false;$('rgb').alt=`${CAMERA_LABELS[selectedCamera]} RGB · F${record.frame}`;
  $('rgb').dataset.frame=String(record.frame);$('rgb').dataset.camera=selectedCamera;
  $('rgb-loading').hidden=true;
  paintWorld(record);paintMap(record);index=nextIndex;$('world').hidden=false;
  $('frame-label').textContent=`F${record.frame} · ${CAMERA_LABELS[selectedCamera]}`;
  $('speed').textContent=record.speed_mps===null?'—':(record.speed_mps*3.6).toFixed(1);
  $('remaining').textContent=record.navigation?record.navigation.remaining_m.toFixed(1):'—';
  $('behavior').textContent=record.display_only?'感知重播':BEHAVIOR_LABELS[record.plan.state]||record.plan.state;
  const names={1:'車',2:'行人',3:'自行車',4:'機車',5:'卡車',6:'公車'},counts={};
  for(const object of record.perception.objects)counts[object.class_id]=(counts[object.class_id]||0)+1;
  $('objects').textContent=Object.entries(counts).map(([id,n])=>`${names[id]||'物件'} ${n}`).join(' · ')||'無當前物件';
  const held=record.perception.dynamics.held_unobserved_tracks;
  $('memory').textContent=held.length?held.map(h=>`${h.track_id.split(':').at(-1)} ${names[h.class_id]||'物件'} · 暫失 ${h.age_since_measurement_s.toFixed(1)}s`).join(' ｜ '):'本幀沒有暫失物件';
  $('memory').dataset.held=String(held.length);
  $('current-tracks').textContent='當前觀測 ID：'+record.perception.objects.map(o=>`${o.track_id.split(':').at(-1)} ${names[o.class_id]||'物件'}`).join(' · ');
  const signalNames={red:'紅燈',green:'綠燈',yellow:'黃燈',unknown:'未確定'};
  $('signal').textContent=record.display_only?'此片段未展示號誌辨識':record.relevant_RGB_signals?.length?
    record.relevant_RGB_signals.map(s=>`${signalNames[s.phase]||s.phase} (${CAMERA_LABELS[s.camera]||s.camera})`).join(' · '):'本幀無相關號誌輸出';
  // Enlarge only a current, same-camera model ROI. No simulator lamp state.
  const signalROI=record.relevant_RGB_signals?.find(s=>s.camera===selectedCamera&&s.bbox_xyxy?.length===4);
  $('signal-crop').hidden=!signalROI;
  if(signalROI) {
    const [x1,y1,x2,y2]=signalROI.bbox_xyxy;
    const left=Math.max(0,x1-5),top=Math.max(0,y1-5),right=Math.min(640,x2+5),bottom=Math.min(360,y2+5);
    const canvas=$('signal-zoom'),ctx=canvas.getContext('2d');
    canvas.width=Math.max(1,Math.round(right-left));canvas.height=Math.max(1,Math.round(bottom-top));
    ctx.drawImage(decoded,left,top,right-left,bottom-top,0,0,canvas.width,canvas.height);
    canvas.dataset.frame=String(record.frame);canvas.dataset.camera=selectedCamera;
  }
  $('seek').value=String(index);$('count').textContent=`${index+1} / ${records.length} 幀`;
  $('time').textContent=`${record.simulation_s.toFixed(1)} / ${bundle.summary.simulation_s.toFixed(1)} s`;
  const descriptions={CRUISE:'沿已規劃路線行駛，持續檢查RGB道路與物件。',FOLLOW:'前方有物件，降低車速並保留間距。',OBSTACLE_STOP:'前方走廊有障礙，停車等待。',GOAL_BRAKE:'接近終點，降低車速並準備煞停。',ARRIVED:'此段紀錄已抵達；後續清理煞停實測為0.0 km/h。'};
  descriptions.PERCEPTION_ONLY='只有感知與因果追蹤；沒有發出駕駛控制。候選線不是目的地導航。';
  descriptions.SIGNAL_STOP='號誌停等；綠燈辨識與控制器解除停等可能有延遲，依原始紀錄呈現。';
  $('decision-description').textContent=descriptions[record.plan.state]||'依當時保存的感知與規劃結果顯示。';
  $('previous').disabled=index===0;$('next').disabled=index===records.length-1;
  notice(bundle.kind==='perception'?'同幀RGB與模型輸出 · 已知場景感知重播 · 不含駕駛控制':'真閉環錄製回放 · 同幀RGB與AI輸出 · 此具名路線已抵達');
  // Diagnostic DOM state is intentionally read-only and contains no GT actors.
  $('player').dataset.index=String(index);$('player').dataset.frame=String(record.frame);
  $('player').dataset.chapter=chapter.id;$('player').dataset.kind=bundle.kind||'drive';
  $('player').dataset.verified='true';return true;
}

async function seek(next) {
  pause();try { await display(next); } catch(error) { fail(error); }
}

function animate(now) {
  if(renderer&&contextValid){
    controls.update();renderer.render(scene,camera);
    for(const {node,position} of heldLabels) {
      const p=position.clone().project(camera);node.hidden=p.z< -1||p.z>1||Math.abs(p.x)>.94||Math.abs(p.y)>.94;
      node.style.left=`${(p.x+1)*50}%`;node.style.top=`${(1-p.y)*50}%`;
    }
  }
  if(playing&&!busy) {
    const elapsed=(now-playStart)/1000*Number($('rate').value);
    const next=frameAtTime(records,simStart+elapsed);
    if(next!==index) {
      busy=true;
      // Keep OrbitControls/rendering responsive while fetch/decode is pending.
      const epoch=sceneGeneration,token=++busyToken;
      display(next).catch(error=>{if(epoch===sceneGeneration)fail(error);}).finally(()=>{if(token===busyToken)busy=false;if(epoch===sceneGeneration&&index===records.length-1)pause();});
    }
    if(index===records.length-1)pause();
  }
  requestAnimationFrame(animate);
}

async function init() {
  const manifestResponse=await fetch('./manifest.json');
  if(!manifestResponse.ok)throw Error('manifest無法載入；請確認伺服器的根目錄為docs');
  const manifest=await manifestResponse.json();
  const catalogHash=manifest.files.find(f=>f.path==='data/showcases.json')?.sha256;
  const bytes=await checkedBytes(await fetch('./data/showcases.json'),catalogHash);
  chapters=JSON.parse(new TextDecoder().decode(bytes)).chapters;
  if(!chapters?.length||new Set(chapters.map(c=>c.id)).size!==chapters.length)throw Error('展示目錄錯誤');
  for(const c of chapters)if(!['drive','perception'].includes(c.kind)||safeAssetPath(c.file)!==`data/${c.file.split('/').at(-1)}`||manifest.files.find(f=>f.path===c.file)?.sha256!==c.sha256)throw Error('章節來源錯誤');
  $('chapter').replaceChildren(...chapters.map(c=>new Option(c.label,c.id)));
  $('camera').replaceChildren(...Object.entries(CAMERA_LABELS).map(([value,label])=>new Option(label,value)));
  initializeWorld();requestAnimationFrame(animate);
  $('chapter').onchange=()=>loadChapter($('chapter').value).catch(fail);
  await loadChapter(chapters[0].id);
  $('play').onclick=async()=>{
    if(playing){pause();return;}
    const epoch=sceneGeneration;
    if(index===records.length-1) {try {if(!await display(0))return;}catch(error){if(epoch===sceneGeneration)fail(error);return;}}
    if(epoch!==sceneGeneration)return;
    simStart=records[index].simulation_s;playStart=performance.now();playing=true;
    $('play').textContent='Ⅱ';$('play').setAttribute('aria-label','暫停回放');
  };
  $('previous').onclick=()=>seek(Math.max(0,index-1));$('next').onclick=()=>seek(Math.min(records.length-1,index+1));
  $('seek').oninput=()=>seek(Number($('seek').value));
  $('camera').onchange=()=>seek(Math.max(0,index));
  $('rate').onchange=()=>{if(playing){simStart=records[index].simulation_s;playStart=performance.now();}};
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  // Keyboard shortcuts are scoped to the player and never hijack form controls.
  $('player').addEventListener('keydown',event=>{
    if(/^(INPUT|SELECT|BUTTON|A)$/.test(event.target.tagName))return;
    if(event.code==='Space'){event.preventDefault();$('play').click();}
    if(event.code==='ArrowRight'){event.preventDefault();$('next').click();}
    if(event.code==='ArrowLeft'){event.preventDefault();$('previous').click();}
  });
}

async function loadChapter(id) {
  const epoch=++sceneGeneration;++requestGeneration;++busyToken;busy=false;pause();
  $('player').dataset.verified='false';$('world').hidden=true;$('rgb').hidden=true;
  $('rgb-loading').textContent='正在驗證章節與同幀影像…';$('rgb-loading').hidden=false;
  for(const node of heldLabels)node.node.remove();heldLabels=[];
  for(const id of ['play','seek','camera','previous','next'])$(id).disabled=true;
  $('bookmarks').replaceChildren();notice('載入章節中…');
  const selected=chapters.find(c=>c.id===id);
  let loaded;
  try {loaded=validateBundle(JSON.parse(new TextDecoder().decode(await checkedBytes(await fetch(selected.file),selected.sha256))));}
  catch(error){if(epoch!==sceneGeneration)return;throw error;}
  if(epoch!==sceneGeneration)return;
  if((loaded.kind||'drive')!==selected.kind)throw Error('章節種類不符');
  bundle=loaded;chapter=selected;records=displayRecords(bundle);index=-1;resizeCells();
  $('chapter').value=id;$('camera').value=chapter.camera;$('seek').max=String(records.length-1);
  $('chapter-description').textContent=chapter.description;
  $('bookmarks').replaceChildren(...chapter.bookmarks.map(b=>{
    if(!Number.isInteger(b.index)||b.index<0||b.index>=records.length)throw Error('章節書籤超出範圍');
    const button=document.createElement('button');button.type='button';button.className='small-button';button.textContent=b.label;button.disabled=true;
    button.onclick=()=>seek(b.index);return button;
  }));
  const applied=await display(0);
  if(epoch!==sceneGeneration||!applied)return;
  for(const id of ['play','seek','camera'])$(id).disabled=false;
  for(const button of $('bookmarks').children)button.disabled=false;
}
init().catch(fail);
