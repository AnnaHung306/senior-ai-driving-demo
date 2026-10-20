import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {CLASS_NAMES,COLORS,toScene,sceneDimensions,sceneYaw,gridPoint,playbackIndex} from './math.js';
import {PredictionClient} from './data-client.js';

const $=id=>document.getElementById(id);
const client=new PredictionClient();
const state={manifest:null,scene:null,index:0,current:null,mode:'follow',selected:null,playing:false,playStart:0,playIndex:0,speed:1,
  request:0,imageRequest:0,imageURL:null,watch:false,feedKey:null,feedTime:null,feedScene:null,feedAge:null,error:null};
const metrics={applies:[],loads:[],fps:[],applied:[],totalApplies:0,unique:new Set(),maxGeometries:0,streamApplied:0,renderedFrames:0};
const viewport=$('viewport'),message=$('viewportMessage');
let renderer,camera,controls,scene,frameGroup,voxelGroup,objectGroups=[],labelNodes=[],lastMetric=0,fpsTicks=0,fpsStart=performance.now(),contextValid=true;

const percentile=(values,p)=>values.length?[...values].sort((a,b)=>a-b)[Math.min(values.length-1,Math.floor((values.length-1)*p))]:null;
const format=(v,n=1)=>Number.isFinite(v)?v.toFixed(n):'—';
const times=()=>state.scene?.frames.map(key=>state.manifest.frames.find(f=>f.key===key).timestamp_s)??[];
function status(text,error=false){$('dataStatus').textContent=text;$('dataStatus').classList.toggle('invalid',error);}
function showMessage(text,error=false){message.textContent=text;message.hidden=false;message.classList.toggle('error',error);}
function pause(){state.playing=false;$('playPause').textContent='▶';$('playPause').setAttribute('aria-label','播放');}
function fail(error){pause();state.error=String(error.message??error);status('資料未套用 · '+state.error,true);showMessage(state.error,true);updateDiagnostics();}

function addBox(group,size,position,material){const mesh=new THREE.Mesh(new THREE.BoxGeometry(...size),material);mesh.position.set(...position);group.add(mesh);return mesh;}
function material(color,opacity=1){return new THREE.MeshStandardMaterial({color,roughness:.68,metalness:.08,transparent:opacity<1,opacity});}
function asset(category,color){
  const group=new THREE.Group(),body=material(color,.85),dark=material(0x172739),glass=material(0x98b9ce,.78),silver=material(0xa5b7c4);
  if(category===2){
    addBox(group,[.34,.53,.24],[0,.05,0],body);
    const head=new THREE.Mesh(new THREE.SphereGeometry(.16,12,8),body);head.position.set(0,.48,0);group.add(head);
    for(const x of [-.12,.12])addBox(group,[.12,.58,.13],[x,-.49,0],dark);
    for(const x of [-.25,.25])addBox(group,[.1,.54,.12],[x,.0,0],body);
    addBox(group,[.1,.08,.05],[0,.49,-.15],dark);
  }else if(category===3||category===4){
    for(const z of [-.4,.4]){const wheel=new THREE.Mesh(new THREE.CylinderGeometry(.24,.24,.095,16),dark);wheel.rotation.z=Math.PI/2;wheel.position.set(0,-.25,z);group.add(wheel);}
    addBox(group,[category===3?.13:.29,.23,.62],[0,-.05,0],body);addBox(group,[.28,.07,.27],[0,.15,.1],dark);
    addBox(group,[.45,.035,.04],[0,.22,-.34],silver);addBox(group,[.055,.36,.055],[0,.02,-.35],silver);
    addBox(group,[.26,.35,.2],[0,.42,0],material(0x7296a6));
    const head=new THREE.Mesh(new THREE.SphereGeometry(.13,10,8),silver);head.position.set(0,.74,-.05);group.add(head);
  }else{
    const large=category>=5;
    addBox(group,[1,.34,1.8],[0,-.15,0],body);
    if(large){
      addBox(group,[1,.98,category===6?1.65:1.12],[0,.38,category===6?0:.3],body);
      if(category===5)addBox(group,[.98,.62,.6],[0,.17,-.6],body);
      addBox(group,[.83,.29,.03],[0,.39,-.91],glass);
      if(category===6)for(const x of [-.505,.505])addBox(group,[.02,.36,1.45],[x,.53,0],glass);
    }else{
      addBox(group,[.83,.37,1.0],[0,.2,.1],body);addBox(group,[.77,.24,.92],[0,.23,.08],glass);
      addBox(group,[.84,.035,1.02],[0,.41,.1],body);
    }
    for(const x of [-.45,.45])for(const z of [-.55,.58]){
      const wheel=new THREE.Mesh(new THREE.CylinderGeometry(.18,.18,.13,12),dark);wheel.rotation.z=Math.PI/2;wheel.position.set(x,-.31,z);group.add(wheel);
    }
    for(const x of [-.32,.32])addBox(group,[.18,.05,.025],[x,-.07,-.91],material(0xf7e6b8));
  }
  const bounds=new THREE.Box3().setFromObject(group),size=bounds.getSize(new THREE.Vector3()),center=bounds.getCenter(new THREE.Vector3());
  group.scale.set(1/size.x,1/size.y,1/size.z);group.position.set(-center.x/size.x,-center.y/size.y,-center.z/size.z);
  const root=new THREE.Group();root.add(group);return root;
}

function disposeTree(group){
  if(!group)return;group.removeFromParent();const geometries=new Set(),materials=new Set();
  group.traverse(object=>{if(object.geometry)geometries.add(object.geometry);if(object.material)for(const m of Array.isArray(object.material)?object.material:[object.material])materials.add(m);});
  for(const geometry of geometries)geometry.dispose();for(const m of materials)m.dispose();
}

function setView(mode){
  state.mode=mode;controls?.dispose();const aspect=viewport.clientWidth/viewport.clientHeight;
  if(mode==='top'){
    camera=new THREE.OrthographicCamera(-29*aspect,29*aspect,29,-29,.1,180);camera.position.set(0,60,-7);camera.up.set(0,0,-1);
  }else{
    camera=new THREE.PerspectiveCamera(46,aspect,.1,180);camera.position.set(...(mode==='follow'?[0,16,23]:[26,25,25]));
  }
  controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,0,-7);controls.enableDamping=true;controls.dampingFactor=.11;
  controls.enableRotate=mode==='free';controls.enablePan=mode==='free';controls.minDistance=7;controls.maxDistance=90;controls.maxPolarAngle=Math.PI*.48;
  controls.minZoom=.5;controls.maxZoom=4;controls.update();
  for(const [id,key] of [['followView','follow'],['topView','top'],['freeView','free']]){$(id).classList.toggle('active',mode===key);$(id).setAttribute('aria-pressed',String(mode===key));}
  $('viewTitle').textContent={follow:'自車中心 · 跟車視角',top:'自車中心 · 俯視視角',free:'自由觀察 · 可旋轉與平移'}[mode];
  $('viewHint').textContent=mode==='free'?'拖曳旋轉 · 滾輪縮放 · 右鍵平移':'滾輪縮放 · 點選物件查看預測';
  document.querySelector('.compass span').textContent=mode==='free'?'自車前方 −Z':'前方 ↑';
  viewport.dataset.view=mode;updateDiagnostics();
}

function resize(){
  if(!renderer||!camera)return;const w=viewport.clientWidth,h=viewport.clientHeight;
  renderer.setSize(w,h,false);
  if(camera.isOrthographicCamera){camera.left=-29*w/h;camera.right=29*w/h;}else camera.aspect=w/h;
  camera.updateProjectionMatrix();
}

function initialize3D(){
  try{renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});}catch(error){throw Error('此瀏覽器無法建立 WebGL2 三維畫面。請使用支援的瀏覽器。');}
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.7));renderer.setClearColor(0x101923,0);renderer.outputColorSpace=THREE.SRGBColorSpace;
  viewport.prepend(renderer.domElement);scene=new THREE.Scene();scene.fog=new THREE.Fog(0x101923,65,105);
  scene.add(new THREE.HemisphereLight(0xe6f4ff,0x496170,2.1));const light=new THREE.DirectionalLight(0xfff2df,2.5);light.position.set(-10,28,5);scene.add(light);
  const grid=new THREE.GridHelper(80,40,0x284052,0x203243);grid.position.set(0,-.07,-5);grid.material.transparent=true;grid.material.opacity=.32;scene.add(grid);
  const ego=asset(1,0xdde7ed);ego.scale.set(1.9,1.45,4.5);ego.position.y=.725;scene.add(ego);
  const halo=new THREE.Mesh(new THREE.RingGeometry(2.7,2.72,64),new THREE.MeshBasicMaterial({color:0x6c9fba,transparent:true,opacity:.3,side:THREE.DoubleSide}));halo.rotation.x=-Math.PI/2;halo.position.y=.04;scene.add(halo);
  setView('follow');resize();new ResizeObserver(resize).observe(viewport);
  renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();contextValid=false;fail(Error('WebGL 繪圖已中斷，播放已暫停。請重新載入。'));});
  const raycaster=new THREE.Raycaster();let down=null;
  renderer.domElement.addEventListener('pointerdown',event=>{down=[event.clientX,event.clientY];});
  renderer.domElement.addEventListener('pointerup',event=>{
    if(!down||Math.hypot(event.clientX-down[0],event.clientY-down[1])>5)return;
    const rect=renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1),camera);
    const hit=raycaster.intersectObjects(objectGroups,true).find(x=>{let p=x.object;while(p){if(p.userData.trackId)return true;p=p.parent;}return false;});
    if(hit){let p=hit.object;while(p&&!p.userData.trackId)p=p.parent;selectObject(p.userData.trackId);}
  });
  renderer.setAnimationLoop(tick);
}

function groundMesh(frame){
  const ground=frame.ground,[rows,cols]=ground.shape;const seen=ground.labels_row_major.filter(v=>v!==255).length;
  const geometry=new THREE.BoxGeometry(ground.grid.resolution_m,.025,ground.grid.resolution_m);
  const mesh=new THREE.InstancedMesh(geometry,material(0xffffff),seen);const matrix=new THREE.Matrix4(),color=new THREE.Color();let n=0;
  for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){
    const label=ground.labels_row_major[row*cols+col];if(label===255)continue;
    const [X,Y,Z]=toScene(gridPoint(ground.grid,row,col));matrix.makeTranslation(X,Y-.02,Z);mesh.setMatrixAt(n,matrix);
    color.setHex({1:0x3b5266,2:0x756d60,9:0x36524c}[label]);mesh.setColorAt(n++,color);
  }
  mesh.instanceMatrix.needsUpdate=true;if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;mesh.computeBoundingSphere();return mesh;
}

function paintMesh(frame){
  const points=frame.lane_paint.centers_ego_xy_m,s=frame.lane_paint.cell_size_m;
  const mesh=new THREE.InstancedMesh(new THREE.BoxGeometry(s,.018,s),new THREE.MeshBasicMaterial({color:0xdcd3a3}),points.length),matrix=new THREE.Matrix4();
  points.forEach((point,i)=>{const [x,y,z]=toScene(point);matrix.makeTranslation(x,y+.025,z);mesh.setMatrixAt(i,matrix);});mesh.instanceMatrix.needsUpdate=true;mesh.computeBoundingSphere();return mesh;
}

function createLines(segments,color,height=.035,opacity=1){
  const vertices=[];for(const segment of segments)for(const point of segment){const p=toScene(point);vertices.push(p[0],p[1]+height,p[2]);}
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));
  return new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color,transparent:opacity<1,opacity}));
}

function renderFrame(frame){
  disposeTree(frameGroup);disposeTree(voxelGroup);voxelGroup=null;frameGroup=new THREE.Group();objectGroups=[];
  for(const item of labelNodes)item.element.remove();labelNodes=[];
  frameGroup.add(groundMesh(frame));frameGroup.add(paintMesh(frame));
  frameGroup.add(createLines(frame.road_sidewalk_interfaces.segments_ego_xy_m,0xa4a390,.038,.6));
  const path=frame.local_path.points_ego_m;
  if(frame.local_path.status==='candidate'&&path.length>1){
    frameGroup.add(createLines(path.slice(1).map((p,i)=>[path[i],p]),0x8be0c3,.08));
    const mid=Math.max(1,Math.floor(path.length*.65)),a=path[mid-1],b=path[mid];
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute([-.24,0,.34,.24,0,.34,0,0,-.34],3));geometry.computeVertexNormals();
    const arrow=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:0x8be0c3,side:THREE.DoubleSide}));arrow.position.set(...toScene([...b,.09]));arrow.rotation.y=-Math.atan2(b[1]-a[1],b[0]-a[0]);frameGroup.add(arrow);
  }
  for(const object of frame.objects){
    const root=new THREE.Group();root.position.set(...toScene(object.center_ego_m));root.rotation.y=sceneYaw(object.yaw_rad);root.userData.trackId=object.track_id;root.userData.source=object;
    const mesh=asset(object.class_id,COLORS[object.class_id]);mesh.scale.set(...sceneDimensions(object.size_lwh_m));root.add(mesh);
    const outline=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(...sceneDimensions(object.size_lwh_m))),new THREE.LineBasicMaterial({color:0xd7bc87,transparent:true,opacity:.57}));outline.name='predictionBounds';root.add(outline);
    objectGroups.push(root);frameGroup.add(root);
    const label=document.createElement('span');label.className='world-label low';label.textContent=object.track_id.split(':').at(-1)+' · '+CLASS_NAMES[object.class_id];$('canvasLabels').append(label);
    labelNodes.push({element:label,object});
  }
  scene.add(frameGroup);if(!frame.objects.some(o=>o.track_id===state.selected))state.selected=null;
  drawVoxelLayers();applyLayerVisibility();renderObjectList();
  const samples=frame.local_path.points_ego_m;$('pathSummary').textContent=frame.local_path.status==='candidate'?`循線候選 · ${format(samples.at(-1)?.[0])} m${frame.local_path.source?.includes('prior')?' · 使用寬度先驗':''}`:'循線候選不可用 · 不產生替代直線';
}

function drawVoxelLayers(){
  disposeTree(voxelGroup);voxelGroup=null;if(!state.current||(!$('showVoxels').checked&&!$('showFree').checked))return;
  const {frame,occupancy}=state.current,[rows,cols,layers]=frame.occupancy.shape,points=[];
  for(let row=0;row<rows;row++)for(let col=0;col<cols;col++)for(let layer=0;layer<layers;layer++){
    const label=occupancy[(row*cols+col)*layers+layer];if(($('showVoxels').checked&&label>=3&&label<=8)||($('showFree').checked&&label===0))points.push([row,col,layer,label]);
  }
  const size=frame.occupancy_grid.resolution_m;
  const mesh=new THREE.InstancedMesh(new THREE.BoxGeometry(size*.88,size*.88,size*.88),new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:.23,depthWrite:false}),points.length);
  const matrix=new THREE.Matrix4(),color=new THREE.Color();points.forEach(([r,c,l,label],i)=>{matrix.makeTranslation(...toScene(gridPoint(frame.occupancy_grid,r,c,l)));mesh.setMatrixAt(i,matrix);color.setHex(label===0?0x497c8b:COLORS[label-2]);mesh.setColorAt(i,color);});
  mesh.instanceMatrix.needsUpdate=true;if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;mesh.computeBoundingSphere();voxelGroup=mesh;scene.add(mesh);
}

function applyLayerVisibility(){
  for(const root of objectGroups)root.getObjectByName('predictionBounds').visible=$('showBoxes').checked;
  for(const label of labelNodes)label.element.hidden=!$('showIDs').checked;
  $('rgbCard').hidden=!$('showRGB').checked;
}

function selectObject(id){state.selected=id;renderObjectList();updateDiagnostics();}
function renderObjectList(){
  const objects=state.current?.frame.objects??[];const list=$('objectList');list.replaceChildren();
  for(const object of [...objects].sort((a,b)=>Math.hypot(...a.center_ego_m.slice(0,2))-Math.hypot(...b.center_ego_m.slice(0,2)))){
    const button=document.createElement('button');button.className='object-row'+(state.selected===object.track_id?' selected':'');button.dataset.trackId=object.track_id;
    const icon=document.createElement('span');icon.className='class-dot';icon.style.borderColor='#'+COLORS[object.class_id].toString(16);icon.textContent={1:'車',2:'人',3:'單',4:'機',5:'貨',6:'巴'}[object.class_id];
    const label=document.createElement('span');label.className='object-identity';const title=document.createElement('strong');title.textContent=CLASS_NAMES[object.class_id];const small=document.createElement('small');small.textContent=object.track_id.split(':').at(-1)+' · '+object.camera_support.length+' 鏡頭';label.append(title,small);
    const distance=document.createElement('span');distance.className='distance';distance.textContent=format(Math.hypot(...object.center_ego_m.slice(0,2)))+' m';button.append(icon,label,distance);button.addEventListener('click',()=>selectObject(object.track_id));list.append(button);
  }
  if(!objects.length){const p=document.createElement('p');p.className='empty-message';p.textContent='當幀沒有可用三維量測';list.append(p);}
  const chosen=objects.find(o=>o.track_id===state.selected),details=$('objectDetails');details.hidden=!chosen;details.replaceChildren();
  $('selectedHint').textContent=chosen?'預測值 · 未校準':'點選查看';
  if(chosen){
    const dl=document.createElement('dl');for(const [key,value] of [['ID',chosen.track_id.split(':').at(-1)],['位置 x/y/z',chosen.center_ego_m.map(v=>format(v,2)).join(' / ')+' m'],['長／寬／高',chosen.size_lwh_m.map(v=>format(v,2)).join(' / ')+' m'],['朝向',format(chosen.yaw_rad*180/Math.PI)+'°'],['模型分數',format(chosen.score_uncalibrated,3)],['觀測次數',String(chosen.track_age_measurements)]]){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=value;dl.append(dt,dd);}details.append(dl);
    const note=document.createElement('span');note.className='uncertainty';note.textContent='朝向向量'+(chosen.orientation_reliable?'有效':'不足')+'，有效性不代表方向準確。';details.append(note);
  }
  viewport.dataset.selectedId=state.selected??'';
}

async function updateRGB(){
  if(!state.current)return;const token=++state.imageRequest,frame=state.current.frame,name=$('cameraSelect').value;
  try{const url=await client.camera(frame,name);if(token!==state.imageRequest){URL.revokeObjectURL(url);return;}
    if(state.imageURL)URL.revokeObjectURL(state.imageURL);state.imageURL=url;const image=$('rgbImage');image.alt=`${frame.scene_id} 影格 ${frame.frame} ${name} 原始 RGB`;
    image.onload=()=>{if(token===state.imageRequest){image.dataset.frame=String(frame.frame);image.dataset.camera=name;}};image.src=url;
    $('rgbCaption').textContent=`F${frame.frame} · 640 × 360 · 原始 RGB 已核對`;
  }catch(error){if(token===state.imageRequest){$('rgbCaption').textContent='影像對照載入失敗：'+error.message;$('rgbImage').removeAttribute('src');}}
}

async function goTo(index,{stream=false}={}){
  if(!state.scene||!contextValid)return;const key=state.scene.frames[index];if(!key)return;
  const token=++state.request,start=performance.now();$('frameSlider').value=String(index);status('載入並核對影格…');
  try{
    const result=await client.load(key);if(token!==state.request)return;const loadEnd=performance.now();
    state.index=index;state.current=result;state.error=null;renderFrame(result.frame);
    controls.update();renderer.render(scene,camera);const applyEnd=performance.now();metrics.applies.push(applyEnd-loadEnd);metrics.loads.push(loadEnd-start);metrics.applied.push(key);
    metrics.totalApplies++;metrics.unique.add(key);for(const history of [metrics.applies,metrics.loads,metrics.applied])if(history.length>512)history.shift();
    if(stream)metrics.streamApplied++;
    message.hidden=true;viewport.dataset.frameKey=key;viewport.dataset.frame=String(result.frame.frame);viewport.dataset.objectCount=String(result.frame.objects.length);
    $('objectCount').textContent=String(result.frame.objects.length);$('unknownCount').textContent=Math.round(result.frame.occupancy_unknown_fraction*100)+'%';
    $('dataFrequency').textContent=format(state.scene.source_frequency_Hz,0);$('frameLabel').textContent=`${index+1} / ${state.scene.frames.length} · F${result.frame.frame}`;
    const t=times();$('timeLabel').textContent=`${format(t[index]-t[0])} / ${format(t.at(-1)-t[0])} s`;
    status(stream?'已接收新輸出 · 已計算推論來源':'已驗證影格 · 按原時間戳回放');updateDiagnostics();updateRGB();
    for(const next of state.scene.frames.slice(index+1,index+3))client.load(next).catch(()=>{});
    return true;
  }catch(error){if(token===state.request)fail(error);}
}

async function changeScene(id,index=0){
  pause();state.scene=state.manifest.scenes.find(s=>s.id===id);if(!state.scene)throw Error('未知場景');
  $('sceneSelect').value=id;$('frameSlider').max=String(state.scene.frames.length-1);state.selected=null;await goTo(index);
}

function updateLabels(){
  if(!camera||!state.current)return;
  for(const label of labelNodes){
    if(!$('showIDs').checked){label.element.hidden=true;continue;}
    const xyz=label.object.center_ego_m,point=new THREE.Vector3(...toScene([xyz[0],xyz[1],xyz[2]+label.object.size_lwh_m[2]/2+.25]));
    const local=point.clone().applyMatrix4(camera.matrixWorldInverse);point.project(camera);
    const visible=local.z<0&&point.z<1&&Math.abs(point.x)<.98&&Math.abs(point.y)<.96;label.element.hidden=!visible;
    if(visible){label.element.style.left=(point.x*.5+.5)*viewport.clientWidth+'px';label.element.style.top=(-point.y*.5+.5)*viewport.clientHeight+'px';}
  }
}

function updateDiagnostics(){
  if(!renderer)return;
  const fps=metrics.fps.at(-1)??null,p95=percentile(metrics.applies,.95),frame=state.current?.frame;
  $('fpsShort').textContent=format(fps,0)+' render FPS';$('renderFPS').textContent=format(fps,1)+' FPS';$('applyLatency').textContent=format(p95,1)+' ms';
  $('perceptionLatency').textContent=frame?format(Object.values(frame.timing_ms).reduce((a,b)=>a+b,0),0)+' ms':'—';
  $('arrivalLatency').textContent=state.watch?format(state.feedAge,0)+' ms':'錄製回放';
  const info=renderer.info;metrics.maxGeometries=Math.max(metrics.maxGeometries,info.memory.geometries);
  const report={mode:state.watch?'monitored_recorded_inference_feed':'timestamped_recorded_inference_replay',view:state.mode,frame_key:viewport.dataset.frameKey??null,
    frame_source_sha256:frame?.world_source_sha256,run_contract_sha256:state.manifest?.run_contract_sha256,checkpoint_sha256:state.manifest?.checkpoint_sha256,
    object_count:frame?.objects.length??0,selected_id:state.selected,applied_frames:metrics.applied,total_applied_frames:metrics.totalApplies,history_limit:512,unique_applied_frames:metrics.unique.size,
    frame_apply_ms:{measurement:'CPU scene rebuild plus render submission; excludes GPU present',samples:metrics.applies.length,p50:percentile(metrics.applies,.5),p95},verified_fetch_ms:{p50:percentile(metrics.loads,.5),p95:percentile(metrics.loads,.95)},
    render_fps:{samples:metrics.fps.length,median:percentile(metrics.fps,.5),latest:fps},renderer_memory:{...info.memory,max_geometries:metrics.maxGeometries},render_calls:info.render.calls,
    stream_applied:metrics.streamApplied,feed_age_ms:state.feedAge,context_valid:contextValid,error:state.error,
    layers:{boxes:$('showBoxes').checked,IDs:$('showIDs').checked,voxels:$('showVoxels').checked,free:$('showFree').checked},
    camera:camera?{type:camera.type,position:camera.position.toArray(),target:controls.target.toArray(),zoom:camera.zoom}:null,
    rendered_object_transforms:objectGroups.map(g=>({track_id:g.userData.trackId,position_three:g.position.toArray(),yaw_y_three:g.rotation.y,dimensions_three:sceneDimensions(g.userData.source.size_lwh_m)})),
    frame_timing_ms:frame?.timing_ms,formal_model_quality_passed:false};
  $('diagnosticReport').value=JSON.stringify(report,null,2);
}

function tick(time){
  if(!contextValid)return;metrics.renderedFrames++;fpsTicks++;
  if(state.playing&&state.scene){
    const {index,ended}=playbackIndex(times(),state.playIndex,(time-state.playStart)*state.speed/1000);
    if(index!==state.index&&Number($('frameSlider').value)!==index)goTo(index);
    if(ended)pause();
  }
  controls.update();renderer.render(scene,camera);updateLabels();
  if(time-fpsStart>=1000){metrics.fps.push(fpsTicks*1000/(time-fpsStart));fpsTicks=0;fpsStart=time;if(metrics.fps.length>300)metrics.fps.shift();}
  if(time-lastMetric>1000){updateDiagnostics();lastMetric=time;}
}

async function pollFeed(){
  if(!state.watch)return;
  try{
    const response=await fetch('/api/feed',{cache:'no-store'}),result=await response.json();
    if(!response.ok||result.status==='invalid')throw Error(result.message??'輸出來源不合法');
    state.feedAge=result.age_ms??null;
    if(result.status==='waiting'||result.status==='stale'){status(result.status==='stale'?'輸出已逾時 · 保留最後已驗證畫面':'等待新推論輸出 · 回放來源');updateDiagnostics();return;}
    if(result.source_mode!=='recorded_inference_replay')throw Error('未知串流模式');
    const frame=result.frame;
    if(state.feedScene===frame.scene_id&&state.feedTime!==null&&frame.timestamp_s<state.feedTime)throw Error('新輸出時間倒退');
    if(frame.key!==state.feedKey||state.error){
      const sceneSpec=state.manifest.scenes.find(s=>s.id===frame.scene_id),index=sceneSpec?.frames.indexOf(frame.key);
      if(!sceneSpec||index<0)throw Error('串流影格不在已驗證資料清單');
      pause();state.scene=sceneSpec;$('sceneSelect').value=sceneSpec.id;$('frameSlider').max=String(sceneSpec.frames.length-1);
      if(await goTo(index,{stream:true})){state.feedKey=frame.key;state.feedScene=frame.scene_id;state.feedTime=frame.timestamp_s;}
    }
  }catch(error){fail(error);}
}

function connectUI(){
  for(const [id,mode] of [['followView','follow'],['topView','top'],['freeView','free']])$(id).addEventListener('click',()=>setView(mode));
  $('resetView').addEventListener('click',()=>setView(state.mode));
  $('sceneSelect').addEventListener('change',()=>{state.watch=false;$('watchFeed').checked=false;changeScene($('sceneSelect').value).catch(fail);});
  $('prevFrame').addEventListener('click',()=>{pause();goTo(Math.max(0,state.index-1));});$('nextFrame').addEventListener('click',()=>{pause();goTo(Math.min(state.scene.frames.length-1,state.index+1));});
  $('frameSlider').addEventListener('input',()=>{pause();goTo(Number($('frameSlider').value));});
  $('playPause').addEventListener('click',()=>{
    if(state.playing){pause();return;}if(!state.current||state.error||!contextValid)return;
    state.watch=false;$('watchFeed').checked=false;const start=state.index===state.scene.frames.length-1?0:state.index;
    if(start===0&&state.index!==0)goTo(0);state.playIndex=start;state.playStart=performance.now();state.playing=true;$('playPause').textContent='Ⅱ';$('playPause').setAttribute('aria-label','暫停');
  });
  $('speedSelect').addEventListener('change',()=>{pause();state.speed=Number($('speedSelect').value);});
  for(const id of ['showBoxes','showIDs','showRGB'])$(id).addEventListener('change',()=>{applyLayerVisibility();updateDiagnostics();});
  for(const id of ['showVoxels','showFree'])$(id).addEventListener('change',()=>{drawVoxelLayers();updateDiagnostics();});
  $('cameraSelect').addEventListener('change',updateRGB);
  $('watchFeed').addEventListener('change',()=>{pause();state.watch=$('watchFeed').checked;state.feedKey=null;state.feedScene=null;state.feedTime=null;state.error=null;if(state.watch)pollFeed();else status('已離開監看 · 錄製回放');});
  $('retryData').addEventListener('click',()=>location.reload());setInterval(pollFeed,500);
  $('testGraphicsLoss').addEventListener('click',()=>{
    const extension=renderer.getContext().getExtension('WEBGL_lose_context');
    if(extension)extension.loseContext();else $('graphicsTestStatus').textContent='此瀏覽器未提供 WebGL 中斷診斷；未執行。';
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  window.addEventListener('beforeunload',()=>{renderer?.dispose();controls?.dispose();if(state.imageURL)URL.revokeObjectURL(state.imageURL);});
}

async function start(){
  try{
    initialize3D();connectUI();state.manifest=await client.initialize();
    const select=$('sceneSelect');for(const item of state.manifest.scenes){const option=document.createElement('option');option.value=item.id;option.textContent=item.id+' · '+item.frames.length+' 組';select.append(option);}select.disabled=false;
    const q=state.manifest.quality_context;$('qualitySummary').textContent=`既有驗證短片：定位 P ${format(q.position_precision_at_2m*100)}%／R ${format(q.position_recall_at_2m*100)}%，完整3D方框 R@IoU .25 ${format(q.box_recall_IoU025*100)}%。ID切換 ${q.ID_switches}／${q.matched_transitions} 個成功配對轉移。品質尚未通過。`;
    await changeScene(state.manifest.scenes[0].id);
  }catch(error){fail(error);}
}
start();
