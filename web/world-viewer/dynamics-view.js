import * as THREE from './vendor/three.module.js';
import {toScene,sceneDimensions} from './math.js';
import {RISK_COLORS} from './dynamics.js';

export function createDynamicsLayer(frame,selected,options){
  const root=new THREE.Group();root.name='motionOverlays';root.userData.renderedForecasts=[];
  function line(points,color,opacity=1,dashed=false,height=.10){
    if(points.length<2)return;
    const geometry=new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(...toScene([p[0],p[1],height]))));
    const material=dashed?new THREE.LineDashedMaterial({color,transparent:true,opacity,dashSize:.34,gapSize:.20,depthWrite:false}):new THREE.LineBasicMaterial({color,transparent:true,opacity,depthWrite:false});
    const mesh=new THREE.Line(geometry,material);if(dashed)mesh.computeLineDistances();root.add(mesh);return mesh;
  }
  for(const o of frame.objects){
    const f=o.forecast,m=o.motion;if(!f)continue;
    const chosen=o.track_id===selected,active=!selected||chosen;
    const opacity=chosen ? 1 : selected ? .18 : .48;
    const color=m.status==='uncertain'?0x8c9aa9:RISK_COLORS[o.risk.status];
    if(options.history&&o.history_ego_xy_m.length>1)line(o.history_ego_xy_m,0x7896b2,active?.65:.15,false,.075);
    if(options.prediction&&f.status==='available'){
      const future=line(f.positions_ego_xy_m,color,opacity,true,.11);
      const raw=Array.from(future.geometry.attributes.position.array),positions=[];
      for(let i=0;i<raw.length;i+=3)positions.push(raw.slice(i,i+3));
      root.userData.renderedForecasts.push({track_id:o.track_id,positions_three:positions,opacity});
      for(const index of [5,10,15]){
        const point=f.positions_ego_xy_m[index];
        const dot=new THREE.Mesh(new THREE.RingGeometry(chosen?.18:.10,chosen?.24:.15,16),new THREE.MeshBasicMaterial({color,transparent:true,opacity,side:THREE.DoubleSide,depthWrite:false}));
        dot.rotation.x=-Math.PI/2;dot.position.set(...toScene([...point,.12]));root.add(dot);
        if(options.uncertainty&&active){
          const r=f.uncertainty_radius_m[index];if(r>.02){
            const ring=new THREE.Mesh(new THREE.RingGeometry(r,r+.035,48),new THREE.MeshBasicMaterial({color,transparent:true,opacity:chosen?.25:.10,side:THREE.DoubleSide,depthWrite:false}));
            ring.rotation.x=-Math.PI/2;ring.position.set(...toScene([...point,.065]));root.add(ring);
          }
        }
      }
    }
    if(options.velocity&&m.velocity_available&&m.status==='moving'&&m.speed_mps>.15){
      const [vx,vy]=m.velocity_ground_ego_mps;const direction=new THREE.Vector3(vy,0,-vx).normalize();
      const arrow=new THREE.ArrowHelper(direction,new THREE.Vector3(...toScene([o.center_ego_m[0],o.center_ego_m[1],o.center_ego_m[2]+o.size_lwh_m[2]/2+.14])),Math.min(4,Math.max(.6,m.speed_mps*.6)),color,.32,.17);
      arrow.traverse(p=>{if(p.material){p.material.transparent=true;p.material.opacity=active?.85:.25;}});root.add(arrow);
    }
  }
  if(options.coasting)for(const h of frame.dynamics?.held_unobserved_tracks??[]){
    const geometry=new THREE.EdgesGeometry(new THREE.BoxGeometry(...sceneDimensions(h.size_lwh_m)));
    const mesh=new THREE.LineSegments(geometry,new THREE.LineDashedMaterial({color:0x8192a8,transparent:true,opacity:.23,dashSize:.20,gapSize:.15,depthWrite:false}));
    mesh.computeLineDistances();mesh.position.set(...toScene(h.center_ego_m));root.add(mesh);
  }
  return root;
}
