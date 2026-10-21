"""Evaluation-only observer. Never called by RGB inference or the numeric planner.

Its return values contain no control information; raw GT lives in audit/ only.
"""
import math
import queue
from pathlib import Path
import numpy as np
from PIL import Image
from .core import receive_frame
from .navigation_atomic import write_json
from .frames import RgbFrame
from .navigation_io import read, sha

ROOT=Path(__file__).resolve().parents[2]


class EvaluationObserver:
    def __init__(self):
        self.config=read(ROOT/'configs/phase10_audit_v1.json')
        self.contract={'source_sha256':sha(Path(__file__)),'config_sha256':sha(ROOT/'configs/phase10_audit_v1.json'),
            'GT_only_in_audit_folder':True,'GT_returned_to_planner':False,'sample_stride':self.config['GT_stride'],
            'GT_uses_capture_snapshot_not_readback_pose':True,'sensor':'eight_raw_instance_segmentation'}

    def setup(self,world,ego,rig,calibration,output):
        import carla
        self.out=output/'audit';self.out.mkdir();self.queues={};sensors=[];self.owned_sensors=sensors;self.bounds={}
        write_json(self.out/'contract.json',self.contract)
        source=self.out/'phase10_observer.py';source.write_bytes(Path(__file__).read_bytes())
        self.calibration=calibration
        for d in rig['cameras']:
            bp=world.get_blueprint_library().find('sensor.camera.instance_segmentation')
            for k,v in {'image_size_x':640,'image_size_y':360,'fov':90,'sensor_tick':0}.items():bp.set_attribute(k,str(v))
            tr=carla.Transform(carla.Location(*d['position_m']),carla.Rotation(yaw=d['yaw_deg']))
            sensor=world.spawn_actor(bp,tr,attach_to=ego);sensors.append(sensor)
            sink=queue.Queue(maxsize=16);self.queues[d['name']]=sink
            def callback(im,q=sink):
                f=RgbFrame.from_carla(im)
                try:q.put_nowait(f)
                except queue.Full:
                    try:q.get_nowait()
                    except queue.Empty:pass
                    try:q.put_nowait(f)
                    except queue.Full:pass
            sensor.listen(callback)
        return sensors

    def capture(self,world,snapshot,ego_id,step,prediction,diagnostics,inventory):
        ego=snapshot.find(ego_id);reference=np.array(ego.get_transform().get_inverse_matrix())
        actors=[];invalid=[]
        for actor in world.get_actors():
            if actor.id==ego_id or not actor.type_id.startswith(('vehicle.','walker.pedestrian.')):continue
            current=snapshot.find(actor.id)
            if current is None:continue
            if actor.id>65535:raise ValueError('Instance actor ID exceeds uint16; ambiguous GT')
            tr=current.get_transform();velocity=current.get_velocity()
            box=self.bounds.get(actor.id)
            if box is None:
                box=actor.bounding_box
                ext=np.array([box.extent.x,box.extent.y,box.extent.z])
                if not np.isfinite(ext).all() or (ext<=1e-6).any():
                    invalid.append({'actor_id':actor.id,'reason':'zero_or_invalid_bbox_during_actor_lifecycle','extent_m':ext.tolist() if np.isfinite(ext).all() else None});continue
                self.bounds[actor.id]=box
            vertices=np.array([[p.x,p.y,p.z] for p in box.get_world_vertices(tr)])
            points=vertices@reference[:3,:3].T+reference[:3,3]
            size=np.array([box.extent.x,box.extent.y,box.extent.z])*2
            axis=points[4]-points[0]
            axes=np.stack([points[4]-points[0],points[2]-points[0],points[1]-points[0]],axis=1)/size
            v=np.array([velocity.x,velocity.y,velocity.z])
            actors.append({'actor_id':actor.id,'blueprint':actor.type_id,'center_ego_m':points.mean(0).tolist(),
                'size_lwh_m':size.tolist(),'yaw_rad':math.atan2(axis[1],axis[0]),
                'box_axes_ego':axes.tolist(),'upright_for_yaw_IoU':bool(np.max(np.abs(axes[2,:2]))<.1 and abs(axes[2,2])>.98),
                'velocity_ego_mps':(reference[:3,:3]@v).tolist(),
                'center_world_m':vertices.mean(0).tolist(),'velocity_world_mps':v.tolist()})
        row={'frame':snapshot.frame,'timestamp_s':snapshot.timestamp.elapsed_seconds,'actors':actors,
             'invalid_actor_geometry':invalid,
             'ego_world_from_reference':ego.get_transform().get_matrix(),'sampled_images':step%self.config['GT_stride']==0}
        row['signal_phases_readback']={str(light.id):str(light.get_state()) for light in world.get_actors().filter('traffic.traffic_light')}
        row['signal_phase_readback_frame']=world.get_snapshot().frame
        if row['sampled_images']:
            folder=self.out/f'{snapshot.frame:08d}';folder.mkdir()
            row['cameras']={}
            for name,sink in self.queues.items():
                f=receive_frame(sink,snapshot.frame,15)
                if abs(f.timestamp-snapshot.timestamp.elapsed_seconds)>1e-5:raise ValueError('GT/RGB timestamp mismatch')
                raw=np.frombuffer(f.raw_data,np.uint8).reshape(360,640,4)[:,:,2::-1].copy()
                ob=diagnostics['observations'][name]
                for label,array in [('instance',raw),('predicted_seg',ob['ids'])]:
                    Image.fromarray(array).save(folder/f'{name}-{label}.png')
                det=[{'box':d['bbox_xyxy'],'label':d['class_id'],'score':d['score']} for d in ob['objects']]
                row['cameras'][name]={'detections':det,'RGB':inventory[name],
                    'instance_sha256':sha(folder/f'{name}-instance.png'),'predicted_seg_sha256':sha(folder/f'{name}-predicted_seg.png')}
            row['single_3D']=diagnostics['single_3D'];row['prior_3D']=diagnostics['prior_3D']
        write_json(self.out/f'{snapshot.frame:08d}.json',row)
