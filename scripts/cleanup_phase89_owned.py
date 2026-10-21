"""Recover only our stopped controller's actors after process interruption."""
import carla,json,sys
c=carla.Client('127.0.0.1',2000);c.set_timeout(15);w=c.get_world()
if '--restore-orphaned-sync' in sys.argv:
    # Caller must first confirm the stopped worker has exited. Allow the newly
    # connected client's actor cache to receive a world snapshot before lookup.
    settings=w.get_settings();settings.synchronous_mode=False;settings.fixed_delta_seconds=None;w.apply_settings(settings)
    w.wait_for_tick(10)
owned=[a for a in w.get_actors().filter('vehicle.*') if a.attributes.get('role_name')=='senior_AI_RGB_hybrid']
ids={a.id for a in owned};children=[a for a in w.get_actors() if a.parent and a.parent.id in ids]
for v in owned:v.apply_control(carla.VehicleControl(brake=1.))
for s in children:
    try:s.stop()
    except RuntimeError:pass
for a in children+owned:a.destroy()
s=w.get_settings()
if owned or '--restore-orphaned-sync' in sys.argv:
    s.synchronous_mode=False;s.fixed_delta_seconds=None;w.apply_settings(s)
print(json.dumps({'owned_vehicle_ids_destroyed':sorted(ids),'owned_sensors_destroyed':len(children),'synchronous':w.get_settings().synchronous_mode}))
