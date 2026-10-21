"""Bounded device-level sample while CARLA/inference run; no configuration changes."""
import json,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'artifacts/reports/phase10-host-sampling-v1-001.json';rows=[]
    if out.exists():raise FileExistsError(out)
    for _ in range(30):
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
        rows.append({'sample_monotonic_s':time.monotonic(),'raw_memory_MiB_utilization_percent_temperature_C_power_W':raw.strip()})
        time.sleep(2)
    d=shutil.disk_usage(ROOT)
    out.write_text(json.dumps({'status':'bounded_shared_device_samples','samples':rows,'disk_free_bytes':d.free,
        'GPU_includes_CARLA_desktop_and_inference':True,'not_per_model_memory':True,'training_started':False},indent=2)+'\n',encoding='utf8')
    print({'samples':len(rows),'out':str(out)})

if __name__=='__main__':main()
