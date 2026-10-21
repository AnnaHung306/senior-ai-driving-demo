"""Same neural observations, original vs cached static geometric fusion."""
import gc,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_io import read,sha
from senior_ai.core import write_json
from senior_ai.live_perception import LivePerception
from senior_ai.bev_fusion import fuse_road
from senior_ai.road_fusion_cache import CachedRoadFusion

def main():
    out=ROOT/'artifacts/runs/phase10-fusion-benchmark-v1-001'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    folder=ROOT/'artifacts/runs/phase89-drive-pedestrian-probe-v1-001';report=read(folder/'report.json')
    selected=[report['records'][i] for i in [0,len(report['records'])//2,len(report['records'])-1]]
    write_json(out/'contract.json',{'source_sha256':sha(Path(__file__)),'source_run':folder.name,
        'frames':[x['frame'] for x in selected],'minimum_median_speedup':1.2,'required_all_outputs_array_equal':True,
        'repeats_per_variant_per_frame':5,'four_weights_unchanged_required':True,'original_Phase4_module_unchanged':True,
        'control_issued':False,'input':'three genuine saved eight-RGB bundles, persistent frozen CUDA inference'})
    engine=LivePerception(read(folder/'calibration.json'));cache=CachedRoadFusion(engine.cameras,engine.grid,engine.c);results=[]
    for entry in selected:
        r=read(folder/entry['file']);rgb={}
        for camera,item in r['RGB'].items():
            assert sha(folder/item['file'])==item['sha256'];rgb[camera]=np.array(Image.open(folder/item['file']))
        engine.infer(rgb,'fusion-benchmark',r['frame'],r['timestamp_s']);ob=engine.audit_output['observations']
        baseline=fuse_road(ob,engine.cameras,engine.grid,engine.c);cached=cache(ob)
        for key in ['probabilities','observed','support','geometric_coverage']:assert np.array_equal(baseline[key],cached[key]),key
        for camera in baseline['maps']:
            for left,right in zip(baseline['maps'][camera],cached['maps'][camera]):assert np.array_equal(left,right,equal_nan=True)
        times={'original':[],'cached':[]}
        for repeat in range(5):
            for variant in (['original','cached'] if repeat%2==0 else ['cached','original']):
                started=time.perf_counter()
                value=fuse_road(ob,engine.cameras,engine.grid,engine.c) if variant=='original' else cache(ob)
                times[variant].append((time.perf_counter()-started)*1000)
                assert np.array_equal(value['probabilities'],baseline['probabilities'])
        results.append({'frame':r['frame'],'times_ms':times,'all_output_arrays_exact':True})
    before=[t for r in results for t in r['times_ms']['original']];after=[t for r in results for t in r['times_ms']['cached']]
    speedup=float(np.median(before)/np.median(after));adopt=speedup>=1.2
    write_json(out/'report.json',{'status':'completed_actual_neural_observation_equivalence_benchmark','frames':results,
        'original_p50_ms':float(np.median(before)),'cached_p50_ms':float(np.median(after)),'speedup':speedup,'adopt_projection_cache':adopt,
        'not_a_new_model_or_accuracy_gain':True,'formal_quality_passed':False})
    if adopt:
        config=ROOT/'configs/phase10_optimization_v1.json'
        if config.exists():raise FileExistsError(config)
        write_json(config,{'projection_cache_enabled':True,'PNG_compress_level':1,
            'fusion_benchmark_sha256':sha(out/'report.json'),'PNG_benchmark_sha256':sha(ROOT/'artifacts/runs/phase10-PNG-benchmark-v1-001/report.json'),
            'geometry_equivalence':'same grid/rig/ground parameters, exact arithmetic order; cache reset per job'})
    del engine;gc.collect()
    import torch
    torch.cuda.empty_cache()
    print({'speedup':speedup,'adopt':adopt})

if __name__=='__main__':main()
