"""Paired lossless PNG encoding experiment; criteria saved before measurements."""
import hashlib,io,json,statistics,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'artifacts/runs/phase10-PNG-benchmark-v1-001'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    names=['phase89-drive-unseen-straight1-v1-001','phase89-drive-signal-probe-v1-001','phase89-drive-pedestrian-probe-v1-001']
    files=[]
    for name in names:
        folder=ROOT/'artifacts/runs'/name;report=json.loads((folder/'report.json').read_text(encoding='utf8'))
        row=report['records'][len(report['records'])//2]
        f=folder/row['file'];r=json.loads(f.read_text(encoding='utf8'))
        files += [folder/r['RGB'][camera]['file'] for camera in ['front','rear','left','right']]
    contract={'levels':[6,1],'paired_repeats':3,'minimum_median_reduction_fraction':.2,'maximum_total_size_ratio':2.,
              'lossless_pixel_equality_required':True,'files':[str(p.relative_to(ROOT)) for p in files],
              'same_images_same_Pillow_same_storage':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf8')
    rows=[]
    for index,p in enumerate(files):
        image=Image.open(p).convert('RGB');pixels=np.asarray(image)
        for repeat in range(3):
            for level in ([6,1] if (index+repeat)%2==0 else [1,6]):
                target=out/'encoded'/f'{index:02d}-{repeat}-{level}.png';target.parent.mkdir(exist_ok=True)
                started=time.perf_counter();image.save(target,compress_level=level);digest=hashlib.sha256(target.read_bytes()).hexdigest();elapsed=(time.perf_counter()-started)*1000
                assert np.array_equal(np.asarray(Image.open(target)),pixels)
                rows.append({'image':str(p.relative_to(ROOT)),'level':level,'repeat':repeat,'encode_write_hash_ms':elapsed,'bytes':target.stat().st_size,'sha256':digest,'decoded_pixels_exact':True})
    times={level:statistics.median(r['encode_write_hash_ms'] for r in rows if r['level']==level) for level in [6,1]}
    sizes={level:sum(r['bytes'] for r in rows if r['level']==level) for level in [6,1]}
    reduction=1-times[1]/times[6];ratio=sizes[1]/sizes[6]
    result={'status':'completed_paired_lossless_encoding_probe','per_image_p50_ms':times,'relative_reduction':reduction,
            'size_ratio_level1_over_level6':ratio,'adopt_level1':reduction>=.2 and ratio<=2.,'rows':rows,
            'limits':'Concurrent CARLA/GPU audit present; paired interleaving reduces but does not remove host scheduling noise. No model acceleration claim.'}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':main()
