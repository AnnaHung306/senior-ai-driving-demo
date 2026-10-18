"""Export actual trained results and model cards; incomplete evidence cannot pass."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.core import write_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def number(value):return '—' if value is None else f'{value*100:.2f}%'


def validate_benchmark(value,model_hashes,demo_frame,demo_inputs):
    """Reject stale model pairs, incomplete receipts and wrong demo bundles."""
    if not demo_inputs.get('input_manifest_sha256') or not isinstance(demo_inputs.get('input_RGB_sha256'),dict) or len(demo_inputs['input_RGB_sha256'])!=8:
        raise ValueError('Demo must bind the actual eight RGB image hashes')
    for key in ['input_manifest_sha256','input_RGB_sha256']:
        if value.get(key)!=demo_inputs[key]:raise ValueError('Benchmark RGB input mismatch')
    keys={'detector_sha256':'detection','segmentation_sha256':'segmentation','phase_sha256':'light_state'}
    if any(key in value for key in keys):
        if any(value.get(key)!=model_hashes[name] for key,name in keys.items()):
            raise ValueError('Integrated benchmark model tuple mismatch')
        if value.get('frame')!=demo_frame or value.get('cameras')!=8:
            raise ValueError('Integrated benchmark must measure the demo eight-view bundle')
        times=value.get('eight_view_RGB_decode_to_CPU_perception_ms',{}).get('samples',[])
        if not times or len(times)!=value.get('measured_bundles') or value.get('warmup_bundles',0)<1:
            raise ValueError('Integrated benchmark has no complete warmed measurements')
        if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in times):
            raise ValueError('Invalid integrated benchmark duration')
        return 'integrated'
    task=value.get('task')
    if task not in ['detection','segmentation'] or value.get('checkpoint_sha256')!=model_hashes[task]:
        raise ValueError('Individual benchmark task/checkpoint mismatch')
    if value.get('samples',0)<1 or value.get('warmup',0)<1:
        raise ValueError('Individual benchmark lacks warmed measurements')
    for name in ['network_and_result_to_cpu_ms','warm_filesystem_RGB_manifest_to_result_ms']:
        for percentile in ['p50','p95']:
            v=value.get(name,{}).get(percentile)
            if not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<=0:
                raise ValueError('Invalid individual benchmark duration')
    return task


def final_receipt(run,state):
    path=run/'final_evaluation.json'
    if not path.exists():return {'status':'not_evaluated'}
    value=read(path)
    if value.get('checkpoint_sha256')!=state['best_sha256']:
        raise ValueError('Final evaluation belongs to another checkpoint: '+run.name)
    if value.get('status')=='passed':
        marker=run/'lockbox_started.json'
        if not marker.exists() or read(marker).get('best_sha256')!=state['best_sha256']:
            raise ValueError('Passed heldout receipt lacks bound evaluation marker: '+run.name)
    return value


def metric_table(task,result):
    if task=='detection':
        text='| 類別 | GT 物件 | AP50 | AP50:95 | Precision@.5 | Recall@.5 |\n|---|---:|---:|---:|---:|---:|\n'
        for name,r in result['per_class'].items():text+=f'| {name} | {r["ground_truth_objects"]} | {number(r["AP50"])} | {number(r["AP50_95"])} | {number(r["precision_at_score"])} | {number(r["recall_at_score"])} |\n'
        return f'mAP50 {number(result["mAP50"])}；mAP50:95 {number(result["mAP50_95"])}。\n\n'+text
    if task=='segmentation':
        text='| 類別 | GT pixels | IoU | Precision | Recall |\n|---|---:|---:|---:|---:|\n'
        for r in result['classes']:text+=f'| {r["name"]} | {r["support_pixels"]} | {number(r["iou"])} | {number(r["precision"])} | {number(r["recall"])} |\n'
        return f'16 類 focus mIoU {number(result["focus_macro_iou"])}；可見道路界面 F1（2px）{number(result["visible_margin"]["f1"])}。\n\n'+text
    text='| 類別 | GT crops | Precision | Recall |\n|---|---:|---:|---:|\n'
    for r in result['classes']:text+=f'| {r["name"]} | {r["support"]} | {number(r["precision"])} | {number(r["recall"])} |\n'
    return f'Macro recall {number(result["macro_recall"])}；accuracy {number(result["accuracy"])}。\n\n'+text


def export(args):
    if args.output.exists():raise FileExistsError('Use a new report version')
    runs={'detection':args.detector_run,'segmentation':args.segmentation_run,'light_state':args.light_run}
    states={name:read(run/'state.json') for name,run in runs.items()}
    for name,run in runs.items():
        if states[name]['status']!='completed' or sha(run/'best.pt')!=states[name]['best_sha256']:raise ValueError('Completed intact model required: '+name)
        if not states[name].get('checkpoint_reload_identical'):raise ValueError('Verified checkpoint reload required: '+name)
        for filename,key in [('validation.json','validation_sha256'),('validation_frames.jsonl','validation_frames_sha256')]:
            if sha(run/filename)!=states[name].get(key):raise ValueError('Validation artifact changed: '+name+'/'+filename)
    finals={name:final_receipt(run,states[name]) for name,run in runs.items()}
    model_hashes={name:state['best_sha256'] for name,state in states.items()}
    demo=read(args.demo/'perception.json')
    for key,name in [('detector_sha256','detection'),('segmentation_sha256','segmentation'),('phase_sha256','light_state')]:
        if demo.get(key)!=model_hashes[name]:raise ValueError('Demo checkpoint mismatch')
    if demo['source']!='learned_prediction' or demo['model_inputs']!='RGB_only' or len(demo['cameras'])!=8:raise ValueError('Actual RGB-only eight-view demo required')
    benchmarks=[(source,read(source)) for source in args.benchmark]
    kinds={validate_benchmark(value,model_hashes,demo['frame'],demo) for _,value in benchmarks}
    if kinds!={'detection','segmentation','integrated'}:raise ValueError('Detector, segmentation and integrated eight-view benchmarks required')
    if len({source.name for source,_ in benchmarks})!=len(benchmarks):raise ValueError('Benchmark filenames must be distinct')
    reserved={'summary.json','perception.json','REPORT.md','surround.png'}
    if any(source.name in reserved for source,_ in benchmarks):raise ValueError('Benchmark filename collides with export artifact')
    origins={name:read(run/'provenance.json') for name,run in runs.items() if name!='light_state'}
    main_audit_sha=sha(args.root/'audit.json')
    if any(origin['data']['audit_sha256']!=main_audit_sha for origin in origins.values()) or states['light_state'].get('data',{}).get('audit_sha256')!=main_audit_sha:
        raise ValueError('Export models must bind the reported main audit')
    args.output.mkdir(parents=True);summary={'scope':'ClearNoon ideal simulation only','models':{},'phase3_complete':False,
        'source_sha256':sha(Path(__file__)),'data_audit_sha256':sha(args.root/'audit.json'),'data_review_sha256':sha(args.root/'visual_review.json')}
    for name,run in runs.items():
        output=args.output/name;output.mkdir();state=states[name];final=finals[name]
        model={'run':run.as_posix(),'checkpoint':(run/'best.pt').as_posix(),'checkpoint_sha256':state['best_sha256'],
               'training_status':state['status'],'heldout_status':final['status']}
        if name in origins and origins[name].get('training_supplement'):
            model['training_supplement']=origins[name]['training_supplement']
        summary['models'][name]=model
        for filename in ['state.json','config.json','provenance.json','coverage.json','dataset_statistics.json','overfit.json','initial_validation.json','epochs.jsonl','validation.json','validation_quality.json','validation_joint.json','validation_joint_quality.json','best_validation.json','development.json','lockbox.json','development_crops.json','lockbox_crops.json','development_joint.json','lockbox_joint.json','lockbox_dynamic_joint.json','lockbox_controlled_joint.json','lockbox_started.json','final_evaluation.json']:
            if (run/filename).exists():shutil.copyfile(run/filename,output/filename)
        for split in ['validation','development','lockbox']:
            folder=run/(split+('_phase_errors' if name=='light_state' else '_errors'))
            if folder.exists():shutil.copytree(folder,output/folder.name)
        card=f'# {name} — {run.name}\n\n實際訓練完成；heldout 驗收：`{final["status"]}`。權重保留本機 ignored 路徑 `{run.as_posix()}/best.pt`，未宣稱已上傳或納入 Git。\n\nSHA256 `{state["best_sha256"]}`。\n\n'
        if name=='detection':card+='MobileNetV3 Faster R-CNN，P2/P3/P4/P5＋pool，官方 COCO backbone／部分 FPN、RPN transfer，再自行微調八類可見物件。兩輪車含同 ID rider/body；臨時障礙物目前只涵蓋核准交通桶資產。使用明確 ignore 的 visible-box 101 recall AP，非完整 COCO evaluator。\n\n'
        elif name=='segmentation':
            c=read(run/'config.json')['segmentation']
            card+=('MobileNetV3 encoder（官方 COCO detector 完整 body transfer、FrozenBN）＋自行訓練多尺度 U-Net decoder，17類；CUDA BF16 autocast、FP32參數／loss／結果；paired scale/crop/ignore-padding/flip只用於train。' if c.get('architecture')=='mobilenet_unet' else 'Tiny U-Net 17 類，沿用舊自訓八類 backbone 及明確對應 heads，新增類別自行訓練。')
            card+=' weighted CE＋foreground Dice。road 是語意道路，可見道路界面只取 road/line 與 sidewalk/terrain 相鄰處，不推斷遮擋後路界或可安全通行區域。\n\n'
        else:card+='LightStateNet 四類 RGB crop 分類；native phase 只作 GT，必須有可見相容燈色，否則 unknown。正式輸入沒有模擬器狀態。主動態 lockbox 黃燈不足，依訓練前凍結計畫額外補收固定視角相位影像；動態／控制場景分別報告，合併結果只供號誌驗收。\n\n'
        if name in origins:
            c=read(run/'config.json')[name]
            if c.get('initial_checkpoint_path'):card+=f'本次從專案已訓練權重 `{c["initial_checkpoint_path"]}`（SHA `{c["initial_checkpoint_sha256"]}`）再微調，重設optimizer/scheduler；不是從零初始化。\n\n'
            if name=='detection':card+=f'原始RGB/框為640×360；模型內部resize最短邊{c.get("min_size",360)}、最長邊{c.get("max_size",640)}，輸出框回復原始座標。\n\n'
            if model.get('training_supplement'):
                extra=model['training_supplement'];card+=f'另用train-only靜態分割補景：{extra["views"]}個獨特視圖、{extra["groups"]}個相關道路錨點，每輪重複採樣{extra["sampling_repeats"]}次；不是新增動態路線。補景audit SHA `{extra["audit_sha256"]}`。\n\n'
        for split,label in [('validation','驗證／選模'),('development','開發測試'),('lockbox','保留測試')]:
            filename=split+('_crops.json' if name=='light_state' and split!='validation' else '.json');path=run/filename
            if path.exists():card+=f'## {label}\n\n'+metric_table(name,read(path))+'\n'
            if name=='light_state' and (run/(split+'_joint.json')).exists():
                joint=read(run/(split+'_joint.json'));card+=f'含偵測漏判的正確狀態 macro recall：{number(joint["macro_correct_state_recall"])}。\n\n'
                card+='| 狀態 | GT 物件 | 含偽號誌的狀態 Precision | 含漏判正確狀態 Recall |\n|---|---:|---:|---:|\n'
                for r in joint['classes']:card+=f'| {r["name"]} | {r["gt_objects"]} | {number(r["correct_state_precision_including_false_light_detections"])} | {number(r["correct_state_recall_including_missed_detections"])} |\n'
                card+='\n'
        card+='## 使用限制\n\n八方向逐鏡頭二維感知；沒有跨鏡頭融合、深度／BEV／3D occupancy、導航規劃或 learned vehicle control。只有晴朗白天合成資料；共用資產、短序列與少量城鎮／路線高度相關，像素／frames 不是獨立試驗，沒有可靠廣泛泛化信賴區間。分類與偵測分數均未校準。規則資料生成器的零碰撞不代表 learned driving。\n'
        (output/'MODEL_CARD.md').write_text(card,encoding='utf-8',newline='\n')
    for source,value in benchmarks:
        shutil.copyfile(source,args.output/source.name)
    shutil.copyfile(args.demo/'surround.png',args.output/'surround.png');shutil.copyfile(args.demo/'perception.json',args.output/'perception.json')
    summary['phase3_complete']=all(m['heldout_status']=='passed' for m in summary['models'].values())
    summary['rgb_only_demo']=True;summary['benchmark_files']=[p.name for p in args.benchmark]
    write_json(args.output/'summary.json',summary)
    text='# Phase 3 晴天正式結果\n\n'+('三個模型已完成實際訓練及保留資料驗收。' if summary['phase3_complete'] else '訓練結果已保存，Phase 3 尚未通過全部品質條件。')+'\n\n'
    text+='原始主資料：18 個 30 秒路線群組、2700 bundles、八方向四模態，共 86,400 PNG；完整 CRC/SHA／位姿／語意一致性及 assistant 視覺抽查通過。train Town01/Town03、validation Town04、development Town05、lockbox Town02。\n\n'
    text+='| 模型 | 保留測試驗收 | 逐類結果 |\n|---|---|---|\n'
    for name,m in summary['models'].items():text+=f'| {name} | {m["heldout_status"]} | [{name} model card]({name}/MODEL_CARD.md) |\n'
    text+='\n![八鏡頭實際 learned predictions](surround.png)\n\n錯例與逐城鎮／逐類 JSON 位於各模型目錄。權重與完整原圖在本機 artifacts/data，Git 保存程式、收據、模型卡與選定錯例。效能收據只涵蓋所列推論範圍，不宣稱 20Hz 閉環。Phase 4 的跨鏡頭融合與三維重建尚未開始。\n'
    (args.output/'REPORT.md').write_text(text,encoding='utf-8',newline='\n');print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--detector-run',type=Path,required=True);p.add_argument('--segmentation-run',type=Path,required=True);p.add_argument('--light-run',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--demo',type=Path,required=True);p.add_argument('--benchmark',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);export(p.parse_args())
