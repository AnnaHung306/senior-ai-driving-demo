"""Read-only confidence diagnostics on saved GT-crop validation predictions.

Thresholds below are exploratory abstention analyses. They do not change model
outputs, the supervised unknown class, frozen acceptance gates, or deployment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def percent(value):return '—' if value is None else f'{value:.2%}'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def analyze(run):
    state=read(run/'state.json')
    if state['status']!='completed' or sha(run/'best.pt')!=state['best_sha256']:
        raise ValueError('Completed intact phase model required')
    for filename,key in [('validation.json','validation_sha256'),('validation_frames.jsonl','validation_frames_sha256')]:
        if sha(run/filename)!=state[key]:raise ValueError('Validation receipt changed')
    result=read(run/'validation.json');names=[r['name'] for r in result['classes']]
    rows=[json.loads(line) for line in (run/'validation_frames.jsonl').read_text(encoding='utf-8').splitlines()]
    matrix=[[0]*4 for _ in range(4)]
    for row in rows:
        if row['split']!='validation':raise ValueError('Validation only')
        confidence=row['softmax_max_uncalibrated']
        if not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('Invalid confidence')
        matrix[row['target']][row['prediction']]+=1
    if matrix!=result['confusion_true_rows_predicted_columns']:raise ValueError('Per-crop confusion mismatch')
    bins=[]
    for i in range(10):
        items=[r for r in rows if i/10<=r['softmax_max_uncalibrated']<(i+1)/10 or i==9 and r['softmax_max_uncalibrated']==1.]
        if not items:continue
        accuracy=sum(r['target']==r['prediction'] for r in items)/len(items)
        mean=sum(r['softmax_max_uncalibrated'] for r in items)/len(items)
        bins.append({'lower':i/10,'upper':(i+1)/10,'count':len(items),'accuracy':accuracy,'mean_softmax_max':mean,'absolute_gap':abs(accuracy-mean)})
    thresholds=[];support=[sum(row) for row in matrix]
    for t in [0.,.5,.9,.95,.99,.999]:
        accepted=[r for r in rows if r['softmax_max_uncalibrated']>=t]
        red=[r for r in accepted if r['target']==1];red_green=sum(r['prediction']==3 for r in red)
        correct=[sum(r['target']==c and r['prediction']==c for r in accepted) for c in range(4)]
        thresholds.append({'threshold':t,'retained_predictions':len(accepted),'coverage':len(accepted)/len(rows),
            'accuracy_on_retained':sum(correct)/len(accepted) if accepted else None,
            'macro_correct_state_recall_including_abstentions':sum(correct[c]/support[c] for c in range(4))/4,
            'accepted_GT_red':len(red),'red_as_green_count':red_green,
            'red_as_green_rate_accepted_GT_red':red_green/len(red) if red else None,
            'red_as_green_rate_all_GT_red':red_green/support[1],
            'unknown_GT_predicted_known_count':sum(r['target']==0 and r['prediction']!=0 for r in accepted)})
    errors=[]
    for actual in range(4):
        for predicted in range(4):
            items=[r for r in rows if r['target']==actual and r['prediction']==predicted]
            if actual==predicted or not items:continue
            conf=[r['softmax_max_uncalibrated'] for r in items]
            errors.append({'target':names[actual],'prediction':names[predicted],'count':len(items),
                           'min_confidence':min(conf),'mean_confidence':sum(conf)/len(conf),'max_confidence':max(conf),
                           'count_at_least_0_99':sum(v>=.99 for v in conf)})
    return {'scope':'Saved GT-crop validation only. Exploratory thresholds are not adopted; model outputs/gates unchanged. Abstention is separate from the supervised unknown/unobservable class.',
            'limitations':'Correlated synthetic frames and few original groups; top-label calibration diagnostics are not general safety probabilities or independent-trial confidence intervals.',
            'checkpoint_sha256':state['best_sha256'],'validation_frames_sha256':state['validation_frames_sha256'],
            'tool_sha256':sha(Path(__file__)),'crops':len(rows),'accuracy':result['accuracy'],
            'top_label_ECE_10_bins':sum(b['count']/len(rows)*b['absolute_gap'] for b in bins),
            'confidence_bins':bins,'exploratory_abstention':thresholds,'error_confidence':errors}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run=args.run.resolve();out=args.output.resolve()
    if out.is_relative_to(run):raise ValueError('Output outside immutable run required')
    result=analyze(run);out.mkdir(parents=True,exist_ok=True)
    temp=out/'diagnostic.json.tmp';temp.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');temp.replace(out/'diagnostic.json')
    text='# 號誌驗證置信度診斷 — 未採用任何新門檻\n\n'
    text+=f"GT crop {result['crops']}個，accuracy {result['accuracy']:.4%}，10bins top-label ECE {result['top_label_ECE_10_bins']:.6f}。這是此validation上的描述，不是安全機率或獨立試驗泛化估計。\n\n"
    text+='棄權與原模型的unknown/unobservable語意類分開；下表只是離線探索，沒有變更模型、品質門檻、joint protocol或推論行為。含棄權recall把未保留的GT也列分母，紅→綠同時列保留GT紅及全部GT紅，避免分母掩飾錯誤。\n\n'
    text+='| Confidence門檻 | 保留率 | 保留預測正確率 | 含棄權macro recall | 紅→綠/保留GT紅 | 紅→綠/全部GT紅 |\n|---|---:|---:|---:|---:|---:|\n'
    for r in result['exploratory_abstention']:
        text+='| '+str(r['threshold'])+' | '+' | '.join(percent(r[key]) for key in ['coverage','accuracy_on_retained','macro_correct_state_recall_including_abstentions','red_as_green_rate_accepted_GT_red','red_as_green_rate_all_GT_red'])+' |\n'
    temp=out/'REPORT.md.tmp';temp.write_text(text,encoding='utf-8');temp.replace(out/'REPORT.md')
    print(json.dumps({'crops':result['crops'],'ECE':result['top_label_ECE_10_bins'],'error_confidence':result['error_confidence']}))
