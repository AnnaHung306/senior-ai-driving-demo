"""Bind final Phase10 results, scope gaps and post-run input inventory."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'artifacts/reports'
EVALUATION = ROOT / 'artifacts/runs/phase10-evaluation-final-v1-002'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def pct(value):
    return 'N/A' if value is None else f'{value * 100:.2f}%'


def main():
    results = read(EVALUATION / 'report.json')
    integrity = read(REPORTS / 'phase10-integrity-final.json')
    browser = read(REPORTS / 'phase10-browser-verified-final.json')
    assert len(results['runs']) == 24 and sum(r['frames'] for r in results['runs']) == 2194
    assert integrity['frames'] == 2194 and integrity['RGB_images'] == 17552
    assert integrity['GT_and_predicted_semantic_images'] == 6288
    assert browser['frames'] == 67 and browser['objects'] == 83 and browser['forecasts'] == 64
    assert results['sampled_eight_RGB_bundles'] == 393
    assert read(REPORTS/'phase10-teardown-final.json')['listeners'] == []
    assert len(read(REPORTS/'phase10-offline-HTTP-final.json')['checks']) == 27
    assert 'Ran 322 tests' in (REPORTS/'phase10-regression-final.log').read_text(encoding='utf8')
    assert 'pass 52' in (REPORTS/'phase10-JS-final.log').read_text(encoding='utf8')
    assert read(REPORTS/'phase10-browser-001/console-final.json') == []
    assert read(REPORTS/'phase10-browser-001/focused-video-playback.json')['readyState'] == 4

    contracts = [read(ROOT/'artifacts/runs'/r['run_id']/'contract.json') for r in results['runs']]
    models = contracts[0]['models']
    missing_cuboid = []
    for run, contract in zip(results['runs'], contracts):
        assert all(value == models[name] for name, value in contract['models'].items())
        if 'cuboid' not in contract['models']:
            source=ROOT/'artifacts/runs'/run['run_id']/'source/src/senior_ai/live_perception.py'
            assert models['cuboid']['sha256'] in source.read_text(encoding='utf8')
            missing_cuboid.append(run['run_id'])
        assert contract['actor_GT_in_planning'] is False
        assert contract.get('development_or_lockbox_used', contract.get('GT_assets_or_development_lockbox_used')) is False
        assert contract['autopilot_or_BasicAgent_used'] is False
    for model in models.values():
        assert sha(ROOT/model['path']) == model['sha256']
    changed_old = subprocess.check_output(['git','diff','--name-only','f378cd1','--',
        'evidence/phase0','evidence/phase1','evidence/phase2','evidence/phase3',
        'evidence/phase4','evidence/phase5','evidence/phase6','evidence/phase7','evidence/phase89'],cwd=ROOT,text=True).strip()
    assert changed_old == '', changed_old

    # Explicitly post-run. Contemporaneous hashes are separately in each record/contract.
    inputs = []
    for run in results['runs']:
        folder = ROOT/'artifacts/runs'/run['run_id']
        candidates = [folder/'report.json', folder/'contract.json', folder/'calibration.json']
        candidates += list((folder/'audit').rglob('*.json')) + list((folder/'audit').rglob('*.png'))
        candidates += list((folder/'frames').glob('*/record.json'))
        for path in sorted(set(candidates)):
            if path.is_file():
                inputs.append({'path':path.relative_to(ROOT).as_posix(),
                    'bytes':path.stat().st_size,'sha256':sha(path)})
    write(REPORTS/'phase10-input-inventory-final.json', {'status':'post_run_inventory',
        'created_utc':datetime.now(timezone.utc).isoformat(), 'files':inputs,
        'not_contemporaneous_collection_manifest':True,
        'RGB_image_hashes':'Each contemporaneous record includes all eight RGB hashes; integrity receipt rechecked all 17552 bytes.'})

    visual_sets = [
        ('phase10-visual-QA-v1-001', ['contact-0.jpg','contact-1.jpg','contact-2.jpg',
            'semantic-IoU.png','performance.png','performance-comparison.png','forecast-vs-persistence.png'] +
            [f'route-contact-{i}.jpg' for i in range(6)]),
        ('phase10-visual-supplement-v1-001',[f'contact-{i}.jpg' for i in range(3)]),
        ('phase10-route-visual-v2-001',[f'contact-{i}.jpg' for i in range(6)])]
    observations = [
        'All listed contact sheets and four aggregate plots were actually viewed, not inferred from file existence.',
        'Original lowest road-IoU panels include road-empty tiny-FP negatives; supplemental road panels require >=5000 GT road pixels.',
        'Car/truck high-confidence confusion and bicycle/rider label ambiguity are visible; distant/truncated missed lamps remain.',
        'GT and prediction masks differ at grass/road boundaries; high aggregate road IoU does not imply metric BEV correctness.',
        'Speed curves expose oscillating stop/go and false stops. Fault/cancel curves end before separately verified cleanup speed zero.',
        'Original vertical/horizontal straight maps collapsed due equal-aspect tiny extent; v2 expands both limits in meters. Old plots retained.',
        'Desktop and 375px mobile actual screenshots reviewed; mobile has no horizontal overflow but text is small. Real 200% zoom not verified.',
        '16:9 focused WebM was actually decoded/playing in browser, 1280x720 duration13.4s; recorded5Hz simulation playback, not live inference.'
    ]
    for name, viewed in visual_sets:
        folder=ROOT/'artifacts/runs'/name
        write(folder/'assistant_review.json', {'status':'actual_visual_review_completed',
            'viewed':[{'file':n,'sha256':sha(folder/n)} for n in viewed],
            'observations':observations,'original_catalog_preserved':True})

    rows = read(REPORTS/'phase10-browser-001/final-pedestrian-v2-frames.json')
    resources=[r['renderer_memory'] for r in rows]
    writes = sum((ROOT/x['path']).stat().st_size for x in inputs if x['path'].endswith('.png'))
    rgb_bytes=sum(p.stat().st_size for run in results['runs'] for p in
                  (ROOT/'artifacts/runs'/run['run_id']/'frames').glob('*/*.png'))
    matrix = [
        {'area':'fixed_new_routes','result':'2/6 arrived; all six preserved','coverage_gap':'same-town assets, no heldout claim'},
        {'area':'2D_accuracy','result':'3144 visible GT/pred mask images; AP50:95/class/size/confusion/score analysis','coverage_gap':'no GT for bus/motorcycle/temporary obstacle; visible-mask protocol differs old validation'},
        {'area':'3D_and_forecast','result':'51 dynamic GT; strict and ignore-aware; learned/prior paired33; CV/persistence paired14 at2s','coverage_gap':'sparse tracking, low future coverage, no full MOT/occupancy-depth GT'},
        {'area':'signal_state','result':'1084 phase-bracketed samples, exact CPU head inputs; ROI association diagnostic','coverage_gap':'234 excluded; association and actual violation certification unresolved'},
        {'area':'performance','result':'2193 timed cycles; 257 noGT; paired cache and PNG; loopback timing','coverage_gap':'one old applied-control record missing timing before instrumentation repair; no public-network or per-kernel GPU trace'},
        {'area':'UI','result':'67 final frames,83 objects,64 Float32 forecasts; eight RGB; desktop/mobile; actual drag/play/seek','coverage_gap':'real200% zoom not verified; small text and RAF long frames remain'},
        {'area':'faults_lifecycle','result':'real sensor.stop, validity, setup exception, WebGL loss, close/watchdog, lock contention','coverage_gap':'no intentional live solid-line crossing; finite tested cases not all failures'},
        {'area':'resources_storage','result':'30 shared-device samples; per-frame renderer resources and actual artifact sizes','coverage_gap':'no sustained soak or per-model memory isolation'},
        {'area':'delivery','result':'CPU replay ZIP609 manifestfiles/512RGB; focused real13.4s WebM','coverage_gap':'no external/public deployment; raw RGB and weights local ignored'},
        {'area':'regression_provenance','result':'322 Python,52 JS,27 HTTP,2194 actual Three frames; frozen source/weights checks','coverage_gap':'engineering correctness is not formal model quality'}
    ]
    original=[r for r in results['runs'] if r['run_id'].endswith('-audit-v1-001')]
    final_cases=[r for r in results['runs'] if any(r['run_id']==f'phase10-drive-{case}' for case in
        ['blocker-probe-v1-002','bicycle-probe-v1-002','pedestrian-probe-v1-001','lead-probe-v1-001','long-red-probe-v1-001'])]
    assert len(original)==6 and sum(r['passed'] for r in original)==2
    assert len(final_cases)==5 and all(r['passed'] for r in final_cases)
    validation={'status':'completed_systematic_bounded_engineering_audit',
        'created_utc':datetime.now(timezone.utc).isoformat(),'baseline_commit':'f378cd1',
        'formal_quality_passed':False,'all_possible_bugs_proven_absent':False,
        'attempts':24,'nonempty_runs':23,'saved_frames':2194,'timed_cycles':2193,
        'RGB_images':17552,'GT_and_predicted_semantic_images':6288,
        'original_preselected_routes':original,'original_route_completion':'2/6',
        'controlled_final_cases_not_replacement_denominator':final_cases,
        'models_unchanged':models,'runtime_example':contracts[-1].get('runtime'),
        'earlier_contracts_without_explicit_cuboid_entry':missing_cuboid,
        'earlier_cuboid_binding':'Literal expected SHA in contemporaneously frozen live_perception.py; later contracts also list cuboid explicitly. Setup-fault exits before model loading.',
        'source_coverage_version_counts':{'full_loaded_dependency_closure':sum('source_closure' in c for c in contracts),'legacy_main_adapter_snapshot':sum('source_closure' not in c for c in contracts)},
        'old_tracked_phase0_through89_evidence_changed':False,
        'development_lockbox_scored':False,'training_or_new_packages':False,
        'RGB_bytes_local':rgb_bytes,'GT_and_predicted_PNG_bytes_local':writes,
        'browser_renderer_geometry_min':min(x['geometries'] for x in resources),
        'browser_renderer_geometry_max':max(x['geometries'] for x in resources),
        'browser_renderer_geometry_final':resources[-1]['geometries'],
        'browser_renderer_texture_max':max(x['textures'] for x in resources),
        'resource_leak_soak_passed':False,'completeness_matrix':matrix,
        'authoritative_evaluation':EVALUATION.relative_to(ROOT).as_posix(),
        'authoritative_evaluation_sha256':sha(EVALUATION/'report.json'),
        'input_manifest_sha256':sha(REPORTS/'phase10-input-inventory-final.json'),
        'issues_document_sha256':sha(ROOT/'PHASE10_ISSUES.md'),
        'limits':results['limits']}
    write(REPORTS/'phase10-validation-final.json',validation)

    lines=['\n## 最終結果（2026-10-21）\n',
        '**Phase10系統稽核與工程修復完成；正式駕駛／模型品質未通過。** 六條預先固定的新路線只有2/6抵達。所有24次attempt（含故障刺激、取消與儀器失敗）保留，23次有影格，共2194幀／17552張RGB。獨立GT每五幀取樣共393組八視角、3144張影像；另保存6288張GT與預測語意PNG。四權重未變、沒有新訓練、套件安裝或舊development／lockbox評分。\n',
        '權威評估是 `artifacts/runs/phase10-evaluation-final-v1-002`；早期001／六案／preliminary均保留。封存入口：[證據README](evidence/phase10/system-audit-v1-001/README.md)，具體缺陷與處置見[24項問題登錄](PHASE10_ISSUES.md)。\n',
        '### 固定六案與受控回歸\n',
        '| 原始路線 | 長度m | 幀數 | 結果 | 末端距離m | 最大橫向誤差m |\n|---|---:|---:|---|---:|---:|']
    for r in original:
        lines.append(f"| {r['run_id'].removeprefix('phase10-drive-').removesuffix('-audit-v1-001')} | {r['distance_m']:.2f} | {r['frames']} | {r['status']} | {r['goal_distance_m']:.3f} | {r['max_cross_track_m']:.3f} |")
    lines += ['\n六案碰撞／實線事件皆0，但只代表低速且有限場景觀測；保守停止不能算抵達。來源、controller重算與清理均驗證，不能取代完成率。路線catalog只依地圖固定選取，同城／視野／資產及訓練歷史不獨立，逐案重疊距離公開，沒有heldout或泛化通過宣稱。\n',
        '工程修復後另外五個具名受控案例抵達：障礙車移除恢復76幀、行人橫穿67、自行車65、前車煞停／恢復111、15秒紅燈→綠燈轉彎236。它們不是五個新的獨立成功率樣本。blocker001因Windows鎖失敗、自行車001因GT bbox生命週期失敗均保留；setup-fault零影格也是一次attempt。七次web job含harness取消、WebGL、關頁與watchdog刺激，不混為正常導航完成率。\n',
        'right2 F30328的道路支持RGB soft／hard與GT同幾何oracle一致；末切線前向比.6297未達.7，沒有套用終點延伸，因此「延伸超過終點造成失敗」在此案例被排除。尚待檢查曲路投影、視野與地圖／取樣對齊，沒有修改.6證據門檻或重跑改寫2/6。號誌關聯存在大量未能解析樣本與缺少對應停止點的停等；四次停止點穿越代理無穩定紅燈，不足以認證遵規。\n',
        '### 語意、偵測、號誌與三維準確率\n',
        'GT由獨立instance cameras及capture snapshot產生，只進audit，不輸入主planner；主路面／物件仍來自八RGB模型。raw IDs、忽略未知／動態策略、可見tight框≥16px且邊長≥4px、score .5與IoU協議均保存。這套可見框協議不等於舊完整validation，不直接宣稱模型退步或進步；temporal重複影格高度相關，沒有獨立樣本置信區間。\n',
        '| 語意類別 | GT像素 | IoU |\n|---|---:|---:|']
    for c in results['segmentation']['per_class']:
        if c['id'] != 12:
            lines.append(f"| {c['name']} | {c['support_pixels']} | {pct(c['IoU'])} |")
    lines += ['\nWall依使用者範圍不作改善目標；無GT支持不能算完美。道路98.06%是2D像素統計，不能當作BEV、深度或閉環品質。\n',
        '| 可見框類別 | GT框 | AP50 | AP50:95 | P@.5 | R@.5 |\n|---|---:|---:|---:|---:|---:|']
    for name,c in results['detection']['per_class'].items():
        lines.append(f"| {name} | {c['ground_truth_objects']} | {pct(c['AP50'])} | {pct(c['AP50_95'])} | {pct(c['precision_at_score'])} | {pct(c['recall_at_score'])} |")
    lines += ['\n同時保存strict-no-ignore與ignore-aware分數；可見但不在actor snapshot的靜態車輛不可直接判為確定假陽性。car在最高分不分類IoU50配對中44/77被叫truck，與class-aware AP全域greedy的4個TP是不同診斷協議。car小框Recall0/38，中框4/39；可見性、截斷與各類分數可靠度詳JSON。行人Recall高但Precision15.44%，表示誤報也很多。\n',
        '號誌head另做1084個phase前後夾定且相位不變的可見crop CPU重播，234個無法夾定者排除；保存exact head_inputs.npz，不稱CUDA bit-exact。GT crop macro Recall85.67%，detector＋state joint macro42.37%；joint red91.45%、yellow37.50%、green40.40%、unknown .12%。GT unknown被head判green173個，需釐清燈頭可觀測性與方向，不能直接算闖紅燈。\n',
        '| 三維／時序指標 | 實測 | 解讀 |\n|---|---|---|',
        '| 可見動態GT／預測 | 51／239 | 視角合併後actor級；靜態未綁定另列 |',
        '| strict中心TP／FP／FN | 26／213／25 | P10.88%、R50.98%；不分類中心TP50 |',
        '| ignore-aware中心TP／FP／FN | 26／197／25 | 16個僅ignored前景來源保持未評，P11.66% |',
        '| full3D IoU≥.25 | 2/51，Recall3.92% | 不可用中心接近取代尺寸／朝向品質 |',
        '| 條件中心MAE／3D IoU均值／yaw MAE | .531m／.1499／51.27° | 僅成功匹配26個，不含漏檢 |',
        '| 同33個2D TP：learned／prior中心MAE | .6406／.6361m | 無明顯改善 |',
        '| 同33個2D TP：learned／prior IoU | .10445／.24170 | learned尺寸與朝向表現較弱；未依此自動換模型 |',
        '| 速度向量MAE | .4835m/s（21匹配） | 動態12個.6562；靜態9個.2532 |',
        '| 抽樣連續ID switch／間隙重識別變ID | 1／3 | 不是完整MOT／HOTA統計 |',
        '| 1秒ADE／FDE／GT覆蓋 | .6793／.8995m／18/46 | 相同匹配靜止基準FDE .8981m |',
        '| 2秒ADE／FDE／GT覆蓋 | .9139／1.4536m／14/41=34.15% | 靜止基準FDE1.5066m，均值差小 |',
        '| 2秒FDE最大值 | CV4.8298m／靜止2.3563m | 平均略好仍有更差尾端 |',
        '\n未來消失、右設限、非upright與無支持類別分開標示。沒有獨立metric BEV深度／全occupancy GT；不能以語意oracle同投影推導幾何準確率。\n',
        '### 效能、畫面與資源\n',
        '| 不掛GT的257個循環 | p50 ms | p95 ms | max ms |\n|---|---:|---:|---:|']
    for key,c in results['performance']['without_GT'].items():
        lines.append(f"| {key} | {c['p50']:.3f} | {c['p95']:.3f} | {c['max']:.3f} |")
    lines += ['\n總計2193個timing，而保存record2194個：舊儀器失敗時有一個已記錄control缺timing，修復後finally記錄；分母不補造。無GT仍約1.2秒／輪，200ms即時門檻未通過。原straight1分段中位數：input/seg137.09、八路det/light403.49、BEV/temporal/retainedlight641.26、cuboid/motion1.42ms；不能把這些中位數加成總p50。startup逐run保留，冷啟動／fault樣本不隱藏。\n',
        '兩項有界同輸入優化：靜態投影cache在3組真八RGB觀測中所有陣列與maps exact，fusion p50 516.02→78.63ms（6.56倍僅該段）；12張實圖×3 paired PNG level6→1，encode/write/hash p50 44.54→24.43ms，減45.15%，bytes×1.555、解碼exact，通過事前≥20%時間／≤2倍大小篩選。採用的optimization config綁benchmark hash。相同54m route原GT版total p50 1623.93ms、優化noGT版1187.70ms；GT／交通負載不同，不能將全部差異當cache因果。\n',
        '最終真瀏覽器完整67幀，83物件、64條實際Float32預測buffer、道路／路線／控制／八RGB逐幀來源相符；後端全部2194幀／1338個cuboid使用實際Three驗證。CPU paint 135次p50 .7／p95 1.5ms；512個active RAF interval p50約16.6／p95 41.735ms，仍有長幀。早期CARLA同開p95 39.29ms，無法從此證明GPU競爭因果。render約60FPS、5Hz模擬時間回放與秒級模型更新必須分開。\n',
        '實際桌機與375px手機頁已檢視，八鏡頭、seek／完整播放、3D拖曳旋轉及reset、路線／錯誤流程有收據。手機無橫向溢出但字偏小；真正200% browser zoom未成功驗證，640px窄窗不能代替。即時標頭重複文案尚留。首版路線圖因等比窄軸壓成細條，另出v2統一米制範圍並實看六頁contact；原圖不覆寫。\n',
        f"本輪RGB實際儲存{rgb_bytes:,}bytes、GT＋pred語意PNG {writes:,}bytes。這是保留證據成本，未做無限run輪替／長時間磁碟壓測。30次粗粒度共用GPU抽樣不能當每模型記憶體；67幀renderer geometry min/max/final={validation['browser_renderer_geometry_min']}/{validation['browser_renderer_geometry_max']}/{validation['browser_renderer_geometry_final']}，texture max={validation['browser_renderer_texture_max']}，短案無單調增長不代表已通過長期洩漏測試。\n",
        '真loopback warm傳輸：state63,971bytes p50 5.047／p95 28.869ms；frontPNG416,021bytes p50 30.911／p95 34.106ms。27項真HTTP包含來源、錯誤／路徑／Host／Origin與純回放限制，通過不代表公開網路安全或延遲。\n',
        '### 修復、故障與回歸\n',
        '修復範圍詳問題T10-11至23：finite/time gate、曲路障礙幾何、紅燈stall、Windows有界atomic重試、GT actor bbox／部分setup清理、control-before-GT紀錄與timing finally、實線立即停止、RGB run/frame一致性、preview競態／錯誤可見性、Three契約、OrbitControls及來源依賴閉包。永久檔案鎖仍要明確失敗，不無限重試。舊core／bev_fusion／權重與Phase0–9證據未改。早期六案及setup-fault的contract只列三權重，cuboid SHA由當時凍結的live_perception字面常數綁定；後續17案contract明列四權重及載入依賴閉包，早期來源範圍較窄的限制保留，沒有事後補造當時快照。\n',
        '真相機sensor.stop後15秒timeout、有效性故障、WebGL失效、setup建立8GT sensor後例外都分別驗證停止／清理。實線stop已加防護與回歸，未故意實車越線。關頁fetch不可靠，原12秒lease實測13.51秒才出report；加入beacon fallback並改3秒lease後，真移動watchdog4.391秒、真關頁4.402秒完成停止且cleanup速度0。未單獨觀察到beacon即送；仍是推論週期加收尾的有界延遲，背景節流也可能觸發保守停止。\n',
        '完整Python322項、0fail／0skip（27.726s），最終3秒lease修改後重跑service與Phase10 service小組全過；JS52項、0fail；27真HTTP；原1123幀Three回歸及新2194幀全部過。Windows沙箱loopback10013與正常host測試分開，初次harness／過嚴契約／錯誤測試期待的失敗log保留，沒有把工具問題當模型成敗。每run以自身凍結source重算planner/controller exact，所有RGB／GT hash核對；沒有借用最新程式重算假稱舊run一致。\n',
        '### 展示、封存與恢復\n',
        '本機閉環仍用 `./Start-NavigationDemo.ps1 -StartSimulator`，`http://127.0.0.1:8768/`自訂Town01道路起終點／預覽／啟動；任意有效路線不保證抵達。`-ReplayOnly`不載GPU或CARLA。詳[DEPLOYMENT.md](DEPLOYMENT.md)。\n',
        '新增16:9實際瀏覽器錄製回放 `phase10-focus-replay.webm`，1280×720、67幀、5FPS、13.4秒、2,483,330bytes，實際browser解碼播放，明標recorded/not live與地圖先驗。影片有損編碼不作準確率輸入；保存原PNG／JSON與SHA。可攜ZIP `phase10-portable-replay-v1-001.zip` 139,595,970bytes，609項manifest＋自身manifest共610檔；含Phase89真行人64幀／512RGB，8770新資料夾實測、409拒絕起車且未import Torch/CARLA。新版加入已實測Orbit旋轉；舊Phase89 ZIP不可旋轉的原文宣稱在部署文件更正，舊ZIP不改。\n',
        'source-snapshot、所有新run與失敗／GT labels／pred masks／exact light inputs／browser收據／原log／錯例圖／影片及post-run input manifest封存於evidence。完整17552RGB、四權重與133.1MiB可攜ZIP保留本機ignored artifacts，不上傳／不公開。每ZIP CRC及逐項SHA核對；manifest證明bytes，不證明模型品質或完整真值。\n',
        '交付時CARLA world為async，vehicles／walkers／sensors均0，之後停止server；本輪8768／8769／8770／8771與2000無listener、沒有Phase10 worker，瀏覽器測試tabs關閉。下一輪從問題清單的新獨立矩陣開始，不自動重開Phase3、Wall或depth。\n',
        '### 完整性與未測項目\n',
        '| 領域 | 已做 | 明示缺口 |\n|---|---|---|']
    for item in matrix:
        lines.append(f"| {item['area']} | {item['result']} | {item['coverage_gap']} |")
    lines += ['\n不能用有限測試保證找出所有問題。後續優先順序是號誌關聯／曲路道路證據、car/truck與3D target、端到端效能、尾端預測／覆蓋及長期穩定性；任何新訓練先固定獨立資料和門檻。\n']
    path=ROOT/'PHASE10.md'
    plan=path.read_text(encoding='utf8')
    assert '## 最終結果' not in plan
    path.write_text(plan+'\n'.join(lines),encoding='utf8')
    print({'status':validation['status'],'attempts':24,'frames':2194,'input_files':len(inputs),
        'RGB_bytes':rgb_bytes,'GT_pred_PNG_bytes':writes,'models_unchanged':len(models)})


if __name__ == '__main__':
    main()
