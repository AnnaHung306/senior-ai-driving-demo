"""Strip GT scene-selection diagnostics from the inference sequence contract."""
import argparse,json,hashlib
from pathlib import Path
def main(args):
    selection=json.loads((args.selection/'selection.json').read_text(encoding='utf-8'))
    target=args.selection/'sequence_inputs.json'
    if target.exists():raise ValueError('Input list already frozen')
    inventory=args.selection/'rgb_only_inventory.json'
    value={'schema_version':1,'scope':'RGB+rig calibration only; selection diagnostics are external and not model inputs',
        'inventory_sha256':hashlib.sha256(inventory.read_bytes()).hexdigest(),
        'sequences':[{'scene_id':x['scene'],'manifests':x['manifests']} for x in selection['selected']]}
    target.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(target)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);main(p.parse_args())
