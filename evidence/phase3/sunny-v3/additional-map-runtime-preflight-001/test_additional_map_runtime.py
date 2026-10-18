from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_additional_map_runtime as runtime


class AdditionalRuntimeTests(unittest.TestCase):
    def exercise(self,interrupted=False,refuse_baseline=False):
        parent=(Path(__file__).resolve().parents[1]/'artifacts/runs').resolve();parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent,prefix='isolated-map-test-') as folder:
            root=Path(folder).resolve();self.assertTrue(root.is_relative_to(parent))
            base=root/'artifacts/carla';base.mkdir(parents=True);(base/'CarlaUE4.exe').write_bytes(b'launcher')
            binary=base/'CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe';binary.parent.mkdir(parents=True);binary.write_bytes(b'patched')
            shared=base/'CarlaUE4/Content/shared.uasset';shared.parent.mkdir(parents=True);shared.write_bytes(b'old')
            saved=base/'CarlaUE4/Saved/runtime.log';saved.parent.mkdir(parents=True);saved.write_text('not distribution')
            for name in ['scripts/prepare_additional_map_runtime.py','scripts/install_additional_maps.py','src/senior_ai/core.py','src/senior_ai/sunny_data.py']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture fingerprint')
            archive=root/'addon.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('CarlaUE4/Content/Carla/Maps/Town06.umap',b't6')
                z.writestr('CarlaUE4/Content/Carla/Maps/Town07.umap',b't7')
                z.writestr('CarlaUE4/Content/shared.uasset',b'new-different-shared')
            dest=base if refuse_baseline else root/'artifacts/carla-phase3-expansion-test'
            out=root/'artifacts/runs/receipt';args=SimpleNamespace(archive=archive,destination=dest,output=out,resume=False)
            with patch.object(runtime,'ROOT',root),patch.object(runtime,'EXPECTED_SIZE',archive.stat().st_size):
                if refuse_baseline:
                    with self.assertRaisesRegex(ValueError,'distinct'):runtime.main(args)
                    self.assertFalse(out.exists());return
                if interrupted:
                    copy=runtime.atomic_copy;calls=[]
                    def fail_second(*values):
                        calls.append(1)
                        if len(calls)==2:
                            values[0].close();raise InterruptedError('simulated clone interruption')
                        return copy(*values)
                    with patch.object(runtime,'atomic_copy',side_effect=fail_second):
                        with self.assertRaises(InterruptedError):runtime.main(args)
                    self.assertEqual(json.loads((out/'state.json').read_text())['status'],'failed_preserve_for_exact_contract_resume')
                    args.resume=True
                runtime.main(args)
                self.assertEqual(shared.read_bytes(),b'old');self.assertEqual(binary.read_bytes(),b'patched')
                self.assertEqual((dest/'CarlaUE4/Content/shared.uasset').read_bytes(),b'new-different-shared')
                self.assertFalse((dest/'CarlaUE4/Saved/runtime.log').exists())
                state=json.loads((out/'state.json').read_text());self.assertTrue(state['baseline_unchanged'])
                self.assertEqual(state['replaced_only_in_new_clone'],1)
                args.resume=True;runtime.main(args)
                self.assertEqual((dest/'CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe').read_bytes(),b'patched')

    def test_original_root_cannot_be_overlay_target(self):self.exercise(refuse_baseline=True)
    def test_different_shared_asset_changes_only_isolated_clone(self):self.exercise()
    def test_clone_interruption_and_completed_resume_are_exact(self):self.exercise(interrupted=True)

    def test_engine_plugin_content_allowed_but_code_and_traversal_rejected(self):
        for name in ['Engine/Content/Font.ufont','Engine/Plugins/Runtime/HairStrands/Content/Hair.uexp',
            'CarlaUE4/Plugins/Carla/Content/Tagged.uasset']:
            self.assertEqual(runtime.isolated_asset_path(name),Path(name))
        for name in ['Engine/Binaries/x.dll','Engine/Content/../Binaries/x.dll','CarlaUE4/Content/x.py',
            'Engine/Plugins/Content/x.exe','C:/Engine/Content/x.uasset','Engine/Content/aux.uasset']:
            with self.subTest(name=name),self.assertRaises(ValueError):runtime.isolated_asset_path(name)


if __name__=='__main__':unittest.main()
