import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
from types import SimpleNamespace
import zipfile
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from install_additional_maps import content_path,same_streams
import install_additional_maps as installer


class AdditionalMapInstallTests(unittest.TestCase):
    def test_map_assets_allowed(self):
        self.assertEqual(content_path('CarlaUE4/Content/Carla/Maps/Town07.umap'),Path('CarlaUE4/Content/Carla/Maps/Town07.umap'))

    def test_escape_executable_and_windows_aliases_rejected(self):
        for name in ['../escape','C:/CarlaUE4/Content/x','/CarlaUE4/Content/x',
            'CarlaUE4/Content/../x','CarlaUE4/Content//x','CarlaUE4/Content/aux.uasset',
            'CarlaUE4/Content/file.','CarlaUE4/Content/x.exe','CarlaUE4/Binaries/x.dll',
            'CarlaUE4\\Content\\x','CarlaUE4/Content/name:stream']:
            with self.subTest(name=name),self.assertRaises(ValueError):content_path(name)

    def test_existing_asset_comparison_is_byte_exact(self):
        self.assertTrue(same_streams(io.BytesIO(b'a'*100),io.BytesIO(b'a'*100)))
        self.assertFalse(same_streams(io.BytesIO(b'a'*100),io.BytesIO(b'a'*99+b'b')))
        self.assertFalse(same_streams(io.BytesIO(b'a'),io.BytesIO(b'aa')))

    def exercise_install(self,mismatch,interrupted=False):
        test_parent=(Path(__file__).resolve().parents[1]/'artifacts/runs').resolve()
        test_parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_parent,prefix='additional-map-test-') as folder:
            root=Path(folder).resolve()
            self.assertTrue(root.is_relative_to(test_parent))  # Before TemporaryDirectory's recursive cleanup.
            destination=root/'artifacts/carla';destination.mkdir(parents=True)
            (destination/'CarlaUE4.exe').write_bytes(b'launcher-preserved')
            binary=destination/'CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe';binary.parent.mkdir(parents=True);binary.write_bytes(b'patched-binary-preserved')
            old=destination/'CarlaUE4/Content/shared.uasset';old.parent.mkdir(parents=True);old.write_bytes(b'existing')
            archive=root/'addon.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('CarlaUE4/Content/Carla/Maps/Town06.umap',b'town6')
                z.writestr('CarlaUE4/Content/Carla/Maps/Town07.umap',b'town7')
                z.writestr('CarlaUE4/Content/shared.uasset',b'different' if mismatch else b'existing')
            output=root/'artifacts/runs/install-test';args=SimpleNamespace(archive=archive,output=output,destination=destination,install=True)
            with patch.object(installer,'ROOT',root),patch.object(installer,'EXPECTED_SIZE',archive.stat().st_size):
                if mismatch:
                    with self.assertRaisesRegex(ValueError,'Existing asset'):installer.main(args)
                elif interrupted:
                    rename=installer.os.rename;calls=[]
                    def interrupt_second_publication(*values):
                        calls.append(values)
                        if len(calls)==2:raise InterruptedError('Simulated interruption before atomic publication')
                        return rename(*values)
                    with patch.object(installer.os,'rename',side_effect=interrupt_second_publication):
                        with self.assertRaises(InterruptedError):installer.main(args)
                    self.assertTrue((destination/'CarlaUE4/Content/Carla/Maps/Town06.umap').exists())
                    self.assertFalse((destination/'CarlaUE4/Content/Carla/Maps/Town07.umap').exists())
                    self.assertEqual(json.loads((output/'report.json').read_text())['status'],'failed')
                    output=root/'artifacts/runs/install-recovered';args.output=output
                    installer.main(args)
                else:installer.main(args)
            self.assertEqual(old.read_bytes(),b'existing');self.assertEqual(binary.read_bytes(),b'patched-binary-preserved')
            report=json.loads((output/'report.json').read_text())
            if mismatch:
                self.assertEqual(report['status'],'failed')
                self.assertFalse((destination/'CarlaUE4/Content/Carla/Maps/Town06.umap').exists())
                self.assertFalse((output/'staged').exists())
            else:
                self.assertEqual(report['new_files'],1 if interrupted else 2)
                self.assertEqual(report['existing_exact_files'],2 if interrupted else 1)
                self.assertEqual((destination/'CarlaUE4/Content/Carla/Maps/Town07.umap').read_bytes(),b'town7')

    def test_existing_different_asset_aborts_before_any_install(self):self.exercise_install(True)

    def test_new_assets_installed_and_old_binary_preserved(self):self.exercise_install(False)

    @unittest.skipUnless(sys.platform=='win32','Windows non-overwriting rename interruption path')
    def test_interrupted_publication_recovers_in_new_receipt_run(self):self.exercise_install(False,True)


if __name__=='__main__':unittest.main()
