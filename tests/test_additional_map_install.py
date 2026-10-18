import io
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from install_additional_maps import content_path,same_streams


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


if __name__=='__main__':unittest.main()
