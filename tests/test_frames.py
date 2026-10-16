from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from senior_ai.frames import RgbFrame


class OwnedFrameTests(unittest.TestCase):
    def test_native_payload_is_copied_before_callback_returns(self):
        payload = bytearray([0, 0, 255, 255, 0, 255, 0, 255])
        native = SimpleNamespace(frame=10, timestamp=0.5, width=2, height=1, raw_data=payload)
        frame = RgbFrame.from_carla(native)
        payload[:] = bytes(8)
        self.assertEqual(frame.raw_data, bytes([0, 0, 255, 255, 0, 255, 0, 255]))

    def test_png_encodes_actual_pixel_order_and_valid_crc(self):
        frame = RgbFrame(10, 0.5, 2, 1, bytes([0, 0, 255, 255, 0, 255, 0, 255]))
        with patch.object(Path, 'write_bytes') as write:
            frame.save_to_disk('test.png')
        png = write.call_args.args[0]
        self.assertEqual(png[:8], b'\x89PNG\r\n\x1a\n')
        offset, chunks = 8, {}
        while offset < len(png):
            size = struct.unpack('>I', png[offset:offset + 4])[0]
            kind, payload = png[offset + 4:offset + 8], png[offset + 8:offset + 8 + size]
            crc = struct.unpack('>I', png[offset + 8 + size:offset + 12 + size])[0]
            self.assertEqual(crc, zlib.crc32(kind + payload))
            chunks[kind] = payload
            offset += size + 12
        self.assertEqual(struct.unpack('>II', chunks[b'IHDR'][:8]), (2, 1))
        self.assertEqual(zlib.decompress(chunks[b'IDAT']), b'\0\xff\0\0\0\xff\0')

    def test_malformed_sensor_payload_is_rejected(self):
        for values in [(0, 0.0, 1, 1, b''), (0, float('nan'), 1, 1, bytes(4)),
                       (0, 0.0, 0, 1, b'')]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                RgbFrame(*values)

    def test_capture_pose_is_owned_and_validated_with_pixels(self):
        matrix = [[float(i == j) for j in range(4)] for i in range(4)]
        native = SimpleNamespace(frame=1, timestamp=.05, width=1, height=1, raw_data=bytes(4),
                                 transform=SimpleNamespace(get_matrix=lambda: matrix))
        owned = RgbFrame.from_carla(native)
        matrix[0][3] = 100
        self.assertEqual(owned.world_from_camera[0][3], 0)
        matrix[1][1] = float('nan')
        with self.assertRaises(ValueError):
            RgbFrame.from_carla(native)
