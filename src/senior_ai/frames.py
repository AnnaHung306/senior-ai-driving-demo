"""Owned RGB sensor payloads, independent of CARLA callback/native lifetimes."""
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import zlib


@dataclass(frozen=True)
class RgbFrame:
    frame: int
    timestamp: float
    width: int
    height: int
    raw_data: bytes
    world_from_camera: tuple | None = None

    def __post_init__(self):
        if self.frame < 0 or not math.isfinite(self.timestamp) or self.width <= 0 or self.height <= 0:
            raise ValueError('Invalid RGB metadata')
        object.__setattr__(self, 'raw_data', bytes(self.raw_data))
        if len(self.raw_data) != self.width * self.height * 4:
            raise ValueError('RGB sensor payload must contain width*height BGRA pixels')
        if self.world_from_camera is not None:
            matrix = tuple(tuple(float(x) for x in row) for row in self.world_from_camera)
            if len(matrix) != 4 or any(len(row) != 4 for row in matrix) or not all(math.isfinite(x) for row in matrix for x in row):
                raise ValueError('Invalid captured camera transform')
            object.__setattr__(self, 'world_from_camera', matrix)

    @classmethod
    def from_carla(cls, image):
        matrix = image.transform.get_matrix() if hasattr(image, 'transform') else None
        return cls(image.frame, image.timestamp, image.width, image.height, bytes(image.raw_data), matrix)

    def save_to_disk(self, path):
        """Lossless PNG serialization of the real BGRA sensor data; no synthetic pixels."""
        rgb = bytearray(self.width * self.height * 3)
        rgb[0::3] = self.raw_data[2::4]
        rgb[1::3] = self.raw_data[1::4]
        rgb[2::3] = self.raw_data[0::4]
        stride = self.width * 3
        scanlines = b''.join(b'\0' + rgb[i:i + stride] for i in range(0, len(rgb), stride))
        def chunk(kind, payload):
            return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
        header = struct.pack('>IIBBBBB', self.width, self.height, 8, 2, 0, 0, 0)
        png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b'')
        Path(path).write_bytes(png)
        return str(path)
