"""Small scratch-trained U-Net; RGB tensor is the only forward argument."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, channels_in, channels_out):
        super().__init__(nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels_out), nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels_out), nn.ReLU())


class TinyUNet(nn.Module):
    def __init__(self, width=16, classes=8):
        super().__init__()
        self.enc1, self.enc2 = Block(3, width), Block(width, width * 2)
        self.enc3, self.bottom = Block(width * 2, width * 4), Block(width * 4, width * 8)
        self.dec3, self.dec2 = Block(width * 12, width * 4), Block(width * 6, width * 2)
        self.dec1, self.head = Block(width * 3, width), nn.Conv2d(width, classes, 1)

    def forward(self, rgb):
        a = self.enc1(rgb)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        x = self.bottom(F.max_pool2d(c, 2))
        for skip, block in [(c, self.dec3), (b, self.dec2), (a, self.dec1)]:
            # Nearest upsampling also supports odd input dimensions deterministically.
            x = block(torch.cat((F.interpolate(x, size=skip.shape[-2:], mode='nearest'), skip), 1))
        return self.head(x)


def map_target(raw, config):
    import numpy as np
    raw = np.asarray(raw)
    if raw.dtype != np.uint8 or (raw > 28).any():
        raise ValueError('Expected CARLA uint8 raw class IDs 0..28')
    lut = np.zeros(29, dtype=np.uint8)
    for i, raw_id in enumerate(config['raw_ids']):
        if raw_id is not None:
            lut[raw_id] = i
    lut[config['ignore_raw_ids']] = config['ignore_index']
    return lut[raw]


PALETTE = [(80, 80, 80), (128, 64, 128), (244, 35, 232), (220, 20, 60),
           (0, 0, 142), (0, 0, 230), (119, 11, 32), (255, 255, 0)]


def colorize(ids):
    import numpy as np
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:8] = PALETTE
    return lut[np.asarray(ids, dtype=np.uint8)]
