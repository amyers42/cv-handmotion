import torch
import torch.nn as nn
import numpy as np

CARDINALITY_ITEM = 16
INPUT_CHANNELS   = 3         # RGB
NUM_CLASSES      = 7         # hand‑pose classes

class ResidualUnit(nn.Module):
    def __init__(self, l, w, ar, bot_mul=1):
        super().__init__()
        bot_channels = int(round(l * bot_mul))
        self.bn1  = nn.BatchNorm2d(l)
        self.bn2  = nn.BatchNorm2d(l)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.C = bot_channels // CARDINALITY_ITEM   # groups
        pad    = (w - 1) * ar // 2
        k      = (w, w)
        d      = (ar, ar)
        p      = (pad, pad)

        self.conv1 = nn.Conv2d(l, l, k, dilation=d, padding=p, groups=self.C, bias=False)
        self.conv2 = nn.Conv2d(l, l, k, dilation=d, padding=p, groups=self.C, bias=False)

    def forward(self, x, skip):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        return x + y, skip

class Skip(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.conv = nn.Conv2d(l, l, kernel_size=1, bias=False)

    def forward(self, x, skip):
        return x, self.conv(x) + skip

class HandyNet(nn.Module):
    def __init__(self,
                 L = 64,
                 W = np.array([11]*8 + [21]*4 + [41]*4),
                 AR = np.array([1]*4 + [4]*4 + [10]*4 + [25]*4)):
        super().__init__()

        self.stem  = nn.Conv2d(INPUT_CHANNELS, L, kernel_size=1, bias=False)
        self.skip1 = Skip(L)

        blocks = []
        for i, (w, r) in enumerate(zip(W, AR)):
            blocks.append(ResidualUnit(L, w, r))
            if (i + 1) % 4 == 0:
                blocks.append(Skip(L))
        if (len(W) + 1) % 4 != 0:              # ensure final skip
            blocks.append(Skip(L))
        self.residual_blocks = nn.ModuleList(blocks)

        self.last_conv  = nn.Conv2d(L, L, kernel_size=1, bias=False)
        self.global_avg = nn.AdaptiveAvgPool2d(1)   # → (N, L, 1, 1)
        self.classifier = nn.Linear(L, NUM_CLASSES)

    def forward(self, x):
        x, skip = self.skip1(self.stem(x), torch.zeros_like(x))   # init skip
        for m in self.residual_blocks:
            x, skip = m(x, skip)
        x = self.last_conv(skip)
        x = self.global_avg(x).flatten(1)      # (N, L)
        return self.classifier(x)              # raw logits for CrossEntropyLoss