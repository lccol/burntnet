import torch

from torch import nn
from typing import Union, Dict, List, Optional, Callable, Tuple, Literal

from morpho import MPMRM, MRMBlock

class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int, 
        nblocks: int,
        f1: int,
        f2: int,
        f3: int,
        with_maxpool: bool=True,
        engine: Literal['unfold', 'convolution']='unfold'
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.nblocks = nblocks
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.out_channels = f1 + f2 + f3
        self.engine = engine
        
        acc = []
        inch = in_channels
        for _ in range(nblocks):
            acc.append(
                MPMRM(
                    inch, 
                    self.out_channels,
                    kernel_size,
                    f1, f2, f3,
                    engine
                )
            )
            inch = self.out_channels
        if with_maxpool:
            acc.append(nn.MaxPool2d(2))
        self.enc = nn.Sequential(*acc)
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)
    
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        f1: int,
        f2: int,
        f3: int,
        with_mrm: bool=False,
        with_tconv: bool=True,
        engine: Literal['unfold', 'convolution']='unfold'
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.out_channels = f1 + f2 + f3
        self.engine = engine
        
        acc = []
        inch = in_channels
        if with_tconv:
            # halve the number of channels, double the resolution
            acc.append(
                nn.ConvTranspose2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=2,
                    stride=2
                )
            )
            inch = self.out_channels
        acc.append(
            MPMRM(
                inch,
                self.out_channels,
                kernel_size,
                f1, f2, f3,
                engine
            )
        )
        if with_mrm:
            acc.append(
                MRMBlock(
                    self.out_channels,
                    kernel_size,
                    f1, f2, f3,
                    engine
                )
            )
        
        self.dec = nn.Sequential(*acc)
        return
    
    def forward(
        self,
        x: torch.Tensor,
        skip_x: torch.Tensor
    ) -> torch.Tensor:
        if not skip_x is None:
            x = torch.concat(
                (skip_x, x),
                dim=1
            )
        x = self.dec(x)
        return x

class BurntNet(nn.Module):
    def __init__(
        self,
        features: List[int],
        in_channels: int,
        kernel_size: int,
        nclasses: int,
        engine: Literal['unfold', 'convolution']
    ) -> None:
        super().__init__()
        self.features = features
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        
        enc = []
        inch = in_channels
        for idx, f in enumerate(features):
            if idx == 0 or idx == len(features) - 1:
                nblocks = 1
            else:
                nblocks = 2
            f1, f2, f3 = compute_features(f)
            enc.append(
                EncoderBlock(
                    inch,
                    kernel_size, 
                    nblocks,
                    f1, f2, f3,
                    with_maxpool=idx != len(features) - 1
                )
            )
            inch = f
        self.enc = nn.ModuleList(enc)
        
        dec = []
        reversed_features = list(reversed(features))
        inch = reversed_features[0]
        for idx, f in enumerate(reversed_features):
            next_f = reversed_features[idx + 1] if idx != len(features) - 1 else f
            f1, f2, f3 = compute_features(next_f)
            dec.append(
                DecoderBlock(
                    inch,
                    kernel_size,
                    f1, f2, f3,
                    with_mrm=idx != 0,
                    with_tconv=idx != 0
                )
            )
            inch = f
        self.dec = nn.ModuleList(dec)
        self.conv = nn.Conv2d(
            f, nclasses,
            kernel_size=1
        )
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_features = []
        feat_map = x
        for m in self.enc:
            t = m(feat_map)
            enc_features.append(t)
            feat_map = t
        assert len(enc_features) == len(self.dec)
        out = enc_features[-1]
        for idx, (f, m) in enumerate(zip(reversed(enc_features), self.dec)):
            if idx == 0:
                out = m(out, None)
            else:
                out = m(out, f)
        out = self.conv(out)
        return out
    
def compute_features(f: int) -> Tuple[int, int, int]:
    f1 = f2 = f3 = f // 3
    if f1 + f2 + f3 != f:
        f1 += f - (f1 + f2 + f3)
    assert f1 + f2 + f3 == f
    return f1, f2, f3
