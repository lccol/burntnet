import torch

from torch import nn
from kornia import morphology as morph

from einops import rearrange
from typing import Callable, List, Tuple, Dict, Callable, Optional, Literal, Union

class QuadraticBaseMorpho(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        morpho_op: Literal['dilation', 'erosion'],
        engine: Literal['unfold', 'convolution']='unfold'
    ) -> None:
        super().__init__()
        assert morpho_op in {'dilation', 'erosion'}, f'Invalid morphological operation specified {morpho_op}'
        assert kernel_size % 2 == 1
        
        self.kernel_size = kernel_size
        self.morpho_op = morpho_op
        self.engine = engine
        
        if morpho_op == 'dilation':
            self.morpho_fn = morph.dilation
        elif morpho_op == 'erosion':
            self.morpho_fn = morph.erosion
        else:
            raise ValueError(f'Invalid morphological operator {morpho_op}')
        
        self.build_params()
        return
    
    def build_params(self) -> None:
        # quadratic structuring function
        # f(x, y) = ax**2 + 2bxy + cy**2
        self.k1 = nn.Parameter(torch.rand(1))
        self.k2 = nn.Parameter(torch.rand(2))
        self.k3 = nn.Parameter(torch.rand(3))
        return
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        struct_elem = self.quadratic_structuring_element(device=x.device)
        kernel = torch.ones(self.kernel_size, self.kernel_size, requires_grad=False)
        kernel[self.kernel_size // 2, self.kernel_size // 2] = 0
        
        out = self.morpho_fn(
            x,
            kernel,
            struct_elem,
            engine=self.engine
        )
        
        return out
    
    def quadratic_structuring_element(
        self,
        device: Optional[str | torch.device]=None
    ) -> torch.Tensor:
        if device is None:
            device = 'cpu'
        with torch.no_grad():
            c = torch.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1, device=device)
            coords = torch.meshgrid(
                c, c, indexing='xy'
            )
        
            x = coords[0].float()
            y = coords[1].float()
            
        structuring_element = self.k1 * x ** 2 + 2 * self.k2 * x * y + self.k3 * y ** 2
        structuring_element /= structuring_element.max()
        return -structuring_element
    
    
class QuadraticMorpho(QuadraticBaseMorpho):
    def __init__(
        self,
        kernel_size: int,
        morpho_op: Literal['dilation', 'erosion'],
        scale: Optional[float]=None,
        engine: Literal['unfold', 'convolution']='unfold'
    ) -> None:
        super().__init__(
            kernel_size,
            morpho_op,
            engine
        )
        self.scale = scale
        return
    
    def build_params(self):
        self.k1 = nn.Parameter(torch.rand(1))
        self.k2 = nn.Parameter(torch.rand(1))
        return
    
    def quadratic_structuring_element(
        self,
        device: Optional[str | torch.device]=None
    ) -> torch.Tensor:
        if device is None:
            device = 'cpu'
        with torch.no_grad():
            c = torch.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1, device=device)
            coords = torch.meshgrid(
                c, c, indexing='xy'
            )
        
            x = coords[0].float()
            y = coords[1].float()
            
        structuring_element = self.k1 * x ** 2 + self.k2 * y ** 2
        structuring_element = structuring_element / structuring_element.max()
        if self.scale:
            structuring_element = structuring_element / 2 * self.scale
        return -structuring_element
    
class RMBlock(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        f1: int,
        f2: int,
        f3: int,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = get_padding(kernel_size)
        self.in_channels = in_channels
        # output channels
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        # residual connection output channels
        self.f_res = f1 + f2 + f3
        
        self.conv1 = nn.Conv2d(
            in_channels, 
            self.f1,
            kernel_size, 
            padding=self.padding
        )
        self.conv2 = nn.Conv2d(
            self.f1, 
            self.f2,
            kernel_size,
            padding=self.padding
        )
        self.conv3 = nn.Conv2d(
            self.f2,
            self.f3,
            kernel_size,
            padding=self.padding
        )
        
        self.conv_res = nn.Conv2d(
            in_channels,
            self.f_res,
            kernel_size,
            padding=self.padding
        )
        
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        # concat along the channel axis
        x_all = torch.cat((x1, x2, x3), dim=1)
        x_res = self.conv_res(x)
        
        res = x_all + x_res
        return res
    
class MRMBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        rm_kernel_size: int,
        f1: int,
        f2: int, 
        f3: int,
        engine: Literal['unfold', 'convolution']='unfold'
    ) -> None:
        super().__init__()
        self.in_channels = in_channels,
        self.rm_kernel_size = rm_kernel_size
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.out_channels = f1 + f2 + f3
        self.engine = engine
        
        self.rm1 = RMBlock(
            rm_kernel_size,
            in_channels,
            f1, f2, f3
        )
        self.rm2 = RMBlock(
            rm_kernel_size,
            self.out_channels,
            f1, f2, f3
        )
        self.erosion = QuadraticMorpho(
            kernel_size=5,
            morpho_op='erosion',
            engine=engine
        )
        self.dilation = QuadraticMorpho(
            kernel_size=5,
            morpho_op='dilation',
            engine=engine
        )
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rm1(x)
        e = self.erosion(x)
        d = self.dilation(x)
        
        ed = d - e
        
        residual = self.rm2(x)
        res = ed + residual
        return res
    
class SeparableConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = get_padding(kernel_size)
        
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=self.padding
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
    
class MPMRM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        f1: int,
        f2: int,
        f3: int,
        engine: Literal['unfold', 'convolution']='unfold'
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f = f1 + f2 + f3
        self.engine = engine
        
        self.mrm = MRMBlock(
            in_channels, 
            rm_kernel_size=3,
            f1=f1,
            f2=f2,
            f3=f3,
            engine=engine
        )
        self.conv = SeparableConvolution(
            self.f,
            out_channels,
            kernel_size=3
        )
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cut the input image into 4 patches
        # B x C x H x W
        assert x.size(2) % 2 == 0, f'H dimension {x.size(2)} % 2 must be 0'
        assert x.size(3) % 2 == 0, f'W dimension {x.size(3)} % 2 must be 0'
        x = rearrange(x, 'b c (nh h) (nw w) -> (b nh nw) c h w', nh=2, nw=2)
        
        x = self.mrm(x)
        x = rearrange(x, '(b nh nw) c h w -> b c (nh h) (nw w)', nh=2, nw=2)
        x = self.conv(x)
        return x
    
def get_padding(kernel_size):
    return (kernel_size - 1) // 2 
