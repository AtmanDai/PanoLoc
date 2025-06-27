import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

from .base import BaseModel

logger = logging.getLogger(__name__)

class GeMPooling(nn.Module):
    """Improved Generalized Mean Pooling with better initialization."""
    def __init__(self, p=3.0, eps=1e-6, learn_p=True):
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer('p', torch.tensor(p))
        self.eps = eps

    def forward(self, x):
        # More numerically stable implementation
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

class ImprovedProjectionHead(nn.Module):
    """Enhanced projection head for contrastive learning."""
    
    def __init__(self, in_channels, out_dim, hidden_dim=None, 
                 num_layers=2, dropout=0.1, use_bn=True, 
                 pooling_type='gem', gem_p=3.0):
        super().__init__()
        
        # Pooling layer
        if pooling_type == 'gem':
            self.pooling = GeMPooling(p=gem_p, learn_p=True)
        elif pooling_type == 'adaptive_avg':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pooling_type == 'adaptive_max':
            self.pooling = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
        self.flatten = nn.Flatten()
        
        # Determine hidden dimension
        if hidden_dim is None:
            hidden_dim = max(out_dim, in_channels // 2)
        
        # Build projection layers
        layers = []
        current_dim = in_channels
        
        for i in range(num_layers):
            # Linear layer
            next_dim = hidden_dim if i < num_layers - 1 else out_dim
            layers.append(nn.Linear(current_dim, next_dim))
            
            # Add normalization and activation for all but last layer
            if i < num_layers - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(next_dim))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            current_dim = next_dim
        
        self.projection = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: [B, C, H, W]
        x = self.pooling(x)      # [B, C, 1, 1]
        x = self.flatten(x)      # [B, C]
        x = self.projection(x)   # [B, out_dim]
        
        # L2 normalize for contrastive learning
        x = F.normalize(x, p=2, dim=1)
        return x

class FixedProjectionHead(nn.Module):
    """Fixed projection head that prevents embedding collapse."""
    
    def __init__(self, in_channels, out_dim, 
                 hidden_dim=None, dropout=0.1, 
                 use_final_norm=True,
                 gem_p=3.0):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = max(out_dim * 2, in_channels // 2)
        
        # Pooling layer
        self.pooling = GeMPooling(p=gem_p, learn_p=True)
        self.flatten = nn.Flatten()
        
        # Projection layers with careful initialization
        self.projection = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.use_final_norm = use_final_norm
        
        # Custom weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Careful weight initialization to prevent collapse."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use smaller initialization to prevent saturation
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: [B, C, H, W]
        x = self.pooling(x)      # [B, C, 1, 1]
        x = self.flatten(x)      # [B, C]
        x = self.projection(x)   # [B, out_dim]
        
        # Optional final normalization
        if self.use_final_norm:
            x = F.normalize(x, p=2, dim=1, eps=1e-8)
        
        return x

# Alternative: Even simpler version for debugging
class SimpleProjectionHead(nn.Module):
    """Ultra-simple projection head for debugging."""
    
    def __init__(self, in_channels, out_dim):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, out_dim),
            # No normalization, no batch norm - keep it simple
        )
        
        # Simple initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.projection(x)

# Alternative: SimCLR-style projection head
class SimCLRProjectionHead(nn.Module):
    """SimCLR-style projection head with ReLU and no final normalization."""
    
    def __init__(self, in_channels, out_dim, hidden_dim=None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = in_channels
        
        self.pooling = GeMPooling(p=3.0, learn_p=True)
        self.flatten = nn.Flatten()
        
        self.projection = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.projection(x)
        # Note: No normalization here - normalization happens in loss computation
        return x

class DecoderBlock(nn.Module):
    def __init__(
        self, previous, out, ksize=3, num_convs=1, norm=nn.BatchNorm2d, padding="zeros"
    ):
        super().__init__()
        layers = []
        for i in range(num_convs):
            conv = nn.Conv2d(
                previous if i == 0 else out,
                out,
                kernel_size=ksize,
                padding=ksize // 2,
                bias=norm is None,
                padding_mode=padding,
            )
            layers.append(conv)
            if norm is not None:
                layers.append(norm(out))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, previous, skip):
        _, _, hp, wp = previous.shape
        _, _, hs, ws = skip.shape
        scale = 2 ** np.round(np.log2(np.array([hs / hp, ws / wp])))
        upsampled = nn.functional.interpolate(
            previous, scale_factor=scale.tolist(), mode="bilinear", align_corners=False
        )
        # If the shape of the input map `skip` is not a multiple of 2,
        # it will not match the shape of the upsampled map `upsampled`.
        # If the downsampling uses ceil_mode=False, we nedd to crop `skip`.
        # If it uses ceil_mode=True (not supported here), we should pad it.
        _, _, hu, wu = upsampled.shape
        _, _, hs, ws = skip.shape
        if (hu <= hs) and (wu <= ws):
            skip = skip[:, :, :hu, :wu]
        elif (hu >= hs) and (wu >= ws):
            skip = nn.functional.pad(skip, [0, wu - ws, 0, hu - hs])
        else:
            raise ValueError(
                f"Inconsistent skip vs upsampled shapes: {(hs, ws)}, {(hu, wu)}"
            )

        return self.layers(skip) + upsampled


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, **kw):
        super().__init__()
        self.first = nn.Conv2d(
            in_channels_list[-1], out_channels, 1, padding=0, bias=True
        )
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(c, out_channels, ksize=1, **kw)
                for c in in_channels_list[::-1][1:]
            ]
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, layers):
        feats = None
        for idx, x in enumerate(reversed(layers.values())):
            if feats is None:
                feats = self.first(x)
            else:
                feats = self.blocks[idx - 1](feats, x)
        out = self.out(feats)
        return out


def remove_conv_stride(conv):
    conv_new = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        bias=conv.bias is not None,
        stride=1,
        padding=conv.padding,
    )
    conv_new.weight = conv.weight
    conv_new.bias = conv.bias
    return conv_new

def create_projection_head(in_channels, global_descriptor_conf):
    """Factory function to create projection head based on config."""
    
    head_type = global_descriptor_conf.get('type', 'improved')
    out_dim = global_descriptor_conf.get('dim', 128)
    
    if head_type == 'improved':
        return ImprovedProjectionHead(
            in_channels=in_channels,
            out_dim=out_dim,
            hidden_dim=global_descriptor_conf.get('hidden_dim'),
            num_layers=global_descriptor_conf.get('num_layers', 2),
            dropout=global_descriptor_conf.get('dropout', 0),
            use_bn=global_descriptor_conf.get('use_bn', True),
            pooling_type=global_descriptor_conf.get('pooling', 'gem'),
            gem_p=global_descriptor_conf.get('gem_p', 3.0)
        )
    elif head_type == 'simclr':
        return SimCLRProjectionHead(
            in_channels=in_channels,
            out_dim=out_dim,
            hidden_dim=global_descriptor_conf.get('hidden_dim')
        )
    else:
        # Fallback to base version
        return nn.Sequential(
            GeMPooling(),
            nn.Flatten(),
            nn.Linear(in_channels, out_dim),
            # nn.BatchNorm1d(out_dim),  # Add normalization
            # nn.ReLU(inplace=True),    # Add activation
            # nn.Linear(out_dim, out_dim)  # Add another layer
        )


def create_debug_projection_head(in_channels, global_descriptor_conf, debug_mode=True):
    """Create projection head with debugging features."""
    out_dim = global_descriptor_conf.get('dim', 128)
    if debug_mode:
        print(f"🔧 Creating SIMPLE projection head for debugging")
        print(f"   Input: {in_channels} -> Output: {out_dim}")
        return SimpleProjectionHead(
            in_channels=in_channels,
            out_dim=out_dim,
        )
    else:
        print(f"🔧 Creating FIXED projection head")
        print(f"   Input: {in_channels} -> Output: {out_dim}")
        return FixedProjectionHead(
            in_channels=in_channels,
            out_dim=out_dim,
            hidden_dim=global_descriptor_conf.get('hidden_dim'),
            dropout=global_descriptor_conf.get('dropout', 0),
            gem_p=global_descriptor_conf.get('gem_p', 3.0)
        )

class FeatureExtractor(BaseModel):
    default_conf = {
        "pretrained": True,
        "input_dim": 3,
        "output_dim": 128,  # # of channels in output feature maps
        "encoder": "resnet50",  # torchvision net as string
        "remove_stride_from_first_conv": False,
        "num_downsample": None,  # how many downsample block
        "decoder_norm": "nn.BatchNorm2d",  # normalization ind decoder blocks
        "do_average_pooling": False,
        "checkpointed": False,  # whether to use gradient checkpointing
        "global_descriptor": None, # projection head for contrastive learning
    }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def build_encoder(self, conf):
        assert isinstance(conf.encoder, str)
        if conf.pretrained:
            assert conf.input_dim == 3
        Encoder = getattr(torchvision.models, conf.encoder)

        kw = {}
        if conf.encoder.startswith("resnet"):
            layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
            kw["replace_stride_with_dilation"] = [False, False, False]
        elif conf.encoder == "vgg13":
            layers = [
                "features.3",
                "features.8",
                "features.13",
                "features.18",
                "features.23",
            ]
        elif conf.encoder == "vgg16":
            layers = [
                "features.3",
                "features.8",
                "features.15",
                "features.22",
                "features.29",
            ]
        else:
            raise NotImplementedError(conf.encoder)

        if conf.num_downsample is not None:
            layers = layers[: conf.num_downsample]
        encoder = Encoder(weights="DEFAULT" if conf.pretrained else None, **kw)
        encoder = create_feature_extractor(encoder, return_nodes=layers)
        if conf.encoder.startswith("resnet") and conf.remove_stride_from_first_conv:
            encoder.conv1 = remove_conv_stride(encoder.conv1)

        if conf.do_average_pooling:
            raise NotImplementedError
        if conf.checkpointed:
            raise NotImplementedError

        return encoder, layers

    
    def _init(self, conf):
        # Preprocessing
        self.register_buffer("mean_", torch.tensor(self.mean), persistent=False)
        self.register_buffer("std_", torch.tensor(self.std), persistent=False)

        # Encoder
        self.encoder, self.layers = self.build_encoder(conf)
        s = 128
        inp = torch.zeros(1, 3, s, s)
        features = list(self.encoder(inp).values())
        self.skip_dims = [x.shape[1] for x in features]
        self.layer_strides = [s / f.shape[-1] for f in features]
        self.scales = [self.layer_strides[0]]

        # Decoder
        norm = eval(conf.decoder_norm) if conf.decoder_norm else None  # noqa
        self.decoder = FPN(self.skip_dims, out_channels=conf.output_dim, norm=norm)

        # --- DEBUGGING PRINT STATEMENT ---
        # This will show us exactly what configuration this class receives.
        print("\n---: FeatureExtractor received config ---")
        print(conf)
        print("-------------------------------------------\n")

        # Create the global descriptor (projection head) if configured
        self.global_descriptor_head = None
        global_descriptor_conf = conf.get("global_descriptor")
        if global_descriptor_conf is not None:
            # The input to the head is the output of the encoder's last stage
            in_channels = self.skip_dims[-1]
            self.global_descriptor_head = create_debug_projection_head(
                in_channels=in_channels, global_descriptor_conf=global_descriptor_conf, debug_mode=True
            )

            print(f"Created {global_descriptor_conf.get('type', 'improved')} "
              f"projection head: {in_channels} -> {global_descriptor_conf.get('dim', 128)}")
        else:
            print("DEBUG: global_descriptor config NOT FOUND.")

        logger.debug(
            "Built feature extractor with layers {name:dim:stride}:\n"
            f"{list(zip(self.layers, self.skip_dims, self.layer_strides))}\n"
            f"and output scales {self.scales}."
        )

    def _forward(self, data):
        image = data["image"]
        image = (image - self.mean_[:, None, None]) / self.std_[:, None, None]

        skip_features = self.encoder(image)

        # Prepare the output dictionary
        pred = {}

        # Compute the global descriptor from the last encoder feature map
        if self.global_descriptor_head is not None:
            # Get the last feature map from the encoder's output dictionary
            last_feat = list(skip_features.values())[-1]
            descriptor = self.global_descriptor_head(last_feat)
            pred["global_descriptor"] = descriptor

        output = self.decoder(skip_features)
        pred["feature_maps"] = [output]
        pred["skip_features"] = skip_features
        return pred
