# ---Feature Extraction and Visualization for OrienterNet---
'''
This file demonstrates how to extract and visualize features from intermediate layers of the OrienterNet model. 
It focuses on:
1. Loading OrienterNet model(orienternet_mgl.ckpt) and KITTI test dataset as sample data
2. Implementing PyTorch hooks to capture feature maps from encoder, decoder, and adaptation layers
3. Visualizing the extracted feature maps using appropriate visualization techniques
'''
import sys
sys.path.insert(0, "/cvhci/temp/ruizedai/onet/OrienterNet")

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from PIL import Image
import torchvision.transforms as transforms
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
from maploc.module import GenericModule
from maploc.evaluation.run import resolve_checkpoint_path

class OrienterNetFeatureExtractor:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        
        # 简化的模型加载
        self.model = load_orienternet_model(checkpoint_path)
        if self.model is None:
            raise ValueError("Failed to load OrienterNet model")
        
        self.model = self.model.to(device)
        self.features = {}
        self.hooks = []
        
        # 动态发现目标层
        self.target_layers = self._discover_target_layers()
        self._register_hooks()
    
    def _discover_target_layers(self):
        """动态发现模型中的关键层"""
        target_layers = {}
        
        print("Discovering model layers...")
        for name, module in self.model.named_modules():
            # Image Encoder 
            if 'image_encoder.encoder.layer4' in name:
                layer_key = "layer4"
                target_layers[f'encoder_{layer_key}'] = name
        
        print(f"Found {len(target_layers)} target layers:")
        for key, value in target_layers.items():
            print(f"  {key}: {value}")
        
        return target_layers
    
    def _get_layer_by_name(self, model, layer_name):
        """通过名称获取层"""
        try:
            layer = model
            for attr in layer_name.split('.'):
                layer = getattr(layer, attr)
            return layer
        except AttributeError:
            return None
    
    def _register_hooks(self):
        """注册hooks"""
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    self.features[name] = output.detach().cpu()
                elif isinstance(output, (list, tuple)):
                    self.features[name] = [o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in output]
                else:
                    self.features[name] = output
                print(f"✓ Captured {name}: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
            return hook
        
        for layer_name, layer_path in self.target_layers.items():
            layer = self._get_layer_by_name(self.model, layer_path)
            if layer is not None:
                hook = layer.register_forward_hook(get_activation(layer_name))
                self.hooks.append(hook)
                print(f"✓ Registered hook for {layer_name}")
            else:
                print(f"✗ Layer {layer_path} not found")
    
    def extract_features(self, image_tensor):
        """基于OrienterNet源代码的正确实现"""
        self.features.clear()
        
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            batch_size = image_tensor.shape[0]
            
            # 创建符合源代码要求的相机对象
            class OrienterNetCamera:
                def __init__(self, K, T_w2cam):
                    self.K = K
                    self.T_w2cam = T_w2cam
                
                def scale(self, factor):
                    # 对应orienternet.py第119行
                    K_new = self.K.clone()
                    K_new[:, :2, :] *= factor
                    return OrienterNetCamera(K_new, self.T_w2cam)
                
                def to(self, device, non_blocking=True):
                    # 对应orienternet.py第120行
                    return OrienterNetCamera(
                        self.K.to(device, non_blocking=non_blocking),
                        self.T_w2cam.to(device, non_blocking=non_blocking)
                    )
                
                def float(self):
                    # 对应orienternet.py第126行
                    return OrienterNetCamera(self.K.float(), self.T_w2cam.float())
            
            try:
                # 使用KITTI标准参数
                K = torch.tensor([[[718.856, 0.0, 607.1928],
                                [0.0, 718.856, 185.2157],
                                [0.0, 0.0, 1.0]]], device=self.device).repeat(batch_size, 1, 1)
                
                T_w2cam = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
                
                batch = {
                    'image': image_tensor,
                    'map': torch.zeros(batch_size, 3, 128, 128, dtype=torch.long, device=self.device),
                    'camera': OrienterNetCamera(K, T_w2cam)
                }
                
                print("Forward pass with OrienterNetCamera...")
                output = self.model(batch)
                print("✅ Complete success!")
                
                return self.features
                
            except Exception as e:
                print(f"Error: {e}")
                # 返回已捕获的特征
                return self.features if self.features else None
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print("✅ All hooks removed")


def load_orienternet_model(checkpoint_path):
    """Load OrienterNet model from checkpoint"""
    try:
        cfg = {'model': {"num_rotations": 360, "apply_map_prior": True}}
        model = GenericModule.load_from_checkpoint(
                checkpoint_path, strict=True, find_best=False, cfg=cfg)
        model = model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"✓ OrienterNet loaded successfully on {device}")
        print(f"✓ Model type: {type(model)}")

        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_sample_image(image_path=None):
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def debug_model_input_requirements(model):
    """调试模型输入要求"""
    print("=== Debugging Model Input Requirements ===")
    
    # 1. 检查forward方法签名
    import inspect
    try:
        sig = inspect.signature(model.forward)
        print(f"Forward method signature: {sig}")
    except Exception as e:
        print(f"Could not inspect signature: {e}")
    
    # 2. 检查模型结构中的线索
    print("\nModel structure analysis:")
    for name, module in model.named_modules():
        if any(keyword in name.lower() for keyword in ['input', 'embed', 'proj']):
            print(f"  {name}: {type(module)}")
            if hasattr(module, 'weight') and module.weight is not None:
                print(f"    Weight shape: {module.weight.shape}")
    
    # 3. 检查模型的配置
    if hasattr(model, 'cfg'):
        print(f"\nModel config: {model.cfg}")
    
    print("=" * 50)

def main():
    """Main function to demonstrate feature extraction and visualization"""
    print("=== OrienterNet Feature Extraction and Visualization ===\n")
    
    # Configuration
    checkpoint_path = resolve_checkpoint_path("orienternet_mgl.ckpt") # Update path as needed
    sample_image_path = "/cvhci/temp/ruizedai/onet/OrienterNet/viz_kitti/feature_visualizations/night/0000000370.png"  # Update with actual KITTI image path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")


    # 1. Create feature extractor
    print("\n1. Setting up feature extractor...")
    extractor = OrienterNetFeatureExtractor(checkpoint_path, device)
    
    # 2. Load sample data
    print("\n2. Loading sample image...")
    image_tensor = load_sample_image(sample_image_path)
    print(f"Image shape: {image_tensor.shape}")

    # 3. Extract features
    print("\n3. Extracting features...")
    features = extractor.extract_features(image_tensor)
    
    if features:
        print(f"\n✓ Successfully extracted features from {len(features)} layers:")
        for layer_name, feature in features.items():
            if isinstance(feature, torch.Tensor):
                print(f"  - {layer_name}: {feature.shape}")
            else:
                print(f"  - {layer_name}: {type(feature)}")
        
        # 4. Visualize features
        print("\n4. Visualizing features...")
        visualizer = FeatureVisualizer()
        
        # Visualize feature maps for each layer
        for layer_name in features.keys():
            if isinstance(features[layer_name], torch.Tensor) and len(features[layer_name].shape) == 4:
                print(f"\nVisualizing {layer_name}...")
                visualizer.visualize_feature_maps(features, layer_name)
                visualizer.visualize_activation_distribution(features, layer_name)
        
        # Visualize overall statistics
        print("\nVisualizing feature statistics...")
        visualizer.visualize_feature_statistics(features)
        
    else:
        print("✗ No features extracted")
    
    # 5. Cleanup
    print("\n5. Cleaning up...")
    extractor.remove_hooks()
    
    print("\n=== Feature extraction and visualization completed ===")

if __name__ == "__main__":
    main()
