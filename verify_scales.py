import torch
from sam2.build_sam import build_sam2

def verify_encoder_scales():
    """
    验证 SAM2 图像编码器 FPN 输出的真实降采样倍数。
    """
    print("正在加载 SAM2 模型以进行验证...")
    # 确保这里的路径是正确的
    sam2_cfg_file = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_ckpt_path = "/hkfs/work/workspace/scratch/tj3409-rdaionet/sam2/checkpoints/sam2.1_hiera_large.pt"
    
    # 加载 SAM2 模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam2_model = build_sam2(sam2_cfg_file, sam2_ckpt_path, device=device)
    image_encoder = sam2_model.image_encoder.eval()

    # 创建一个已知尺寸的虚拟输入图像
    # SAM 模型通常使用 1024x1024 的输入尺寸
    input_height, input_width = 1024, 1024
    dummy_image = torch.randn(1, 3, input_height, input_width).to(device)
    print(f"输入图像尺寸: ({input_height}, {input_width})")
    print("-" * 30)

    # 执行前向传播
    with torch.no_grad():
        # 获取 FPN 的输出特征
        fpn_features = image_encoder(dummy_image)['backbone_fpn']

    # 计算并打印每一层的降采样倍数
    calculated_scales = []
    print("FPN 输出特征图尺寸及计算出的降采样倍数:")
    for i, feature_map in enumerate(fpn_features):
        b, c, h, w = feature_map.shape
        # 降采样倍数 = 输入尺寸 / 特征图尺寸
        scale_h = input_height / h
        scale_w = input_width / w
        
        # 正常情况下，高和宽的缩放比例应该是一样的
        if scale_h != scale_w:
            print(f"警告: 第 {i} 层的 H/W 降采样倍数不一致！")
        
        print(f"层级 {i}: HxW = ({h}, {w})  |  计算出的降采样倍数 = {scale_h}")
        calculated_scales.append(int(scale_h))
        
    print("-" * 30)
    print(f"最终计算出的 scales 列表为: {calculated_scales}")
    print("请使用这个列表更新你 hybrid_model.py 中的 self.image_encoder.scales。")
    
    return calculated_scales

if __name__ == '__main__':
    verify_encoder_scales()