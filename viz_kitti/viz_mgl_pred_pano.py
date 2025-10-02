import sys
sys.path.insert(0, "/hkfs/work/workspace/scratch/tj3409-rdaionet/OrienterNet")

from pathlib import Path
import os
import torch
import yaml
from torchmetrics import MetricCollection
from omegaconf import OmegaConf as OC
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pytorch_lightning import seed_everything

import maploc
from maploc.data import MapillaryPanoDataModule, MapillaryDataModule
from maploc.data.torch import unbatch_to_device, collate
from maploc.module import GenericModule
from maploc.models.metrics_v1gt_60deg import Location2DError, AngleError
from maploc.evaluation.run import resolve_checkpoint_path
from maploc.evaluation.viz import plot_example_pano, plot_example_views

from maploc.models.voting import argmax_xyr, fuse_gps
from maploc.osm.viz import Colormap, plot_nodes
from maploc.utils.viz_2d import plot_images, features_to_RGB, save_plot, add_text
from maploc.utils.viz_localization import likelihood_overlay, plot_pose, plot_dense_rotations, add_circle_inset

torch.set_grad_enabled(False)
plt.rcParams.update({'figure.max_open_warning': 0})

conf = OC.load(Path(maploc.__file__).parent / 'conf/data/mapillary_washington60_paired.yaml')
conf = OC.merge(conf, OC.create(yaml.full_load("""
data_dir: "datasets/MGL_final"
loading:
    val: {batch_size: 6, num_workers: 0} # Batch size should be num_views multiples for panoramas
    train: ${.val}
add_map_mask: true
return_gps: true
""")))
OC.resolve(conf)
dataset_module = MapillaryPanoDataModule(conf)
# dataset_module = MapillaryDataModule(conf)
dataset_module.prepare_data()
dataset_module.setup()

# experiment = "orienternet_mgl.ckpt"
experiment = "OrienterNet_MGL_ft_washington60_paired" 
# experiment = "experiment_name/checkpoint-step=N.ckpt"  # a given checkpoint
path = resolve_checkpoint_path(experiment)
print(path)
cfg = {'model': {"x_max": 18,"num_rotations": 256, "apply_map_prior": True}}
model = GenericModule.load_from_checkpoint(
    path, strict=True, find_best=not experiment.endswith('.ckpt'), cfg=cfg)
model = model.eval().cuda()
assert model.cfg.data.resize_image == dataset_module.cfg.resize_image

out_dir = Path('viz_kitti/washinton60_paired')
if out_dir is not None:
    os.makedirs(out_dir, exist_ok=True)

seed_everything(25) # best = 25

# --- START: MODIFIED PANORAMA-AWARE VISUALIZATION LOOP ---

# --- FIX: Correct way to get the validation dataset ---
val_dataset = dataset_module.dataset("val")
# --- END FIX ---

num_panoramas_to_viz = 1
num_panoramas_total = len(val_dataset) // 6

# Setup metrics
metrics = MetricCollection(model.model.metrics()).to(model.device)
metrics["xy_gps_error"] = Location2DError("uv_gps", model.cfg.model.pixel_per_meter)

for i in range(num_panoramas_to_viz):
    pano_idx = i
    start_idx = pano_idx * 6
    
    if start_idx + 5 >= len(val_dataset):
        print(f"Not enough images left for a full panorama at index {start_idx}. Stopping.")
        break
        
    # 1. Manually create a batch of 3 for one panorama
    batch_list = [
        val_dataset[start_idx], 
        val_dataset[start_idx+1],
        val_dataset[start_idx+2],
        val_dataset[start_idx+3],
        val_dataset[start_idx+4],
        val_dataset[start_idx+5]]
    batch = collate(batch_list)
    
    # 2. Transfer to device and get prediction
    batch_ = model.transfer_batch_to_device(batch, model.device, i)
    pred = model(batch_)
    pred = {k:v.float() if isinstance(v, torch.HalfTensor) else v for k,v in pred.items()}
    
    # 3. Ground truth data is from the main view (view 1)
    gt_data = batch_list[0]
    gt_data_v1 = batch_list[0]
    gt_data_v2 = batch_list[2]
    gt_data_v3 = batch_list[4]
    pred["uv_gps"] = gt_data["uv_gps"][None] # Add batch dim

    # 4. Calculate metrics. Our metrics module is already panorama-aware.
    results = metrics(pred, batch_)
    results.pop("xy_expectation_error", None)
    for k in list(results):
        if "recall" in k:
            results.pop(k, None)
    
    loss = model.model.loss(pred, batch_)
    print(f'Pano {i} | Loss: {loss["total"].item():.2f} | Metrics: ', {k: round(v.item(), 2) for k, v in results.items()})

    # 5. Unbatch for visualization.
    # pred is already B=1, so unbatch extracts the single panorama prediction.
    # We pass the ground truth for the main view.
    pred_unbatched = unbatch_to_device(pred)
    '''
    data_unbatched_v1 = unbatch_to_device(gt_data_v1)
    data_unbatched_v2 = unbatch_to_device(gt_data_v2)
    data_unbatched_v3 = unbatch_to_device(gt_data_v3)
    data_unbatched = [data_unbatched_v1, data_unbatched_v2, data_unbatched_v3]
    '''
    data_unbatched = unbatch_to_device(gt_data)
    # 6. Call the visualization function
    plot_example_pano(i, model, pred_unbatched, data_unbatched, results, plot_bev=True, out_dir=out_dir, show_gps=True)
# --- END: MODIFIED PANORAMA-AWARE VISUALIZATION LOOP ---
