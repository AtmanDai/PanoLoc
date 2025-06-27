from .kitti.dataset import (
    KittiMixDataModule,
    KittiContrastiveDataModule,
    KittiDataModule,
    KittiRainyDataModule, 
    KittiNightDataModule, 
    KittiFoggyDataModule, 
    KittiCloudyDataModule,
    KittiMotionBlurDataModule,
    KittiUnderExposureDataModule,
    KittiOverExposureDataModule,
)
    

from .mapillary.dataset import MapillaryDataModule

modules = {
    "mapillary": MapillaryDataModule, 
    "kitti": KittiDataModule,
    "kitti_mix": KittiMixDataModule,
    "kitti_contrastive": KittiContrastiveDataModule, 
    "kitti_rainy": KittiRainyDataModule, 
    "kitti_night": KittiNightDataModule,
    "kitti_cloudy": KittiCloudyDataModule,
    "kitti_foggy": KittiFoggyDataModule,
    "kitti_over": KittiOverExposureDataModule,
    "kitti_under": KittiUnderExposureDataModule,
    "kitti_motion_blur": KittiMotionBlurDataModule,
    }
