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
    

from .mapillary.dataset import (
    MapillaryDataModule,
    MapillaryPanoDataModule,
    Mapillary2KValDataModule,
    MapillaryRainyDataModule,
    MapillarySnowyDataModule,
    MapillaryNightDataModule,
    MapillaryFoggyDataModule,
    MapillaryOverExposureDataModule,
    MapillaryUnderExposureDataModule,
    MapillaryMotionBlurDataModule,
) 

modules = {
    "mapillary": MapillaryDataModule,
    "mapillary_pano": MapillaryPanoDataModule,
    "mapillary_2kval": Mapillary2KValDataModule,
    "mapillary_rainy": MapillaryRainyDataModule,
    "mapillary_snowy": MapillarySnowyDataModule,
    "mapillary_night": MapillaryNightDataModule,
    "mapillary_foggy": MapillaryFoggyDataModule,
    "mapillary_over_exposure": MapillaryOverExposureDataModule,
    "mapillary_under_exposure": MapillaryUnderExposureDataModule,
    "mapillary_motion_blur": MapillaryMotionBlurDataModule, 
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
