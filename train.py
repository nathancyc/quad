import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.utils.visualizer import Visualizer
from torch import nn
import torch

import random
import cv2

import json
import os

torch.cuda.empty_cache()

annotations_path = "/home/nathan/documents/quad/json_annotations/train_final_v2.json"
image_path = "/data/SKU110K_fixed/images"

detectron2.data.datasets.load_coco_json(json_file = annotations_path, image_root = image_path)
register_coco_instances("my_dataset", {}, annotations_path, image_path)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))
cfg.DATASETS.TRAIN = ("my_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.WEIGHTS = "/home/nathan/documents/quad/model_final_f10217.pkl"
cfg.SOLVER.IMS_PER_BATCH = 8

# Weight decay
cfg.SOLVER.BASE_LR = 0.0050  # learning rate
cfg.SOLVER.GAMMA = 0.0002
# The iteration number to decrease learning rate by GAMMA.
cfg.SOLVER.STEPS = (1000,)

cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 10240  # Adjust based on your needs
cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
cfg.OUTPUT_DIR = "/home/nathan/documents/quad/outputv2"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Train the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
