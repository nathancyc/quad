from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import torch
import detectron2
from detectron2.data.datasets import register_coco_instances, load_coco_json

torch.cuda.empty_cache()

annotations_path = "/home/nathan/documents/quad/json_annotations/val_final_v2.json"
image_path = "/data/SKU110K_fixed/images"

detectron2.data.datasets.load_coco_json(json_file = annotations_path, image_root = image_path)
register_coco_instances("my_test_dataset", {}, annotations_path, image_path)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))
print(0)
cfg.DATASETS.TEST = ("my_test_dataset",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "/home/nathan/documents/quad/output/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.OUTPUT_DIR = "/home/nathan/documents/quad/output"
print(1)
evaluator = COCOEvaluator(
    dataset_name="my_test_dataset",
    tasks=["bbox", "segm"],
    distributed=False,
    output_dir="/home/nathan/documents/quad/evaluation",
)
print(2)
val_loader = build_detection_test_loader(cfg, "my_test_dataset")
print(3)
results = inference_on_dataset(DefaultTrainer.build_model(cfg), val_loader, evaluator)
print(results)



