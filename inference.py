import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.utils.visualizer import Visualizer
import cv2
import random
import os

annotations_path = "/home/nathan/documents/quad/json_annotations/train_final_v2.json"
image_path = "/data/SKU110K_fixed/images"

detectron2.data.datasets.load_coco_json(json_file = annotations_path, image_root = image_path)
register_coco_instances("my_dataset", {}, annotations_path, image_path)
metadata1 = MetadataCatalog.get("my_dataset")
dataset_dicts = DatasetCatalog.get("my_dataset")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))
cfg.MODEL.WEIGHTS = "/home/nathan/documents/quad/output/model_final.pth"
predictor = DefaultPredictor(cfg)

for i in range(20):
    print(f"{i+1}/20 images")
    n = random.randint(1, 1000)
    image = "test_" + str(n) + ".jpg"
    image_s = "predicted_" + str(n) + ".jpg"
    
    image_path = os.path.join("/data/SKU110K_fixed/images/", image)
    im = cv2.imread(image_path)
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], metadata=metadata1, scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    save_path = os.path.join("/home/nathan/documents/quad/test/", image_s)
    
    cv2.imwrite(save_path, v.get_image()[:, :, ::-1])

