"""
YOLO inference wrapper
"""
import cv2
import json
import numpy as np
import torch
import os

from modules import safe
from PIL import Image

def ultralytics_inference(image, model_path, conf_thres, label, classes=None, max_per_img=100, device="cpu"):
    from ultralytics import YOLO

    safe_torch_load = torch.load
    try:
        torch.load = safe.unsafe_torch_load
        model = YOLO(model_path)
    finally:
        torch.load = safe_torch_load

    # override class names
    classes_path = model_path.rsplit(".", 1)[0] + ".json"
    _classes = None
    if os.path.exists(classes_path):
        with open(classes_path) as f:
            _classes = json.load(f)

    if classes is not None:
        if classes is str:
            classes = [classes]
        if _classes is None and model.names is not None:
            _classes = list(model.names.values())

        if len(classes) == 0:
            classes = None
        else:
            classes = [_classes.index(cls) for cls in classes if cls in _classes]

    result = model(image, conf=conf_thres, device=device, max_det=max_per_img, classes=classes)[0]

    bboxes = None
    scores = None
    labels = None
    masks = None
    if result.boxes is not None:
        bboxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy() # box confidence scores
        if _classes is not None:
            labels = [f"{label}-{_classes[cls.astype(np.intp)]}" if cls < len(_classes) else f"{label}-{str(cls.astype(np.intp))}"
                        for cls in result.boxes.cls.cpu().numpy()
                     ] # labels with class
        elif result.names is not None:
            labels = [f"{label}-{result.names[cls.astype(np.intp)]}" for cls in result.boxes.cls.cpu().numpy()] # has class names
        else:
            labels = [f"{label}-{str(cls.astype(np.intp))}" for cls in result.boxes.cls.cpu().numpy()] # no class name given

    if result.masks is not None:
        masks = result.masks.xy

    segms = []
    if masks is not None:
        for segm in masks:
            # mask segments to bitmap masks
            mask = np.zeros((image.height, image.width), np.uint8)
            # fill segm to mask
            cv2.fillConvexPoly(mask, segm.astype(np.intp), 255)
            # save mask
            segms.append(mask.astype(bool))

    if bboxes is None and masks is None:
        return [[], [], [], []]

    if labels is None:
        labels = [label] * len(bboxes)

    if scores is None:
        scores = np.array([0.0] * len(bboxes)).astype(np.float32)

    # return preview
    preview = result.plot(font_size=8)
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(preview)

    return [labels, bboxes, segms, scores] + [preview]
