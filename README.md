# μ Detection Detailer
This is yet another detection detailer named μ-Detection Detailer (shortly μ-DDetailer) forked from [DDetailer](https://github.com/dustysys/ddetailer)
as a postprocess extension for [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Features
1. bbox/segment object detection using [MMDetection](https://github.com/open-mmlab/mmdetection), [MMYOLO](https://github.com/open-mmlab/mmyolo), [mediapipe](https://github.com/google/mediapipe) and [ultralytics](https://github.com/ultralytics/ultralytics)
2. Inpainting helper added to support dynamically generated masks/segments viewer using js canvas to select detections easily.
3. Support importing ADetailer/DDetailer infotext settings
4. both mmdet 2.x / mmdet 3.x are supported
5. Support Load/Save presets

## Screenshot
![image](https://github.com/wkpark/uddetailer/assets/232347/7f297cc3-4379-43d0-a74e-91183a78f42c)

## Introduction
It uses the [MMDetection](https://github.com/open-mmlab/mmdetection) library to detect objects, mask them especially human faces, and then partially redraw them to bring out the details.
Optionally It uses [MMYOLO](https://github.com/open-mmlab/mmyolo), [mediapipe](https://github.com/google/mediapipe) and [ultralytics](https://github.com/ultralytics/ultralytics) libraries to detect objects.

![adoringfan](/misc/ddetailer_example_1.png)

### Segmentation
Default models enable person and face instance segmentation.

![amgothic](/misc/ddetailer_example_2.png)

### Detailing
With full-resolution inpainting, the extension is handy for improving faces without the hassle of manual masking.

![zion](/misc/ddetailer_example_3.gif)

## Installation
1. Use `git clone https://github.com/wkpark/uddetailer.git` from your SD web UI `/extensions` folder. Alternatively, install from the extensions tab with url `https://github.com/wkpark/uddetailer`
2. Press the `Apply and restart` button or restart SD web UI.

The models and dependencies should download automatically. To install them manually, follow the [official instructions for installing mmdet](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-mim-recommended). The models can be [downloaded here](https://huggingface.co/dustysys/ddetailer) and should be placed in `/models/mmdet/bbox` for bounding box (`anime-face_yolov3`) or `/models/mmdet/segm` for instance segmentation models (`dd-person_mask2former`). See the [MMDetection docs](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html) for guidance on training your own models. For using official MMDetection pretrained models see [here](https://github.com/dustysys/ddetailer/issues/5#issuecomment-1311231989), these are trained for photorealism. See [Troubleshooting](https://github.com/wkpark/uddetailer#troubleshooting) if you encounter issues during installation.

It also supports additional [ultralytics](https://github.com/ultralytics/ultralytics) detection models and you can install YoloV8 models manually. ultralytics's models should be placed in the `models/yolo` dir.

## Usage
To use μ Detection Detailer in SD web UI, `enable` μ Detection Detailer first, and select optional parameters, then click 'Generate'. Here are some tips:
- `anime-face_yolov3` can detect the bounding box of faces as the primary model while `dd-person_mask2former` isolates the head's silhouette as the secondary model by using the bitwise AND option. Refer to [this example](https://github.com/dustysys/ddetailer/issues/4#issuecomment-1311200268).
- The dilation factor expands the mask, while the x & y offsets move the mask around.
- You can change the default prompt, negative prompt, and even checkpoint model.
- The extension is available in the img2img tab as well and can improve the quality of your 10 pulls with moderate settings (low denoise).

## Troubleshooting
If you get the message ERROR: 'Failed building wheel for pycocotools' follow [these steps](https://github.com/dustysys/ddetailer/issues/1#issuecomment-1309415543).

For any other issues installing, open an [issue](https://github.com/wkpark/uddetailer/issues).

## Credits
dustysys/[DDetailer](https://github.com/dustydust/ddetailer) - Author of the original Detection Detailer.

bingsu/[ADetailer](https://github.com/bing-su/adetailer) - Author of the ADetailer and creator of face_yolov8\*pt, hand_yolov8\*.pt and more.

hysts/[anime-face-detector](https://github.com/hysts/anime-face-detector) - Creator of `anime-face_yolov3`, which has impressive performance on a variety of art styles.

skytnt/[anime-segmentation](https://huggingface.co/datasets/skytnt/anime-segmentation) - Synthetic dataset used to train `dd-person_mask2former`.

jerryli27/[AniSeg](https://github.com/jerryli27/AniSeg) - Annotated dataset used to train `dd-person_mask2former`.

open-mmlab/[MMDetection](https://github.com/open-mmlab/mmdetection) - Object detection toolset. `dd-person_mask2former` was trained via transfer learning using their [R-50 Mask2Former instance segmentation model](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former#instance-segmentation) as a base.

open-mmlab/[MMYOLO](https://github.com/open-mmlab/mmyolo) - MMYOLO is an open-source toolbox for YOLO series algorithms based on PyTorch and MMDetection.

google/[mediapipe](https://github.com/google/mediapipe) - Mediapipe is an open-source framework for building real-time perceptual computing applications.

ultralytics/[ultralytics](https://github.com/ultralytics/ultralytics) - Ultralytics is an open-source framework to support object detection and tracking, instance seg, classification and pose estimation framework.

AUTOMATIC1111/[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - Web UI for Stable Diffusion, base application for this extension.
