"""
gender detection based on https://github.com/Sklyvan/Age-Gender-Prediction

resnet-18-age-0.60-gender-93-f16.safetensors is a converted float16 model
from https://github.com/Sklyvan/Age-Gender-Prediction/blob/main/Models/ResNet-18/ResNet-18%20Age%200.60%20%2B%20Gender%2093.pt
"""
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import safetensors

from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse

# from basicsr
def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


_classes = 9
_groups = [
    '00-10', '11-20', '21-30',
    '31-40', '41-50', '51-60',
    '61-70', '71-80', '81-90',
]

model = models.resnet18(pretrained=False)

model.fc = nn.Linear(512, _classes + 2)
model = nn.Sequential(model, nn.Sigmoid())

path = os.path.join(os.path.dirname(__file__), "..", "..", "models")

if not os.path.exists(path):
    os.mkdir(path)

# download
modelname = "resnet-18-age-0.60-gender-93-f16.safetensors"
modelfile = os.path.join(path, modelname)
if not os.path.exists(modelfile):
    load_file_from_url("https://huggingface.co/wkpark/muddetailer/resolve/main/models/" + modelname, path)

state_dict = safetensors.torch.load_file(modelfile)
model.load_state_dict(state_dict)
model.eval()
model.to("cpu")


transform = transforms.Compose([transforms.ToTensor()])


def gender_info(image, bbox, use_cuda=True, verbose=False):
    debug_gender = False

    image = np.array(image)
    dw = bbox[2] - bbox[0]
    dh = bbox[3] - bbox[1]
    x1, x2, y1, y2 = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])

    if dw != dh:
        # preserve aspect ratio
        if dh > dw:
            pad = int((dh - dw)/2.)
            x1 = x1 - pad
            x2 = x2 + pad
            if x1 < 0:
                x2 = x2 - x1
                x1 = 0
        else:
            pad = int((dw - dh)/2.)
            y1 = y1 - pad
            y2 = y2 + pad
            if y1 < 0:
                y2 = y2 - y1
                y1 = 0

    cropped = image[y1:y2, x1:x2]
    cropped = cv2.resize(cropped, (200, 200))
    if debug_gender:
        from PIL import Image
        im = Image.fromarray(cropped)
        im.save("gender.png")

    # extract gender info
    image = transform(cropped).unsqueeze(0)
    if use_cuda:
        image = image.to("cuda")
        image = image.half()
        model.half()
        model.to("cuda")
    else:
        image = image.float()
        model.float()

    labels = model(image)[0]

    age = torch.argmax(labels[:_classes])
    gender = int(torch.argmax(labels[_classes:]))
    gender = 'male' if gender == 0 else 'female'

    c1 = float(torch.max(labels[:_classes]))
    c2 = float(torch.max(labels[_classes:]))

    if verbose:
        output = [round(float(x), 3) for x in labels]
        print(gender, output)

    if use_cuda:
        model.to("cpu")

    return gender, _groups[age], [round(c1, 3), round(c2, 3)]
