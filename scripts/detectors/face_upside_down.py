"""
Simple upside down detection
"""
import cv2
import numpy as np
import torch

#model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

def is_face_upside_down(image, bbox, use_cuda=True, verbose=False):
    debug = True
    verbose = True

    image = np.array(image)
    x1, x2, y1, y2 = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])

    face = image[y1:y2, x1:x2].copy()
    if debug:
        from PIL import Image
        im = Image.fromarray(face)
        im.save("face.png")

    # gray scale face
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Detect eyes within the face region
    eyes = model.detectMultiScale(face)
    if len(eyes) < 2:
        print("eyes = ", len(eyes))
        # not detected eyes. in this case, the image is suspected to be upside down.
        return True

    eyes_ypos = [eye[1] for eye in eyes]
    if verbose:
        print("detected eyes y-pos=", eyes_ypos)

    eyes_ypos = sorted(eyes_ypos, reverse=True)[:2]

    # detect eyes in the flipped face
    flipped_face = cv2.flip(face, 0)
    #flipped_eyes = model.detectMultiScale(flipped_face, 1.1, 3)
    flipped_eyes = model.detectMultiScale(flipped_face)

    if len(flipped_eyes) < 2:
        # not detected eyes. in this case, the flipped image suspected to be upside down.
        print("flipped eyes = ", len(flipped_eyes))
        return False

    flipped_eyes_ypos = [eye[1] for eye in flipped_eyes]
    if verbose:
        print("detected flipped face's eyes y-pos=", flipped_eyes_ypos)
    flipped_eyes_ypos = sorted(flipped_eyes_ypos, reverse=True)[:2]

    # Check if the face is upside down by comparing eyes positions
    if sum(flipped_eyes_ypos) * 0.5 > sum(eyes_ypos) * 0.5:
        return False
    return True
