import cv2
import mediapipe as mp
import numpy as np

from PIL import Image


def mediapipe_detector_face(image,
                            modelname,
                            confidence,
                            label,
                            classes=None,
                            max_num_faces=100):
    if modelname == "mediapipe_face_short":
        model_selection = 0
    else:
        model_selection = 1

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    bboxes = []
    scores = []
    npimg = np.array(image)
    with mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=confidence) as face_detector:

        w, h = image.size
        results = face_detector.process(npimg)

        if not results.detections:
            return [[]] * 4

        preview = npimg.copy()
        for detection in results.detections[:max_num_faces]:
            #print(mp_face_detection.get_key_point(
            #    detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            mp_drawing.draw_detection(preview, detection)

            bbox = detection.location_data.relative_bounding_box
            x0, y0 = bbox.xmin * w, bbox.ymin * h
            ww, hh = bbox.width * w, bbox.height * h
            x1, y1 = x0 + ww, y0 + hh
            #bbox = detection.bounding_box
            #x0, y0 = bbox.origin_x, bbox.origin_y
            #x1, y1 = x0 + bbox.width, y0 + bbox.height
            bbox = np.array([x0, y0, x1, y1], dtype=np.float32)
            bboxes.append(bbox)

            # XXX score
            #category = detection.categories[0]
            #scores.append(round(category.score, 2))
            scores.append(np.float32(0))

    npimg = npimg[:, :, ::-1].copy()

    #results = [[]] * 4 # not work
    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(label)
        results[1].append(bboxes[i])
        results[3].append(scores[i])

    #preview_image = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    return results + [Image.fromarray(preview)]


def mediapipe_detector_facemesh(image,
                                modelname,
                                confidence,
                                label,
                                classes=None,
                                max_num_faces=100):
    mp_facemesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    #drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    masks = []
    scores = []
    bboxes = []
    with mp_facemesh.FaceMesh(static_image_mode=True,
                              min_detection_confidence=confidence,
                              max_num_faces=max_num_faces,
                              refine_landmarks=True) as face_detector:
        w, h = image.size
        npimg = np.array(image)

        # detect 468 facial landmarks
        results = face_detector.process(npimg)

        npimg = npimg[:, :, ::-1].copy()
        gray = cv2.cvtColor(npimg, cv2.COLOR_BGR2GRAY)

        # draw mesh and face_landmarks
        if not results.multi_face_landmarks:
            return [[]] * 4

        preview = npimg.copy()
        for landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=preview,
                landmark_list=landmarks,
                connections=mp_facemesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.
                get_default_face_mesh_tesselation_style())

            #mp_drawing.draw_landmarks(
            #    image=preview,
            #    landmark_list=landmarks,
            #    connections=mp_facemesh.FACEMESH_CONTOURS,
            #    landmark_drawing_spec=None,
            #    connection_drawing_spec=mp_drawing_styles.
            #    get_default_face_mesh_contours_style())

            #mp_drawing.draw_landmarks(
            #    image=preview,
            #    landmark_list=landmarks,
            #    connections=mp_facemesh.FACEMESH_IRISES,
            #    landmark_drawing_spec=None,
            #    connection_drawing_spec=mp_drawing_styles.
            #    get_default_face_mesh_iris_connections_style())

            # prepare bbox, masks
            points = np.intp([(m.x * w, m.y * h) for m in landmarks.landmark])

            # create convex hull from facial mesh points
            hull = cv2.convexHull(points)
            # blank mask
            mask = np.zeros((gray.shape), np.uint8)
            # fill convex hull to mask
            cv2.fillConvexPoly(mask, hull, 255)
            #cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
            # save mask
            mask_bool = mask.astype(bool)
            masks.append(mask_bool)

            # get bbox
            bb = cv2.boundingRect(hull)
            x0, y0, x1, y1 = int(bb[0]), int(bb[1]), int(bb[0]+bb[2]), int(bb[1]+bb[3])

            bbox = np.array([x0, y0, x1, y1], dtype=np.float32)
            bboxes.append(bbox)

            # adetailer method
            #points = np.array([(m.x * w, m.y * h) for m in landmarks.landmark])
            #outline = convex_hull(points)
            #mask = Image.new("L", image.size, "black")
            #draw = ImageDraw.Draw(mask)
            #draw.polygon(outline, fill="white")
            # getbbox
            #bbox = mask.resize(image.size).getbbox()
            #bboxes.append(np.array(bbox))
            # Image to np.array().astype(bool)
            #masks.append(np.array(mask).astype(bool))

            # XXX scores
            scores.append(np.float32(0.0))

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(label)
        results[1].append(bboxes[i])
        results[2].append(masks[i])
        results[3].append(scores[i])

    preview_image = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    return results + [preview_image]
