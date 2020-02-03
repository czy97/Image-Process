from PIL import ImageDraw
import cv2

def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.

    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].

    Returns:
        an instance of PIL.Image.
    """

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline='white')

    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([
                (p[i] - 1.0, p[i + 5] - 1.0),
                (p[i] + 1.0, p[i + 5] + 1.0)
            ], outline='blue')

    return img_copy

def show_bboxes_cv2(img, bounding_boxes, facial_landmarks):

    font = cv2.FONT_HERSHEY_SIMPLEX
    for bbox in bounding_boxes:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
        text = str(bbox[-1])
        cv2.putText(img, text, (int(bbox[0]), int(bbox[1])), font, 0.4, (0, 255, 0), 1)

    # plot landmarks
    for landmarks in facial_landmarks:
        for i in range(5):
            cv2.circle(img, (int(landmarks[i]), int(landmarks[i+5])), 1, (0, 0, 255), -1)

    return img

