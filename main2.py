import math
import time
from threading import Thread
import numpy as np
import cv2
import tensorflow as tf

fall = 0

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

MIN_CROP_KEYPOINT_SCORE = 0.2

input_size = 192
interpreter = tf.lite.Interpreter(model_path="movenet_singlepose_lightning.tflite")
interpreter.allocate_tensors()

keypoints_with_scores = []


def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores


def init_crop_region(image_height, image_width):
    """Defines the default crop region.

    The function provides the initial crop region (pads the full image from both
    sides to make it a square image) when the algorithm cannot reliably determine
    the crop region from the previous frame.
    """
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width

    return {
        'y_min': y_min,
        'x_min': x_min,
        'y_max': y_min + box_height,
        'x_max': x_min + box_width,
        'height': box_height,
        'width': box_width
    }


def torso_visible(keypoints):
    """Checks whether there are enough torso keypoints.

    This function checks whether the model is confident at predicting one of the
    shoulders/hips which is required to determine a good crop region.
    """
    return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] >
             MIN_CROP_KEYPOINT_SCORE or
             keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] >
             MIN_CROP_KEYPOINT_SCORE) and
            (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] >
             MIN_CROP_KEYPOINT_SCORE or
             keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] >
             MIN_CROP_KEYPOINT_SCORE))


def determine_torso_and_body_range(keypoints, target_keypoints, center_y, center_x):
    """Calculates the maximum distance from each keypoints to the center location.

    The function returns the maximum distances from the two sets of keypoints:
    full 17 keypoints and 4 torso keypoints. The returned information will be
    used to determine the crop size. See determineCropRegion for more detail.
    """
    torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for joint in torso_joints:
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        if dist_y > max_torso_yrange:
            max_torso_yrange = dist_y
        if dist_x > max_torso_xrange:
            max_torso_xrange = dist_x

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for joint in KEYPOINT_DICT.keys():
        if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
            continue
        dist_y = abs(center_y - target_keypoints[joint][0]);
        dist_x = abs(center_x - target_keypoints[joint][1]);
        if dist_y > max_body_yrange:
            max_body_yrange = dist_y

        if dist_x > max_body_xrange:
            max_body_xrange = dist_x

    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]


def determine_crop_region(keypoints, image_height, image_width):
    """Determines the region to crop the image for the model to run inference on.

    The algorithm uses the detected joints from the previous frame to estimate
    the square region that encloses the full body of the target person and
    centers at the midpoint of two hip joints. The crop size is determined by
    the distances between each joints and the center point.
    When the model is not confident with the four torso joint predictions, the
    function returns a default crop which is the full image padded to square.
    """
    target_keypoints = {}
    for joint in KEYPOINT_DICT.keys():
        target_keypoints[joint] = [
            keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
            keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
        ]

    if torso_visible(keypoints):
        center_y = (target_keypoints['left_hip'][0] +
                    target_keypoints['right_hip'][0]) / 2;
        center_x = (target_keypoints['left_hip'][1] +
                    target_keypoints['right_hip'][1]) / 2;

        (max_torso_yrange, max_torso_xrange,
         max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
            keypoints, target_keypoints, center_y, center_x)

        crop_length_half = np.amax(
            [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
             max_body_yrange * 1.2, max_body_xrange * 1.2])

        tmp = np.array(
            [center_x, image_width - center_x, center_y, image_height - center_y])
        crop_length_half = np.amin(
            [crop_length_half, np.amax(tmp)]);

        crop_corner = [center_y - crop_length_half, center_x - crop_length_half];

        if crop_length_half > max(image_width, image_height) / 2:
            return init_crop_region(image_height, image_width)
        else:
            crop_length = crop_length_half * 2;
            return {
                'y_min': crop_corner[0] / image_height,
                'x_min': crop_corner[1] / image_width,
                'y_max': (crop_corner[0] + crop_length) / image_height,
                'x_max': (crop_corner[1] + crop_length) / image_width,
                'height': (crop_corner[0] + crop_length) / image_height -
                          crop_corner[0] / image_height,
                'width': (crop_corner[1] + crop_length) / image_width -
                         crop_corner[1] / image_width
            }
    else:
        return init_crop_region(image_height, image_width)


def crop_and_resize(image, crop_region, crop_size):
    """Crops and resize the image to prepare for the model input."""
    boxes = [[crop_region['y_min'], crop_region['x_min'],
              crop_region['y_max'], crop_region['x_max']]]
    output_image = tf.image.crop_and_resize(
        image, box_indices=[0], boxes=boxes, crop_size=crop_size)
    return output_image


def run_inference(movenet, image, crop_region, crop_size):
    """Runs model inferece on the cropped region.

    The function runs the model inference on the cropped region and updates the
    model output to the original image coordinate system.
    """
    image_height, image_width, _ = image.shape
    input_image = crop_and_resize(
        tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
    # Run model inference.
    keypoints_with_scores = movenet(input_image)
    # Update the coordinates.
    for idx in range(17):
        keypoints_with_scores[0, 0, idx, 0] = (
                                                      crop_region['y_min'] * image_height +
                                                      crop_region['height'] * image_height *
                                                      keypoints_with_scores[0, 0, idx, 0]) / image_height
        keypoints_with_scores[0, 0, idx, 1] = (
                                                      crop_region['x_min'] * image_width +
                                                      crop_region['width'] * image_width *
                                                      keypoints_with_scores[0, 0, idx, 1]) / image_width
    return keypoints_with_scores


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def checkFall(threshold, period):
    global fall
    time.sleep(10)
    while True:

        yNose = keypoints_with_scores[0][0][0][0]
        yFeet = (keypoints_with_scores[0][0][14][0] + keypoints_with_scores[0][0][13][0]) / 2

        yDiff = yFeet - yNose

        time.sleep(period)

        yNose = keypoints_with_scores[0][0][0][0]
        yFeet = (keypoints_with_scores[0][0][14][0] + keypoints_with_scores[0][0][13][0]) / 2

        yDiff2 = yFeet - yNose
        if yDiff2 < threshold * yDiff:
            print("Fall detected "+ str(fall))
            fall=fall+1
            #print(yDiff-yDiff2)


def checkFall2(threshold, period):
    time.sleep(10)
    while True:
        flags = [False, False, False]
        legs = [[0, 0], [0, 0], [0, 0]]
        # 1st heuristic
        yNose = keypoints_with_scores[0][0][0][0]
        yFeet = (keypoints_with_scores[0][0][16][0] + keypoints_with_scores[0][0][15][0]) / 2
        yDiff = yFeet - yNose
        print(yDiff)

        time.sleep(period)

        yNose = keypoints_with_scores[0][0][0][0]
        yFeet = (keypoints_with_scores[0][0][16][0] + keypoints_with_scores[0][0][15][0]) / 2
        yDiff2 = yFeet - yNose
        print(yDiff2)

        if yDiff2 < threshold * yDiff:
            flags[0] = True
            print("Fall detected")

        legs[0][0] = (keypoints_with_scores[0][0][16][1] + keypoints_with_scores[0][0][15][1]) / 2
        legs[0][1] = (keypoints_with_scores[0][0][16][0] + keypoints_with_scores[0][0][15][0]) / 2

        legs[1][0] = (keypoints_with_scores[0][0][13][1] + keypoints_with_scores[0][0][14][1]) / 2
        legs[1][1] = (keypoints_with_scores[0][0][13][0] + keypoints_with_scores[0][0][14][0]) / 2

        legs[1][0] = (keypoints_with_scores[0][0][11][1] + keypoints_with_scores[0][0][12][1]) / 2
        legs[1][1] = (keypoints_with_scores[0][0][11][0] + keypoints_with_scores[0][0][12][0]) / 2

        lineA = ((legs[0][1] + legs[1][1]) / (legs[0][0] - legs[1][0]))
        lineB = legs[0][1] - ((legs[0][1] + legs[1][1]) / (legs[0][0] - legs[1][0])) * legs[0][0]

        d = abs(lineA * legs[2][0] + -1 * legs[2][1] + lineB) / (math.sqrt(lineA ** 2 + lineB ** 2))
        c = math.sqrt((legs[2][0] - legs[0][0]) ** 2 + (legs[2][1] - legs[0][1]) ** 2)

        sinA = d / c

        alpha = np.arcsin(sinA)
        print("alpha: ", alpha)

        if alpha > np.pi / 3:
            flags[1] = True

def main():
    #cap = cv2.VideoCapture("fall.mp4")
    cap = cv2.VideoCapture(1)
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    crop_region = init_crop_region(frame_height, frame_width)

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            global keypoints_with_scores
            keypoints_with_scores = run_inference(movenet, frame, crop_region, crop_size=[input_size, input_size])
            draw_connections(frame, keypoints_with_scores, KEYPOINT_EDGE_INDS_TO_COLOR, 0.4)
            draw_keypoints(frame, keypoints_with_scores, 0.4)
            crop_region = determine_crop_region(keypoints_with_scores, frame_height, frame_width)
            cv2.imshow("Pose Estimation", frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        # Break the loop
        else:
            break

    # Release video capture object
    cap.release()
    # Close all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    thread = Thread(target=checkFall, args=(0.7, 0.2,), daemon=True)
    thread.start()
    main()