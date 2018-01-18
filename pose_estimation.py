import dlib
import cv2
import numpy as np
import sys
from image import read_im_and_landmarks
import matplotlib.pyplot as plt
from constants import FACE_POINTS, MOUTH_POINTS, RIGHT_BROW_POINTS, LEFT_BROW_POINTS, RIGHT_EYE_POINTS, LEFT_EYE_POINTS, NOSE_POINTS, JAW_POINTS, ALIGN_POINTS, OVERLAY_POINTS, COLOUR_CORRECT_BLUR_FRAC


FEATHER_AMOUNT = 11
SCENE_SCALE_FACTOR = 0.5
WEBCAM_SCALE_FACTOR = 0.5

PREDICTOR_PATH = "./models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_parts(shape):
    return shape.part(30), shape.part(36), shape.part(45), shape.part(48), shape.part(54);

def draw_points(frame, nose, left_eye, right_eye, left_mouth, right_mouth):
    cv2.circle(frame, (nose.x, nose.y), 2, (0,0,255), -1) # nose
    cv2.circle(frame, (left_eye.x, left_eye.y), 2, (0,0,255), -1) # left eye
    cv2.circle(frame, (right_eye.x, right_eye.y), 2, (0,0,255), -1) # right eye
    cv2.circle(frame, (left_mouth.x, left_mouth.y), 2, (0,0,255), -1) # left mouth
    cv2.circle(frame, (right_mouth.x, right_mouth.y), 2, (0,0,255), -1) # right mouth

def get_landmarks(im):
    rects = detector(im)

    if len(rects) == 0:
        return np.array([]);
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]) # return the matrix of x y coordinates of landmarks

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

def alphaBlend(img1, img2, mask):
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/1.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/1.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended


cv2.namedWindow('webcam')
cv2.namedWindow('scene')

webcam = cv2.VideoCapture(int(sys.argv[1]))
scene = cv2.VideoCapture(sys.argv[2])


while(True):
    webcam_frame = webcam.read()[1]
    webcam_frame = cv2.resize(webcam_frame, (0,0), fx=WEBCAM_SCALE_FACTOR, fy=WEBCAM_SCALE_FACTOR)

    scene_frame = scene.read()[1]
    scene_frame = cv2.resize(scene_frame, (0,0), fx=SCENE_SCALE_FACTOR, fy=SCENE_SCALE_FACTOR)

    # WEBCAM
    webcam_dets = detector(webcam_frame)
    for k, d in enumerate(webcam_dets):
        webcam_shape = predictor(webcam_frame, d)
        webcam_nose, webcam_left_eye, webcam_right_eye, webcam_left_mouth, webcam_right_mouth = get_parts(webcam_shape);
        webcam_mask = get_face_mask(webcam_frame, get_landmarks(webcam_frame))
        # draw_points(webcam_frame, webcam_nose, webcam_left_eye, webcam_right_eye, webcam_left_mouth, webcam_right_mouth)


    # SCENE
    scene_dets = detector(scene_frame)
    for k, d in enumerate(scene_dets):
        scene_shape = predictor(scene_frame, d)
        scene_nose, scene_left_eye, scene_right_eye, scene_left_mouth, scene_right_mouth = get_parts(scene_shape);
        draw_points(scene_frame, scene_nose, scene_left_eye, scene_right_eye, scene_left_mouth, scene_right_mouth)


    # map face to other
    # pts_src = np.array([[webcam_left_eye.x, webcam_left_eye.y], [webcam_right_eye.x, webcam_right_eye.y], [webcam_left_mouth.x, webcam_left_mouth.y], [webcam_right_mouth.x, webcam_right_mouth.y]])
    # pts_dst = np.array([[scene_left_eye.x, scene_left_eye.y], [scene_right_eye.x, scene_right_eye.y], [scene_left_mouth.x, scene_left_mouth.y], [scene_right_mouth.x, scene_right_mouth.y]])
    # - SEEMS BETTER WITH NOSE
    pts_src = np.array([[webcam_left_eye.x, webcam_left_eye.y],[webcam_nose.x, webcam_nose.y], [webcam_right_eye.x, webcam_right_eye.y], [webcam_left_mouth.x, webcam_left_mouth.y], [webcam_right_mouth.x, webcam_right_mouth.y]])
    pts_dst = np.array([[scene_left_eye.x, scene_left_eye.y],[scene_nose.x, scene_nose.y], [scene_right_eye.x, scene_right_eye.y], [scene_left_mouth.x, scene_left_mouth.y], [scene_right_mouth.x, scene_right_mouth.y]])

    # - TRANSLATION ENTRE LES DEUX IMAGES
    h, status = cv2.findHomography(pts_src, pts_dst)
    webcam_face_translated = cv2.warpPerspective(webcam_frame, h, (scene_frame.shape[1],scene_frame.shape[0]))
    webcam_face_mask_translated = cv2.warpPerspective(webcam_mask, h, (scene_frame.shape[1],scene_frame.shape[0]))



    # mask = np.array(webcam_face_mask_translated, dtype="uint16")
    # # cv2.imshow("Masked",mask)

    # masked = cv2.bitwise_and(webcam_face_translated,webcam_face_translated,mask=mask)
    # cv2.imshow("Masked",masked)

    cv2.imshow('mapped_mask', webcam_face_mask_translated);
    cv2.imshow('mapped_face', webcam_face_translated);


    mask = webcam_face_mask_translated
    print mask
    blended1 = alphaBlend(scene_frame, webcam_face_translated, mask)
    cv2.imshow("blended", blended1)


    # comb_img = scene_frame
    # comb_img[mask] = webcam_face_translated[mask]
    # # comb_img[~mask] = webcam_face_translated[~mask]
    # cv2.imshow("debug", comb_img)



    print "webcam_face_mask_translated : " + str(webcam_face_mask_translated.shape)
    print "webcam_face_translated : " + str(webcam_face_translated.shape)
    print "scene_frame : " + str(scene_frame.shape)
    print "========\n"

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
