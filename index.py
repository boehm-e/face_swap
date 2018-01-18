import cv2
import numpy as np
import sys
from image import get_landmarks, read_im_and_landmarks, transformation_from_points, get_face_mask, warp_im, correct_colours
from constants import FACE_POINTS, MOUTH_POINTS, RIGHT_BROW_POINTS, LEFT_BROW_POINTS, RIGHT_EYE_POINTS, LEFT_EYE_POINTS, NOSE_POINTS, JAW_POINTS, ALIGN_POINTS, OVERLAY_POINTS, COLOUR_CORRECT_BLUR_FRAC

webcam = cv2.VideoCapture(int(sys.argv[1]))
scene = cv2.VideoCapture(sys.argv[2])

while(True):
    im1_scene, landmarks1_webcam = read_im_and_landmarks(scene.read()[1], 0.5)
    im2_webcam, landmarks2_scene = read_im_and_landmarks(webcam.read()[1], 0.5)

    if (landmarks1_webcam.size == 0) or (landmarks2_scene.size == 0):
        continue

    M = transformation_from_points(landmarks1_webcam[ALIGN_POINTS],
                                   landmarks2_scene[ALIGN_POINTS])
    mask2_scene = get_face_mask(im2_webcam, landmarks2_scene)
    mask1_webcam = get_face_mask(im1_scene, landmarks1_webcam)

    warped_mask = warp_im(mask2_scene, M, im1_scene.shape)
    combined_mask = np.max([mask1_webcam, warped_mask], axis=0)
    cv2.imshow("combined_mask", combined_mask )


    warped_im2 = warp_im(im2_webcam, M, im1_scene.shape)
    cv2.imshow("warped", warped_im2 )
    warped_corrected_im2 = correct_colours(im1_scene, warped_im2, landmarks1_webcam)
    cv2.imshow("warped_corrected_im2", warped_corrected_im2 )

    output_im = im1_scene * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask


    cv2.imshow("out", output_im.astype(im1_scene.dtype) )


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
