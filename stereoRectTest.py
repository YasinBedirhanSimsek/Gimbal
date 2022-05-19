import cv2
import numpy as np
from gStreamerCamera import CSI_Camera
import time

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

####################################################

window_name_left  = 'stereo_rectified_left'
window_name_right = 'stereo_rectified_right'

left_camera_id  = 0
right_camera_id = 1

left_camera  = CSI_Camera(left_camera_id,  2, 640, 480)
right_camera = CSI_Camera(right_camera_id, 2, 640, 480)

left_camera.open()
right_camera.open()

left_camera.start()
right_camera.start()

if left_camera.video_capture.isOpened() and right_camera.video_capture.isOpened():

    cv2.namedWindow(window_name_left,  cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_name_right, cv2.WINDOW_AUTOSIZE)

    try:
        while True:

            _, image_left  = left_camera.read()
            _, image_right = right_camera.read()

            t = time.time() 

            frame_left_rect  = cv2.remap(src=image_left,  map1 = stereoMapL_x, map2 = stereoMapL_y, interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT, borderValue = 0)
            frame_right_rect = cv2.remap(src=image_right, map1 = stereoMapR_x, map2 = stereoMapR_y, interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT, borderValue = 0)
            
            print(time.time() - t)

            cv2.imshow(window_name_left, frame_left_rect)
            cv2.imshow(window_name_right, frame_right_rect) 

            keyCode = cv2.waitKey(1) & 0xFF

            if keyCode == ord('y'):

                print('image%d captured')

                cv2.imwrite('left.png',  frame_left_rect)
                cv2.imwrite('right.png', frame_right_rect)


            if keyCode == 27:
                break

    finally:
        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()
    
    cv2.destroyAllWindows()
    
else:
    print("Error: Unable to open both cameras")
    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()