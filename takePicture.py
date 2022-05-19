# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
import gStreamerCamera

image_save_path = 'calibration/left_camera_data/input_images'
#image_save_path = 'calibration/right_camera_data/input_images'

def show_camera():

    camera_id = 0
    image_no = 0
    window_title = "cam%d" % (camera_id)

    camera = gStreamerCamera.CSI_Camera(camera_id, 2, 640, 480)
    camera.open()
    camera.start()

    if camera.video_capture.isOpened():

        try:
            while True:
                _, image = camera.read()

                cv2.imshow(window_title, image)

                keyCode = cv2.waitKey(30) & 0xFF

                if keyCode == 27:
                    break

                if keyCode == ord('y'):

                    print('image%d captured' % (image_no))

                    cv2.imwrite(image_save_path + '/%d.png' % (image_no), image)

                    image_no+=1
        finally:
            camera.stop()
            camera.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open both cameras")
        camera.stop()
        camera.release()

if __name__ == "__main__":
    show_camera()

