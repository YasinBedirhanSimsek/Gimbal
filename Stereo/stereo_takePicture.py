import cv2
from gStreamerCamera import CSI_Camera

image_save_path_left  = 'stereo_calibration/left_camera'
image_save_path_right = 'stereo_calibration/right_camera'

window_name_left  = 'left_image_window'
window_name_right = 'right_image_window'

left_camera_id  = 0
right_camera_id = 1

left_camera  = CSI_Camera(left_camera_id,  2, 640, 480)
right_camera = CSI_Camera(right_camera_id, 2, 640, 480)

left_camera.open()
right_camera.open()

left_camera.start()
right_camera.start()

image_no = 0

if left_camera.video_capture.isOpened() and right_camera.video_capture.isOpened():

    cv2.namedWindow(window_name_left,  cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_name_right, cv2.WINDOW_AUTOSIZE)

    try:
        while True:

            _, left_image = left_camera.read()
            _, right_image = right_camera.read()

            cv2.imshow(window_name_left, left_image)
            cv2.imshow(window_name_right, right_image)

            keyCode = cv2.waitKey(30) & 0xFF

            if keyCode == 27:
                break

            if keyCode == ord('y'):

                print('image%d captured' % (image_no))

                print(image_save_path_left  + '/%d_left.png'  % (image_no))
                print(image_save_path_right + '/%d_right.png' % (image_no))

                cv2.imwrite(image_save_path_left  + '/%d_left.png'  % (image_no), left_image)
                cv2.imwrite(image_save_path_right + '/%d_right.png' % (image_no), right_image)

                image_no+=1

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