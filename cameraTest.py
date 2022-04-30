from gStreamerCamera import CSI_Camera

import cv2

camera_id = 1
flip_method = 2 
width = 1280
height = 720

camera = CSI_Camera(camera_id, flip_method, width, height)
camera.open()
camera.start()

window_title = 'cam%d' % (camera_id)

if camera.video_capture.isOpened():

    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            _, image = camera.read()

            cv2.imshow(window_title, image)

            keyCode = cv2.waitKey(30) & 0xFF

            if keyCode == 27:
                break
    finally:
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()
else:
    print("Error: Unable to open both cameras")
    camera.stop()
    camera.release()




