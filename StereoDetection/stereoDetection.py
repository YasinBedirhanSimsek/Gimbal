import numpy as np
import jetson.inference
import jetson.utils
import cv2
import vpi
import time

#################################

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold = 0.5)
print("Model loaded, Starting Stereo")
time.sleep(1)

#################################

left_camera_id  = 0
right_camera_id = 1

#################################

inputWidth  = 1920
inputHeight = 1080
inputSize = (inputWidth, inputHeight)

#################################

cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

warpMapLeft = vpi.WarpMap(vpi.WarpGrid(inputSize))
wx_L,wy_L = np.asarray(warpMapLeft).transpose(2,1,0)
wx_L[:,0:1080] = stereoMapL_x.transpose(2,1,0)[0, :]
wy_L[:,0:1080] = stereoMapL_x.transpose(2,1,0)[1, :]

warpMapRight = vpi.WarpMap(vpi.WarpGrid(inputSize))
wx_R,wy_R = np.asarray(warpMapRight).transpose(2,1,0)
wx_R[:,0:1080] = stereoMapR_x.transpose(2,1,0)[0, :]
wy_R[:,0:1080] = stereoMapR_x.transpose(2,1,0)[1, :]

#################################

leftCamera  = jetson.utils.videoSource(f"csi://{left_camera_id}",  [f"--input-width={inputWidth}", f"--input-height={inputHeight}"])
rightCamera = jetson.utils.videoSource(f"csi://{right_camera_id}", [f"--input-width={inputWidth}", f"--input-height={inputHeight}"])

frameLeftGray  = jetson.utils.cudaAllocMapped(width=inputWidth, height=inputHeight, format='gray8')
frameRightGray = jetson.utils.cudaAllocMapped(width=inputWidth, height=inputHeight, format='gray8')

#################################

streamLeft = vpi.Stream()
streamRight = vpi.Stream()
streamStereo = streamLeft

#################################

maxDisparity = 64
baseLine = 150
cameraFocalLenght = 2704
distanceConversionConstant = baseLine * cameraFocalLenght
distance = np.zeros((270, 480))

#################################

def showDistance(event, x, y, flags, param):
    print(distance[y,x])

windowNameDst = 'Distance'
cv2.namedWindow(windowNameDst)
cv2.setMouseCallback(windowNameDst, showDistance)

#################################

while True:

    frameLeft  = leftCamera.Capture()
    frameRight = rightCamera.Capture()

    detections = net.Detect(frameLeft, overlay="none")

    jetson.utils.cudaConvertColor(frameLeft, frameLeftGray) 
    jetson.utils.cudaConvertColor(frameRight,frameRightGray) 

    jetson.utils.cudaDeviceSynchronize()

    frameLeft_np  = jetson.utils.cudaToNumpy(frameLeftGray)
    frameRight_np = jetson.utils.cudaToNumpy(frameRightGray)

    frameLeft_vpi  = vpi.asimage(frameLeft_np)
    frameRight_vpi = vpi.asimage(frameRight_np)

    with vpi.Backend.CUDA:

        with streamLeft:
            frameLeft_rect  = frameLeft_vpi.remap(warpMapLeft) \
                                            .rescale((480,270), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO) \
                                            .convert(vpi.Format.U16, scale=1)

        with streamRight:
              frameRight_rect = frameRight_vpi.remap(warpMapRight) \
                                            .rescale((480,270), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO) \
                                            .convert(vpi.Format.U16, scale=1)
    
        with streamStereo:
            disparity = vpi.stereodisp(frameLeft_rect, frameRight_rect, out_confmap=None, backend=vpi.Backend.CUDA, window=10, maxdisp=maxDisparity) \
                            .convert(vpi.Format.U8, scale=255.0 / (32 * maxDisparity))

            with disparity.lock():
                disparity.cpu()[disparity.cpu() < 10] = 10
                distance = (baseLine * cameraFocalLenght) /  (disparity.cpu()  + 1e-10)
                detectinoStereoResult = cv2.cvtColor(disparity.cpu(), cv2.COLOR_GRAY2BGR)
                
                for detectedObject in detections:
                    topLeft     = (int(detectedObject.Left/4), int(detectedObject.Top / 4))
                    bottomRight = (int(detectedObject.Right/4), int(detectedObject.Bottom / 4))
                    center      = (int(detectedObject.Center[0]/4), int(detectedObject.Center[1]/ 4))
                    cv2.rectangle(detectinoStereoResult, (int(detectedObject.Left/4), int(detectedObject.Top / 4)), (int(detectedObject.Right/4), int(detectedObject.Bottom / 4)), (255,0,0), 2)
                    cv2.putText(detectinoStereoResult, net.GetClassDesc(detectedObject.ClassID), topLeft, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1 ,cv2.LINE_AA)
                    cv2.putText(detectinoStereoResult, "{:10.4f}".format(distance[center[1], center[0]]), center,  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                
                cv2.imshow(windowNameDst, detectinoStereoResult)

                #disparity.cpu()[disparity.cpu() < 10] = 10
                #distance = (baseLine * cameraFocalLenght) /  (disparity.cpu()  + 1e-10)
                #cv2.imshow(windowNameDst, disparity.cpu())

    keyCode = cv2.waitKey(1) & 0xFF

    if keyCode == 27:
        break

vpi.clear_cache()
leftCamera.Close()
rightCamera.Close()
cv2.destroyAllWindows()

#gst-launch-1.0 ximagesrc num-buffers=1000 use-damage=0 ! video/x-raw ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=a.mp4