import numpy as np
import jetson.inference
import jetson.utils
import cv2
import vpi

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

rectficationStream   = vpi.Stream()

#################################

while True:

    frameLeft  = leftCamera.Capture()
    frameRight = rightCamera.Capture()

    jetson.utils.cudaConvertColor(frameLeft, frameLeftGray) 
    jetson.utils.cudaConvertColor(frameRight,frameRightGray) 

    jetson.utils.cudaDeviceSynchronize()

    frameLeft_np  = jetson.utils.cudaToNumpy(frameLeftGray)
    frameRight_np = jetson.utils.cudaToNumpy(frameRightGray)

    frameLeft_vpi  = vpi.asimage(frameLeft_np)
    frameRight_vpi = vpi.asimage(frameRight_np)

    with vpi.Backend.CUDA:

        with rectficationStream:

            frameLeft_rect  = frameLeft_vpi.remap(warpMapLeft).rescale((640,480), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
            frameRight_rect = frameRight_vpi.remap(warpMapRight).rescale((640,480), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
    
            cv2.imshow("Rectified", np.concatenate((frameLeft_rect.cpu(), frameRight_rect.cpu()), axis=1))
    
    keyCode = cv2.waitKey(1) & 0xFF

    if keyCode == 27:
        break

cv2.destroyAllWindows()

# while display.IsStreaming():
#     img = camera.Capture()
#     display.Render(img)
#     display.SetStatus("Left Camera")

'''
    with vpi.Backend.CUDA:
     with streamLeft:
         left = vpi.asimage(np.asarray(Image.open(args.left))).convert(vpi.Format.Y16_ER, scale=scale)
     with streamRight:
         right = vpi.asimage(np.asarray(Image.open(args.right))).convert(vpi.Format.Y16_ER, scale=scale)

'''
