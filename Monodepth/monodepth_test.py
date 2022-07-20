import jetson.inference
import jetson.utils

from depthnet_utils import depthBuffers

# load the segmentation network
net = jetson.inference.depthNet("fcn-mobilenet")

# create buffer manager
buffers = depthBuffers(None)

# create video sources & outputs
input = jetson.utils.videoSource("csi://0")
output = jetson.utils.videoOutput("display://0")

# process frames until user exits
while True:
    # capture the next image
    img_input = input.Capture()

    # allocate buffers for this size image
    buffers.Alloc(img_input.shape, img_input.format)

    # process the mono depth and visualize
    net.Process(img_input, buffers.depth, "viridis-inverted", "linear")

    # composite the images
    if buffers.use_input:
        jetson.utils.cudaOverlay(img_input, buffers.composite, 0, 0)
        
    if buffers.use_depth:
        jetson.utils.cudaOverlay(buffers.depth, buffers.composite, img_input.width if buffers.use_input else 0, 0)

    # render the output image
    output.Render(buffers.composite)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
    jetson.utils.cudaDeviceSynchronize()
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

