import numpy as np
import cv2

# Inspired by : https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py

prototxt_path = "models1/colorization_deploy_v2.prototxt"
model_path = "models1/colorization_release_v2.caffemodel"
kernel_path = "models1/pts_in_hull.npy"
image_path = 'lake.jpg'

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path) #load model and set it to test mode
points = np.load(kernel_path) #load points from disk

points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)] 
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype=np.float32)] #setting the reference blob for the network

bw_image = cv2.imread(image_path) #black and white image
normalized = bw_image.astype("float32") / 255.0 #normalizing the image
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB) #convert bgr to lab

resized = cv2.resize(lab, (224, 224)) #resize the image
L = cv2.split(resized)[0] #split the image into L channel
L -= 50 #subtract 50 from the L channel

net.setInput(cv2.dnn.blobFromImage(L)) #set the input for the network
ab = net.forward()[0, :, :, :].transpose((1, 2, 0)) #get the output from the network and transpose it
ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0])) #resize the output to the original image size
L = cv2.split(lab)[0] #split the original image into L channel

colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2) #concatenate the L and ab channels
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR) #convert the colorized image from LAB to BGR
colorized = (255.0 * colorized).astype("uint8") #convert the colorized image to uint8 type

cv2.imshow("BW", bw_image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()
#points = points.transpose().reshape(2, 313, 1, 1) 
#This line is reshaping a data structure called 'points'. It first transposes the data (swaps rows and columns), then reshapes it into a 4-dimensional array with dimensions 2 x 313 x 1 x 1. This is likely preparing the data for input into a neural network layer.

#net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)] 
# Here, we're accessing a specific layer in a neural network (likely for image colorization) named "class8_ab". We're setting its 'blobs' (which are essentially the layer's data) to our reshaped 'points' array, converted to 32-bit float type.

#net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype=np.float32)] 
# Similar to the previous line, this is setting the 'blobs' for another layer named "conv8_313_rh". However, instead of using our 'points' data, it's creating a new array filled with the value 2.606, with dimensions 1 x 313, also in 32-bit float type.