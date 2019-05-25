import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow_float(name, img):
	cv2.imshow(name, cv2.normalize(img, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1))

def getGaussianKernel(kernel_size):
	if(kernel_size % 2 == 0):
		raise Exception('kernel_size should be an odd nb. The value of kernel_size was: {}'.format(kernel_size))

	center = (int)(kernel_size/2),
	img = np.zeros((kernel_size,kernel_size))
	img[center, center] = 1.0
	return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

def getSobel(axis):
	if axis == 0:
		return np.array([ [-1,-2,-1],
						  [ 0, 0, 0],
						  [ 1, 2, 1]])
	else:
		return np.array([ [-1, 0, 1],
						  [-2, 0, 2],
						  [-1, 0, 1]])


def cornerHarris(imgf, kernel_size, alpha):
	gaussian_kernel = getGaussianKernel(kernel_size)

	#compute gaussian derivatives at each pixel
	sobelx=cv2.filter2D(gaussian_kernel,-1,getSobel(0))
	sobely=cv2.filter2D(gaussian_kernel,-1,getSobel(1))

	#compute gaussian derivatives at each pixel
	Ix=cv2.filter2D(imgf,-1,sobelx)
	Iy=cv2.filter2D(imgf,-1,sobely)

	#compute second moment matrix M 
	Ixsq = np.multiply(Ix , Ix)
	Iysq = np.multiply(Iy , Iy)
	I_zeros = np.zeros(Ixsq.shape)
	M_all_pixels = np.array([[Ixsq, I_zeros],[I_zeros, Iysq]])

	#rearrange axis so that there i an M matrix for each pixel
	M_per_pixel =  np.moveaxis(np.moveaxis(M_all_pixels, 0, -1), 0, -1)

	#compute corner response function for each pixel
	trace = np.trace(M_per_pixel, axis1=2,axis2=3)
	det = np.linalg.det(M_per_pixel)
	R =  det - ( np.multiply(trace, trace) * alpha)
	#R = np.divide(det, trace + np.ones(trace.shape)*0.0000001)
	return R


def non_maximum_suppresor(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			w = x + windowSize
			h = y + windowSize
			h = np.min([h,image.shape[0]])
			w = np.min([w,image.shape[1]])
			window = image[y:h, x:w]
			i = np.argmax(window)
			mask = np.zeros((h-y,w-x))
			mask[int(i/mask.shape[1])][int(i%mask.shape[1])] = 1
			image[y:h, x:w] = np.multiply(window,mask)

	return image


def show_corners(name, corners, img, threshold = 0.01):
	#result is dilated for marking the corners, not important
	img_rgb = img.copy()
	# Threshold for an optimal value, it may vary depending on the image.
	corners = non_maximum_suppresor(corners, 7,7)
	corners = cv2.dilate(corners,None)
	img_rgb[corners > threshold * np.max(corners)]=[0,0,255]
	cv2.imshow(name, img_rgb)


#----------------------------------------------------

imgScale = 1
img_rgb = cv2.imread("hcorner.jpg")
newX,newY = img_rgb.shape[1]*imgScale, img_rgb.shape[0]*imgScale
img_rgb = cv2.resize(img_rgb,(int(newX),int(newY)))
img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
imgf =  np.float32(img)

ground_truth = cv2.cornerHarris(imgf,2,3,0.04)
ours = cornerHarris(imgf, 5, 0.04)
show_corners("opencv", ground_truth, img_rgb)
show_corners("ours", ours, img_rgb, 0.024)


cv2.waitKey (-1)
