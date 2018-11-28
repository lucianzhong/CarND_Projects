import os
import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# step 1 Compute the camera calibration matrix and distortion coefficients given a set of chessboard images

# Read in chessboard images
images = glob.glob("camera_cal/calibration*.jpg")
plt.figure(figsize=(20, 10))
for counter_images, fname in enumerate(images):
    img = plt.imread(fname)
    plt.subplot(len(images) // 5, 5, counter_images + 1)
    plt.imshow(img)

# extract image points form chessborad and extract object points
# generate object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
NUM_X = 9
NUM_Y = 6
object_points = np.zeros((NUM_Y * NUM_X, 3), np.float32)
object_points[:, :2] = np.mgrid[0:NUM_X, 0:NUM_Y].T.reshape(-1, 2)
# store object points and image points in arrays
objpoints = []
imgpoints = []
images = glob.glob('camera_cal/calibration*.jpg')

# convert images to grayscales
for counter_image, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (NUM_X, NUM_Y), None)

# Finding chessboard corners and add object points and image points
    if ret == True:
        objpoints.append(object_points)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (NUM_X, NUM_Y), corners, ret)
        cv2.imshow('chessboard images', img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

# calibrate camera by using image points and object points
img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# save the calibration results
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("camera_dist.p", "wb"))

# step 2 Apply a distortion correction to raw images
#define the undistortion function
def undistortion_function(img, mtx=mtx, dist=dist):
    distorsion_correction = cv2.undistort(img, mtx, dist, None, mtx)
    return distorsion_correction

# distortion correction and plot these images
calibration_images = glob.glob('camera_cal/calibration*.jpg')
plt.figure(figsize=(40, 20))
for fname in calibration_images:
    img = cv2.imread(fname)
    dst = undistortion_function(img)
    cv2.imwrite('output_images/undistorted_' + fname.replace('camera_cal/', ''), dst)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax1.set_title('Original ' + fname.replace('camera_cal/', ''), fontsize=10)
    ax2.imshow(dst)
    ax2.set_title('Undistorted ' + fname.replace('camera_cal/', ''), fontsize=10)


# step 3 Use color transforms, gradients, etc., to create a thresholded binary image.
# define gradient and color threshold function
# Define Sobel operators in the x direction
def abs_sobel_threshold(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

# define function to calculate gradient magnitude in both x and y direction
def mag_threshold(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Pixals lower than threshold in zero,otherwise is one and output a binary image
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

# define function to calculate gradient direction in both x and y direction
def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Pixals lower than threshold in zero,otherwise is one and output a binary image
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary_output

# Convert RGB space to HLS space and separate the S channel
def s_channel_translation(img, s_thresh=(170, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 2]
    # Apply threshold to S channel and generate binary image
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    return s_binary

#step 4:Apply a perspective transform to rectify binary image ("birds-eye view")

# Define source and destination points for the perspective transform
def get_warp_points(image):
    source_points = np.float32([[580, 460], [700, 460], [1040, 680], [300, 680]])
    destination_points = np.float32([[260, 0], [1040, 0], [1040, 720], [260, 720]])
    return source_points, destination_points

# define the perspective transformation function
def warp_image(img, src_point, dst_point, img_size):
    M = cv2.getPerspectiveTransform(src_point, dst_point)
    Minv = cv2.getPerspectiveTransform(dst_point, src_point)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, Minv

# Test perspective transformation by using test1.jpg by convert to grayscale,distorsion correction and perspective transformation
image = cv2.imread("test_images/test1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = undistortion_function(image)
src, dst = get_warp_points(image)
# Plot the chosen source and destination points on the original image
points_image_src = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
points_image_dst = points_image_src.copy()
src_pts = src.reshape((-1, 1, 2)).astype("int32")
cv2.polylines(points_image_src, [src_pts], True, (255, 0, 0), thickness=5)
plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.imshow(points_image_src)
plt.title("Source points")
plt.axis("off")
cv2.imwrite("output_images/source_points.jpg", points_image_src)
dst_pts = dst.reshape((-1, 1, 2)).astype("int32")
cv2.polylines(points_image_dst, [dst_pts], True, (0, 255, 0), thickness=5)
plt.subplot(1, 2, 2)
plt.imshow(points_image_dst)
plt.title("Destination points")
plt.axis("off")
cv2.imwrite("output_images/destination_points.jpg", points_image_dst)

# Test perspective transformation by using straight_lines1.jpg and save its bird's view
image1 = cv2.imread("test_images/straight_lines1.jpg")
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image1 = undistortion_function(image1, mtx, dist)
src, dst = get_warp_points(image1)
warped1,_,_= warp_image(image1, src, dst, (image.shape[1], image.shape[0]))
cv2.imwrite("output_images/warped_straight_lines1.jpg", warped1)
plt.figure(figsize=(16,8))
plt.subplot(1, 2, 1)
plt.imshow(image1)
plt.title("original straight road");
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(warped1)
plt.title("Birds eye view for straight road");
plt.axis("off")

# Test perspective transformation by using straight_lines1.jpg and save its bird's view
image2 = cv2.imread("test_images/test2.jpg")
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image2 = undistortion_function(image2, mtx, dist)
warped2,_,_= warp_image(image2, src, dst, (image2.shape[1], image2.shape[0]))
cv2.imwrite("output_images/warped_test2.jpg", warped2)

# Convert raw image1.jpg to RGB space, HLS space and HSV space and plot the perspective transform
image1_R = warped1[:,:,0]
image1_G = warped1[:,:,1]
image1_B = warped1[:,:,2]
# HLS space
image1_HLS = cv2.cvtColor(warped1, cv2.COLOR_RGB2HLS)
image1_H = image1_HLS[:,:,0]
image1_L = image1_HLS[:,:,1]
image1_S = image1_HLS[:,:,2]
# HSV space
image1_HSV = cv2.cvtColor(warped1, cv2.COLOR_RGB2HSV)
image1_H2 = image1_HSV[:,:,0]
image1_S2 = image1_HSV[:,:,1]
image1_V = image1_HSV[:,:,2]

fig, axs = plt.subplots(3,3, figsize=(20, 10))
fig.subplots_adjust(hspace = .2, wspace=.1)
axs = axs.ravel()
axs[0].imshow(image1_R, cmap='gray')
axs[0].set_title('RGB R-channel', fontsize=15)
axs[1].imshow(image1_G, cmap='gray')
axs[1].set_title('RGB G-Channel', fontsize=15)
axs[2].imshow(image1_B, cmap='gray')
axs[2].set_title('RGB B-channel', fontsize=15)
axs[3].imshow(image1_H, cmap='gray')
axs[3].set_title('HLS H-Channel', fontsize=15)
axs[4].imshow(image1_L, cmap='gray')
axs[4].set_title('HLS L-channel', fontsize=15)
axs[5].imshow(image1_S, cmap='gray')
axs[5].set_title('HLS S-Channel', fontsize=15)
axs[6].imshow(image1_H2, cmap='gray')
axs[6].set_title('HSV H-Channel', fontsize=15)
axs[7].imshow(image1_S2, cmap='gray')
axs[7].set_title('HSV S-channel', fontsize=15)
axs[8].imshow(image1_V, cmap='gray')
axs[8].set_title('HSV V-Channel', fontsize=15)

# Apply graident threshold and plot binary images
warped_grayscale = cv2.cvtColor(warped2, cv2.COLOR_RGB2GRAY)
warped_sobel_threshold_x = abs_sobel_threshold(warped_grayscale, orient='x', sobel_kernel=9, thresh=(20, 100))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.1)
ax1.imshow(warped_grayscale, cmap='gray')
ax1.set_title('Origin', fontsize=15)
ax2.imshow(warped_sobel_threshold_x, cmap='gray')
ax2.set_title('Sobel X', fontsize=15)

warped_sobel_threshold_y = abs_sobel_threshold(warped_grayscale, orient='y', sobel_kernel=9, thresh=(20, 100))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.1)
ax1.imshow(warped_grayscale, cmap='gray')
ax1.set_title('Origin', fontsize=15)
ax2.imshow(warped_sobel_threshold_y, cmap='gray')
ax2.set_title('Sobel Y', fontsize=15)

warped_mag_threshold = mag_threshold(warped_grayscale, sobel_kernel=9, mag_thresh=(20, 100))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.1)
ax1.imshow(warped_grayscale, cmap='gray')
ax1.set_title('Origin', fontsize=15)
ax2.imshow(warped_mag_threshold, cmap='gray')
ax2.set_title('Magient Thresh', fontsize=15)

warped_dir_threshold = dir_threshold(warped_grayscale, sobel_kernel=9, thresh=(0.7, 1.3))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.1)
ax1.imshow(warped_grayscale, cmap='gray')
ax1.set_title('Origin', fontsize=15)
ax2.imshow(warped_dir_threshold, cmap='gray')
ax2.set_title('Direction Image', fontsize=15)

# Define HLS space
def HLS_channel(img, channel='h', thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    if channel == 'h':
        image_channel = hls[:, :, 0]
    elif channel == 'l':
        image_channel = hls[:, :, 1]
    else:
        image_channel = hls[:, :, 2]
    image_channel = image_channel * (255 / np.max(image_channel))
    # channel value lower tha threshold which is zero otherwise is one and return binary images
    binary = np.zeros_like(image_channel)
    binary[(image_channel >= thresh[0]) & (image_channel <= thresh[1])] = 1
    return binary

# Define RGB channel
def RGB_channel(img, channel='r', thresh=(0, 255)):
    if channel == 'r':
        image_channel = img[:, :, 0]
    elif channel == 'g':
        image_channel = img[:, :, 1]
    else:
        image_channel = img[:, :, 2]

    image_channel = image_channel * (255 / np.max(image_channel))
 # channel value lower tha threshold which is zero otherwise is one and return binary images
    binary = np.zeros_like(image_channel)
    binary[(image_channel >= thresh[0]) & (image_channel <= thresh[1])] = 1
    return binary

# Define HSV space
def HSV_channel(img, channel='h', thresh=(0, 255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    if channel == 'h':
        image_channel = hsv[:, :, 0]
    elif channel == 's':
        image_channel = hsv[:, :, 1]
    else:
        image_channel = hsv[:, :, 2]
    image_channel = image_channel * (255 / np.max(image_channel))
# channel value lower tha threshold which is zero otherwise is one and return binary images
    binary = np.zeros_like(image_channel)
    binary[(image_channel >= thresh[0]) & (image_channel <= thresh[1])] = 1
    return binary

# Define Lab space
def Lab_channel(img, channel='l', thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float)
    if channel == 'l':
        img_channel = lab[:, :, 0]
    elif channel == 'a':
        img_channel = lab[:, :, 1]
    else:
        img_channel = lab[:, :, 2]
 # channel value lower tha threshold which is zero otherwise is one and return binary images
    binary = np.zeros_like(img_channel)
    binary[(img_channel >= thresh[0]) & (img_channel <= thresh[1])] = 1
    return binary

# application of HLS threshold method. The results shows that HSL_S channel is good to detect white lines
warped_HLS_H = HLS_channel(warped2, channel='h', thresh=(170, 255))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.1)
ax1.imshow(warped2, cmap='gray')
ax1.set_title('Origin', fontsize=15)
ax2.imshow(warped_HLS_H, cmap='gray')
ax2.set_title('HLS H-Channel', fontsize=15)

warped_HLS_L = HLS_channel(warped2, channel='l', thresh=(190, 255))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.1)
ax1.imshow(warped2, cmap='gray')
ax1.set_title('Origin', fontsize=15)
ax2.imshow(warped_HLS_L, cmap='gray')
ax2.set_title('HLS L-Channel', fontsize=15)

warped_HLS_S = HLS_channel(warped2, channel='s', thresh=(170, 255))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.1)
ax1.imshow(warped2, cmap='gray')
ax1.set_title('Origin', fontsize=15)
ax2.imshow(warped_HLS_S, cmap='gray')
ax2.set_title('HLS S-Channel', fontsize=15)


# application of HSV threshold method.
warped_HSV_H = HSV_channel(warped2, channel='h', thresh=(170, 255))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.1)
ax1.imshow(warped2, cmap='gray')
ax1.set_title('Origin', fontsize=15)
ax2.imshow(warped_HSV_H, cmap='gray')
ax2.set_title('HSV H-Channel', fontsize=15)

warped_HSV_S = HSV_channel(warped2, channel='s', thresh=(170, 255))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.1)
ax1.imshow(warped2, cmap='gray')
ax1.set_title('Origin', fontsize=15)
ax2.imshow(warped_HSV_S, cmap='gray')
ax2.set_title('HSV S-Channel', fontsize=15)

warped_HSV_V = HSV_channel(warped2, channel='v', thresh=(170, 255))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.1)
ax1.imshow(warped2, cmap='gray')
ax1.set_title('Origin', fontsize=15)
ax2.imshow(warped_HSV_V, cmap='gray')
ax2.set_title('HSV V-Channel', fontsize=15)

# application of Lab threshold method. The results shows that Lab_B channel is good to detect yellow lines
warped_Lab_b = Lab_channel(warped2, channel='b', thresh=(190, 255))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.1)
ax1.imshow(warped2, cmap='gray')
ax1.set_title('Origin', fontsize=15)
ax2.imshow(warped_Lab_b, cmap='gray')
ax2.set_title('LAB B-Channel', fontsize=15)

# combine HLS_S channel, Lab_b channel and RGB_R channel
# test by using straight_lines2.jpg
image_combine = cv2.imread("test_images/straight_lines2.jpg")
image_combine = cv2.cvtColor(image_combine, cv2.COLOR_BGR2RGB)
image_combine = undistortion_function(image_combine, mtx, dist)
image_combine_warped, _, _ = warp_image(image_combine, src, dst, (image_combine.shape[1], image_combine.shape[0]))

image_combine_HLS_binary = HLS_channel(image_combine_warped, channel='s', thresh=(220, 255))
image_combine_Lab_binary = Lab_channel(image_combine_warped, channel='b', thresh=(180, 255))
image_combine_RGB_binary = RGB_channel(image_combine_warped, channel='r', thresh=(180, 255))

image_combine_channel_binary = np.zeros_like(image_combine_HLS_binary)
image_combine_channel_binary[(image_combine_HLS_binary == 1) | (image_combine_Lab_binary == 1) | (image_combine_RGB_binary == 1)] = 1

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.1)
ax1.imshow(image_combine, cmap='gray')
ax1.set_title('Origin', fontsize=15)
ax2.imshow(image_combine_channel_binary, cmap='gray')
ax2.set_title('output Image', fontsize=15)

# define combined binary function
def combine_binary_function(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    abs_x = abs_sobel_threshold(gray, orient='x', sobel_kernel=9, thresh=(20, 100))
    abs_y = abs_sobel_threshold(gray, orient='y', sobel_kernel=9, thresh=(20, 100))
    magnitude = mag_threshold(gray, sobel_kernel=9, mag_thresh=(50, 255))
    direction = dir_threshold(gray, sobel_kernel=9, thresh=(0.7, 1.3))
    gradient = np.zeros_like(gray)
    gradient[(abs_x == 1)] = 1
    HLS_binary = HLS_channel(image, channel='s', thresh=(220, 255))
    Lab_binary = Lab_channel(image, channel='b', thresh=(210, 255))
    RGB_binary = RGB_channel(image, channel='r', thresh=(210, 255))
    channel_binary = np.zeros_like(Lab_binary)
    channel_binary[(HLS_binary == 1) | (Lab_binary == 1) | (RGB_binary == 1)] = 1

    color_binary = np.dstack((np.zeros_like(gradient), gradient, channel_binary))
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(gradient)
    combined_binary[(channel_binary == 1) | (gradient == 1)] = 1
    return color_binary, combined_binary

# define the whole pipline
def pipeline(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    undistorted_image = undistortion_function(image, mtx, dist)
    warped_img, M, Minv = warp_image(undistorted_image, src, dst, (undistorted_image.shape[1], undistorted_image.shape[0]))
    color_binary, combined_binary = combine_binary_function(warped_img)
    return combined_binary, Minv, color_binary

# test the pipeline by using stright_line2.jpg
img = cv2.imread("test_images/straight_lines2.jpg")
combined_binary, _, _ = pipeline(img)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
ax1.set_title("Origin")
ax1.imshow(img)
ax1.axis("off")
ax2.set_title("Combined Warped")
ax2.imshow(combined_binary)
ax2.axis("off")

# step 5 Determine the curvature of the lane and vehicle position with respect to center.
def get_curvature(leftx, lefty, rightx, righty, ploty, image_size):
    y_eval = np.max(ploty)
    # convert pixel to meter
    ym_per_pixel = 25 / 720
    xm_per_pixel = 3.7 / 825
    # polynomial to fit the curve
    left_fit = np.polyfit(lefty * ym_per_pixel, leftx * xm_per_pixel, 2)
    right_fit = np.polyfit(righty * ym_per_pixel, rightx * xm_per_pixel, 2)

    # curvature radius calculation
    left_curvature = ((1 + (2 * left_fit[0] * y_eval * ym_per_pixel + left_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit[0])
    right_curverature = ((1 + (2 * right_fit[0] * y_eval * ym_per_pixel + right_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit[0])

# assuming the camera is in the center of the vehicle and calculate lan deviation from the center of lane.
    scene_height = image_size[0] * ym_per_pixel
    scene_width = image_size[1] * xm_per_pixel

    left_interception = left_fit[0] * scene_height ** 2 + left_fit[1] * scene_height + left_fit[2]
    right_interception = right_fit[0] * scene_height ** 2 + right_fit[1] * scene_height + right_fit[2]
    calculated_center = (left_interception + right_interception) / 2.0

    lane_deviation = (calculated_center - scene_width / 2.0)

    return left_curvature, right_curverature, lane_deviation

# define function to find and plot lane lines in iamges
def find_lane_lines(binary_warped, visual=False):
    if visual == True:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0] / 2):, :], axis=0)
    # By find the peak of the left and the halves of the histogram which are the starting points for the left and right lines
    middle_point = np.int(histogram.shape[0] / 2)
    left_x_base = np.argmax(histogram[:middle_point])
    right_x_base = np.argmax(histogram[middle_point:]) + middle_point

    nwindows = 7
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Find non-zero pixels and get the x,y position
    nonzero_pixel = binary_warped.nonzero()
    nonzeroy = np.array(nonzero_pixel[0])
    nonzero_x = np.array(nonzero_pixel[1])

    leftx_current = left_x_base
    rightx_current = right_x_base

# windows width, min pixels number to re-center window
    margin = 100
    minpix = 30
# empty lists to store left and right pixel indices
    left_lane_indices = []
    right_lane_indices = []

# one by one step the window
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if visual == True:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # get non zero pixels
        good_left_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzero_x >= win_xleft_low) & (
        nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzero_x >= win_xright_low) & (
        nonzero_x < win_xright_high)).nonzero()[0]

        # Append these indices in the lists
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)

        # re-center pixels on the next window if the pixels are more than min pixels
        if len(good_left_indices) > minpix:
            leftx_current = np.int(np.mean(nonzero_x[good_left_indices]))
        if len(good_right_indices) > minpix:
            rightx_current = np.int(np.mean(nonzero_x[good_right_indices]))

    # Concatenate the arrays of indices
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # get positions of the left and right line pixel
    left_x = nonzero_x[left_lane_indices]
    left_y = nonzeroy[left_lane_indices]
    right_x = nonzero_x[right_lane_indices]
    right_y = nonzeroy[right_lane_indices]

    # second order polynomial to fit the left and right lines
    left_line_fit = np.polyfit(left_y, left_x, 2)
    right_line_fit = np.polyfit(right_y, right_x, 2)

    # calculate x and y values and plot
    y_value = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fit_x = left_line_fit[0] * y_value ** 2 + left_line_fit[1] * y_value + left_line_fit[2]
    right_fit_x = right_line_fit[0] * y_value ** 2 + right_line_fit[1] * y_value + right_line_fit[2]

    left_curverad, right_curverad, lane_deviation = get_curvature(left_x, left_y, right_x, right_y, y_value,
                                                                  binary_warped.shape)

    if visual == True:
        out_img[nonzeroy[left_lane_indices], nonzero_x[left_lane_indices]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_indices], nonzero_x[right_lane_indices]] = [0, 0, 255]
        return left_fit_x, right_fit_x, y_value, left_line_fit, right_line_fit, left_curverad, right_curverad, lane_deviation, out_img
    else:
        return left_fit_x, right_fit_x, y_value, left_line_fit, right_line_fit, left_curverad, right_curverad, lane_deviation

# test the find lane lines function by using test1.jpg
img = cv2.imread("test_images/test1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

undistorted_image = undistortion_function(img, mtx, dist)
color_binary, combined_binary = combine_binary_function(undistorted_image)
combined_binary_warped, _, Minv = warp_image(combined_binary, src, dst, (img.shape[1], img.shape[0]))
left_fit_x, right_fit_x, plot_y, left_fit, right_fit, left_curverad, right_curverad, lane_deviation, out_img = find_lane_lines(
    combined_binary_warped, visual=True)

plt.figure(figsize=(16, 8))
plt.imshow(out_img)
plt.plot(left_fit_x, plot_y, color='yellow')
plt.plot(right_fit_x, plot_y, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.axis("off")
# save image
cv2.imwrite("output_images/bad_line_find.jpg", out_img)

# add other method to get better finding results. These methods have been defined previous: sobel x, sobel y,
# magnitude of gradient. direction threshold,color threshold
img = cv2.imread("test_images/test4.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

undistorted_image = undistortion_function(img, mtx, dist)
color_binary, combined_binary = combine_binary_function(undistorted_image)
combined_binary_warped, _, Minv = warp_image(combined_binary, src, dst, (img.shape[1], img.shape[0]))
left_fit_x, right_fit_x, plot_y, left_fit, right_fit, left_curverad, right_curverad, lane_deviation, out_img = find_lane_lines(combined_binary_warped, visual=True)

plt.figure(figsize=(16,8))
plt.imshow(out_img)
plt.plot(left_fit_x, plot_y, color='yellow')
plt.plot(right_fit_x, plot_y, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.axis("off")
# save image
cv2.imwrite("output_images/good_line_find.jpg", out_img)

# step 7 Warp the detected lane boundaries back onto the original image
def draw_lane_lines_on_image(binary_warped, undistorted_img, Minv, left_fitx, right_fitx, ploty, left_radius, right_radius,
                        lane_deviation):
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted_img.shape[1], undistorted_img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.4, 0)

    curvature_text = "Curvature: Left = " + str(np.round(left_radius, 2)) + ", Right = " + str(
        np.round(right_radius, 2))
    cv2.putText(result, curvature_text, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    deviation_text = "Lane deviation from center = {:.2f} m".format(lane_deviation)
    cv2.putText(result, deviation_text, (30, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    return result

result = draw_lane_lines_on_image(combined_binary_warped, undistorted_image, Minv, left_fit_x, right_fit_x, plot_y, left_curverad, right_curverad, lane_deviation)
plt.figure(figsize=(16,8))
plt.imshow(result)
plt.axis("off")
# save in file
binary = 255 * result.astype("uint8")
cv2.imwrite("output_images/final_result.jpg", result)

# define the image processing pipeline
def image_processing_pipline(image):
    undistorted_image = undistortion_function(image, mtx, dist)
    undistorted_image = cv2.GaussianBlur(undistorted_image, (5, 5), 0);
    color_binary, combined_binary = combine_binary_function(undistorted_image)
    combined_binary, _, Minv = warp_image(combined_binary, src, dst, (image.shape[1], image.shape[0]))

    left_fitx, right_fitx, ploty, _, _, left_curverad, right_curverad, lane_deviation = find_lane_lines(combined_binary)
    lane_lines_img = draw_lane_lines_on_image(combined_binary, undistorted_image, Minv, left_fitx, right_fitx, ploty,
                                         left_curverad, right_curverad, lane_deviation)
    return lane_lines_img

# run the image processing pipeline on the test_images
files = os.listdir('./test_images')
files = [ './test_images/{}'.format(file) for file in files ]
for file in files:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = image_processing_pipline(img)
    plt.figure(figsize=(20,10))
    plt.imshow(result)

# create videos
video_output = "project_video_output.mp4"
clip1 = VideoFileClip("project_video.mp4")
clip1_output = clip1.fl_image(image_processing_pipline)
clip1_output.write_videofile(video_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_output))

video_challenge_output = "video_challenge_output.mp4"
clip1 = VideoFileClip("challenge_video.mp4")
clip1_output = clip1.fl_image(image_processing_pipline)
clip1_output.write_videofile(video_challenge_output, audio=False)


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_challenge_output))














