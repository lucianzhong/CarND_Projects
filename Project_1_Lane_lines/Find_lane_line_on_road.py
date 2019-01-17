#importing  packages 
import matplotlib.pyplot as plt   # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np
import cv2
import math

# Read in images and Create image output folder
import os
os.listdir("test_images/")
dir_name_load = "test_images/"
dir_name_save = "test_images_output/"
if not os.path.exists(dir_name_save):
   os.mkdir(dir_name_save)


# Define the Grayscale transform
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Define the Canny transform
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)
# Define Gaussian Noise Kernel
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Define an ROI
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)    # the new matrix with the same shape as img and all initilize as zero
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


#  This function draws lines with color and thickness
def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # step 1: seprarate the left and right line's endpoints
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_line_x.append(x1)
                left_line_y.append(y1)
                left_line_x.append(x2)
                left_line_y.append(y2)
            else:
                right_line_x.append(x1)
                right_line_y.append(y1)
                right_line_x.append(x2)
                right_line_y.append(y2)

    # step 2: find parameters fitted the line and get the line's endpoints
    left_coef = np.polyfit(left_line_y, left_line_x, 1)
    right_coef = np.polyfit(right_line_y, right_line_x, 1)

    y_max = img.shape[0]
    y_min = min(left_line_y + right_line_y)

    left_polyvalue = np.poly1d(left_coef)
    right_polyvalue = np.poly1d(right_coef)

    left_x1 = int(left_polyvalue(y_max))
    left_x2 = int(left_polyvalue(y_min))
    right_x1 = int(right_polyvalue(y_max))
    right_x2 = int(right_polyvalue(y_min))

# Draw lines
    cv2.line(img, (left_x1, y_max), (left_x2, y_min), color, thickness)
    cv2.line(img, (right_x1, y_max), (right_x2, y_min), color, thickness)

# Define hough transform
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img



# Defined weighted image
# Initial_img * α + img * β + λ
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)



for file_name in os.listdir(dir_name_load):
    print(file_name)
    image = mpimg.imread(dir_name_load + file_name)

# Find lanes in images    
    # Step 1: grayscale the image
    gray = grayscale(image)
    plt.figure(1)
    plt.imshow(gray, cmap='gray')   #cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间
    plt.imsave(dir_name_save + 'step1.jpg', gray)    
    # Step 2;Gaussian Smoothing
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)
    plt.figure(2)
    plt.imshow(blur_gray, cmap='gray')
    plt.imsave(dir_name_save + 'step2.jpg', blur_gray)    
    # Step 3: Canny Edge Detection
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    plt.figure(3)
    plt.imshow(edges)
    plt.imsave(dir_name_save + 'step3.jpg', edges)    
    # Step 4: select interest region
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    # Mask the image with a four sided polygon
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (470, 320), (490, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    plt.figure(4)
    plt.imshow(masked_edges)
    plt.imsave(dir_name_save + 'step4.jpg', masked_edges)    
    # Step 5: Hough Tranform line detection
    rho = 2
    theta = np.pi / 180
    threshold = 50
    min_line_len = 25
    max_line_gap = 25
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    plt.figure(5)
    plt.imshow(line_image)
    plt.imsave(dir_name_save + 'step5.jpg', line_image)
    # Step 6: weighted images
    lines_edges = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
    plt.figure(6)
    plt.imshow(lines_edges)
    #plt.show()
    plt.imsave(dir_name_save + file_name, lines_edges)
    # save the outputs
    plt.imsave(dir_name_save + file_name, lines_edges)


# process the video
def process_image(image):
    # Step 1: grayscale the image
    gray = grayscale(image)
    # Step 2: Gaussian smoothing
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)
    # Step 3: Canny Edge Detection
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    # Step 4: select interest region
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    # Mask the image with a four sided polygon
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (470, 320), (490, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    # Step 5: Hough Tranform line detection
    rho = 2
    theta = np.pi / 180
    threshold = 50
    min_line_len = 25
    max_line_gap = 25
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    # Step 6: merge lines
    index = (line_image[:, :, 0] == 255)
    # Step 7: weighted images
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
    return result


# Import packages which are used to edit videos
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# Create video output folder
isExists=os.path.exists("test_videos_output")
if not isExists:
    os.mkdir("test_videos_output")

# Find lanes in solidWhiteRight.mp4
white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

# Find lanes in solidYellowLeft.mp4
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

# Find lanes in challenge.mp4
challenge_output = 'test_videos_output/challenge.mp4'
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)

