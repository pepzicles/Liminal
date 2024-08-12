#!/usr/bin/env python
# coding: utf-8

# # This code constantly generates an gif/ video output out of the images saved
# You'd need to run the run_notebook.py script in terminal to generate a new updated gif/ video every 10 mins 

# # Tracking buoys

pip install buoyant
from buoyant import Buoy


# # Get Image URLs

## NEW YORK HARBOUR ENTRANCE BUOY: STATION 44065
## https://www.ndbc.noaa.gov/station_page.php?station=44065
ny_station = Buoy(44065)
print(ny_station.image_url)

## EAST SANTA BARBARA: STATION 46053
## https://www.ndbc.noaa.gov/station_page.php?station=46053
sb_station = Buoy(46053)
print(sb_station.image_url)

## SAN DIEGO, TANNER BANK: STATION 46047
## https://www.ndbc.noaa.gov/station_page.php?station=46047
sd_station = Buoy(46047)
print(sd_station.image_url)


# Get current time and date
import os
from datetime import datetime, timedelta

current_datetime = datetime.now()
print("Current Date and Time:", current_datetime)

current_date = current_datetime.date()
print(current_date)

current_time = current_datetime.strftime('%H:%M')
print(current_time)


# Create a new folder every day a new day starts

def create_folder_everyday(base_path):
    current_date = datetime.now().date()
    current_folder_path = os.path.join(base_path, str(current_date))
    
    #if folder doesnt exist yet
    if not os.path.exists(current_folder_path):
        os.makedirs(current_folder_path)
        print(f"New folder created: {current_folder_path}")
        
    return current_folder_path


## Creating new folder every day for each buoy
base_path_ny = "downloadedImages/NY/raw_images"
base_path_sb = "downloadedImages/SB/raw_images"
base_path_sd = "downloadedImages/SD/raw_images"

#Call function
newFolder_ny = create_folder_everyday(base_path_ny)
newFolder_sb = create_folder_everyday(base_path_sb)
newFolder_sd = create_folder_everyday(base_path_sd)


# # Save images
import urllib.request
import os

# Get the current working directory
current_directory = os.getcwd()
print(f"The current working directory is: {current_directory}")



## NEW YORK HARBOUR ENTRANCE BUOY: STATION 44065

#Save image as a file as {time}.jpg in the date folder
local_filename_ny = f"{current_time}"
folder_path_ny = f"{newFolder_ny}"

print(folder_path_ny)

# Specify the local filename for saving the image
local_filename_ny = os.path.join(folder_path_ny, f"{local_filename_ny}.jpg")

try:
    # Download the image
    urllib.request.urlretrieve(ny_station.image_url, local_filename_ny)
    print(f"Image saved as {local_filename_ny}")
except Exception as e:
    print(f"Error downloading image: {e}")



pip install Pillow




# Show image

from PIL import Image

image = Image.open(local_filename_ny)

image


# In[331]:


## EAST SANTA BARBARA: STATION 46053

# #Save image as a file as {time}.jpg in the date folder
# local_filename_sb = f"{current_time}"
# folder_path_sb = f"{newFolder_sb}"

# print(folder_path_sb)

# # Specify the local filename for saving the image
# local_filename_sb = os.path.join(folder_path_sb, f"{local_filename_sb}.jpg")

# try:
#     # Download the image
#     urllib.request.urlretrieve(sb_station.image_url, local_filename_sb)
#     print(f"Image saved as {local_filename_sb}")
# except Exception as e:
#     print(f"Error downloading image: {e}")


# In[332]:


## SAN DIEGO, TANNER BANK: STATION 46047

# #Save image as a file as {time}.jpg in the date folder
# local_filename_sd = f"{current_time}"
# folder_path_sd = f"{newFolder_sb}"

# print(folder_path_sd)

# # Specify the local filename for saving the image
# local_filename_sd = os.path.join(folder_path_sd, f"{local_filename_sd}.jpg")

# try:
#     # Download the image
#     urllib.request.urlretrieve(sd_station.image_url, local_filename_sd)
#     print(f"Image saved as {local_filename_sd}")
# except Exception as e:
#     print(f"Error downloading image: {e}")



import sys
print(sys.version)


# # Cropping the images
# https://youtu.be/7IL7LKSLb9I

pip install patchify


import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
from PIL import Image


# First, cut the bottom descriptive strip away (2880px x 31px)


#Get dimensions of image
width, height = image.size

bottomStripHeight = 31
topStripHeight = height - bottomStripHeight

#Crop image to keep the top portion
topPortion = image.crop((0, 0, width, topStripHeight))
topPortion


# Then, cut the remaining picture into 6 images


newWidth, newHeight = topPortion.size

# Calculate the width of each part
part_newWidth = width // 6

part_newWidth

save_folder_path = "downloadedImages/NY/processed_images"

#Crop image into 6 parts
for i in range(6):
    left = i * part_newWidth
    upper = 0
    right = (i + 1) * part_newWidth
    lower = newHeight
    
    part = topPortion.crop((left, upper, right, lower))
    part.save(os.path.join(save_folder_path, f"{current_date}|{current_time}|{i+1}.jpg"))
    
    display(part)


# # Automate ROI


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
from PIL import Image


# Directory paths
input_folder = "downloadedImages/NY/processed_images"
detectedROI_folder = "downloadedImages/NY/processed_images/detectedROI"
cropped_folder = "downloadedImages/NY/processed_images/cropped_images"
detectedLines_folder = "downloadedImages/NY/processed_images/cropped_images/detectedLines_images"
superimposed_folder = "downloadedImages/NY/processed_images/cropped_images/detectedLines_images/superimposedROI_onOriginal"
original_ROI_lineDetection_folder = "downloadedImages/NY/processed_images/cropped_images/detectedLines_images/superimposedROI_onOriginal/original_ROI_lineDetection"
rotated_images_folder = "downloadedImages/NY/processed_images/cropped_images/detectedLines_images/superimposedROI_onOriginal/original_ROI_lineDetection/rotatedImages"
rotated_images_line_folder = "downloadedImages/NY/processed_images/cropped_images/detectedLines_images/superimposedROI_onOriginal/original_ROI_lineDetection/rotatedImages_wLineDetection"
expanded_images_folder = "downloadedImages/NY/processed_images/cropped_images/detectedLines_images/superimposedROI_onOriginal/original_ROI_lineDetection/rotatedImages/expandedImages"
expanded_images_line_folder = "downloadedImages/NY/processed_images/cropped_images/detectedLines_images/superimposedROI_onOriginal/original_ROI_lineDetection/rotatedImages_wLineDetection/expandedImages_wLineDetection"
shifted_images_folder = "downloadedImages/NY/processed_images/cropped_images/detectedLines_images/superimposedROI_onOriginal/original_ROI_lineDetection/rotatedImages/expandedImages/shiftedImages"
shifted_images_line_folder = "downloadedImages/NY/processed_images/cropped_images/detectedLines_images/superimposedROI_onOriginal/original_ROI_lineDetection/rotatedImages_wLineDetection/expandedImages_wLineDetection/shiftedImages_lineDetection"
shiftedExpanded_folder = "downloadedImages/NY/processed_images/cropped_images/detectedLines_images/superimposedROI_onOriginal/original_ROI_lineDetection/rotatedImages/expandedImages/shiftedImages/expandedShifted"
shiftedExpanded_line_folder = "downloadedImages/NY/processed_images/cropped_images/detectedLines_images/superimposedROI_onOriginal/original_ROI_lineDetection/rotatedImages_wLineDetection/expandedImages_wLineDetection/shiftedImages_lineDetection/expandedShifted_lineDetection"
output_folder = "downloadedImages/NY/processed_images/rotated_images"
finalOutputs_folder = "downloadedImages/NY/processed_images/cropped_images/detectedLines_images/superimposedROI_onOriginal/original_ROI_lineDetection/rotatedImages/expandedImages/shiftedImages/expandedShifted/finalOutputs"


# Iterate through the images

#load images

# Iterate over each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Read the image
        image = cv2.imread(os.path.join(input_folder, filename))
        
        # Display the image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(filename)
#         plt.axis('off')
        plt.show()


# Function to find ROI for the images
def automateROI(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range of blue color in HSV
    # HSV Values (Hue, Saturation, Value) 
    lower_blue = np.array([50, 10, 10])
    upper_blue = np.array([150, 255, 150])
    
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Bitwise-AND mask and original image
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Convert segmented image to grayscale
    gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_segmented, (5, 5), 0)
    
    # Perform edge detection using the Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Perform image dilation to close gaps in edges
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(dilated_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return None
    if not contours:
        print(f"No contours/ no ROI found for {filename}")
        print("")
        return None
    
    if contours:
        print(f"ROI Found for {filename}")

    # Find the contour with the maximum area (presumably the horizon line)
    max_contour = max(contours, key=cv2.contourArea)

    # Get bounding box of the contour
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # Increase the height of the rectangle by 20 pixels
    h += 20
    
    # Create a rectangle covering the entire width of the image and with the height of the bounding box
    roi_rectangle = np.array([[0, y], [image.shape[1], y], [image.shape[1], y + h], [0, y + h]])

    # Draw the rectangle on the original image
    result_image = cv2.drawContours(image.copy(), [roi_rectangle], -1, (0, 255, 0), 2)
    
    return result_image, y, h


roi_coordinates = {}


for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Read the image
        image = cv2.imread(os.path.join(input_folder, filename))
        
        #Then, automateROI
        result = automateROI(image)
        
        if result is not None:
            detectedROI_image, y, h = result

            # Save images into detectedROI_folder folder
            output_path = os.path.join(detectedROI_folder, filename)
            cv2.imwrite(output_path, detectedROI_image)
            print(f"Image with ROI saved to: {output_path}")
            print("")
            
            # Crop images with respective ROI
            cropped_image = image[y:y+h, 0:image.shape[1]]
        
            # Save cropped images into cropped_folder folder
            cropped_output_path = os.path.join(cropped_folder, filename)
            cv2.imwrite(cropped_output_path, cropped_image)
            print(f"Cropped image saved to: {cropped_output_path}")
            print("")
            print("-------")
            
            print("y: " + str(y))
            print("h: " + str(h))
             # Store ROI coordinates in the roi_coordinates dictionary
            roi_coordinates[filename]= [y, h]
            
        else:
#           print(f"No ROI detected for {filename}")
            print("")
            print("-------")


print(roi_coordinates)


# Show images that have ROI detected

# Iterate over each image in the detectedROI_folder folder
for filename in os.listdir(detectedROI_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Read the image
        image = cv2.imread(os.path.join(detectedROI_folder, filename))
        
        # Display the image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(filename)
        plt.show()


# Show cropped images

# Iterate over each image in the cropped_folder
for filename in os.listdir(cropped_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Read the image
        image = cv2.imread(os.path.join(cropped_folder, filename))
        
        # Display the image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(filename)
        plt.show()


# # Detect lines using Hough Line Transform

# Now, apply Hough Transform to all of the cropped (ROI) images in the cropped_images folder
# 
# Note: Change threshold accordingly

for filename in os.listdir(cropped_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        
        #Read image
        image = cv2.imread(os.path.join(cropped_folder, filename))
        
        #Convert to gray 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
        
        # Display the gray images
        plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


# Creating gradient array to store all the gradients
import shutil

# List to store filenames of processed images with lines detected
images_w_detectedLines = []
gradients = []


# Creating a function to detect Hough Transform Lines
def detect_and_visualize_lines(image_path):
    # Read image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    # Check if any lines are detected
    if lines is None:
        print(f"No lines detected in {image_path}. Moving on to the next image.")
        return
    
    # Find the strongest line (longest) among the detected lines
    # to ensure that there is only ONE line that shows up
    longest_line = None
    max_length = 0
    
    
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length > max_length:
            max_length = length
            longest_line = line
    
    # Draw the strongest line on the image
    rho, theta = longest_line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Calculate gradient and append to gradients list
    gradient = -np.cos(theta) / np.sin(theta)
    print(f"Gradient = {gradient}")
    gradients.append(gradient)
    
    # Save the image with the detected line
    output_path = os.path.join(detectedLines_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Image with detected line saved to: {output_path}")
    
    
    # Display the image with detected lines
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Hough Line Detection")
    plt.axis('off')
    plt.show()
    
    return gradients


# Calling the function to act on all the images in the cropped_folder
# 
# Note: Threshold depends on each image

# In[350]:


for filename in os.listdir(cropped_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        image_path = os.path.join(cropped_folder, filename)
        detect_and_visualize_lines(image_path)


# # Impose the new detected line on to the original images (before cropping ROI)

# Now, iterate over each image in the detectedLines_images folder
for filename in os.listdir(detectedLines_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Read the image with detected lines
        detected_lines_image = cv2.imread(os.path.join(detectedLines_folder, filename))
        
        # Check if the filename has corresponding ROI coordinates
        if filename in roi_coordinates:
            # Read the corresponding original image from the processed_images folder
            original_image = cv2.imread(os.path.join(input_folder, filename))  # Change to processed_images_folder if necessary
            
            # Get the ROI coordinates from the dictionary
            y, h = roi_coordinates[filename]
            
            # Superimpose the cropped image onto the original image at the determined position
            original_image[y:y+h, 0:detected_lines_image.shape[1]] = detected_lines_image
            
            # Save and display the superimposed image
            superimposed_output_path = os.path.join(superimposed_folder, filename)
            cv2.imwrite(superimposed_output_path, original_image)
            print(f"Superimposed image saved to: {superimposed_output_path}")
            
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            plt.title("Superimposed Image")
            plt.axis('off')
            plt.show()
        else:
            print(f"No ROI coordinates found for {filename}. Skipping.")


# # Get original images from input_folder that correspond to the images that successfully went through ROI and Hough Line Detection

# All the images that have lines and ROI detected
for filename in os.listdir(detectedLines_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        images_w_detectedLines.append(filename)

print(images_w_detectedLines)


# Save these images into original_ROI_lineDetection folder

for filename in images_w_detectedLines:
    original_image_path = os.path.join(input_folder, filename)
    if os.path.exists(original_image_path):
        shutil.copy(original_image_path, original_ROI_lineDetection_folder)
        print(f"Original image copied to: {original_ROI_lineDetection_folder}/{filename}")
    else:
        print(f"Original image not found for {filename}")


# Okay great. so now, I have all the original images with ROI AND Line detected saved in original_ROI_lineDetection_folder

# # Get corresponding gradients of these images that successfully went through ROI and Hough Line Detection

#All the gradients of all the images that have lines detected
print(gradients)


# # Now, rotate each image according to its respective gradient
# Iterate over each image and its respective gradient
for filename, gradient in zip(images_w_detectedLines, gradients):
    # Read the image
    image_path = os.path.join(original_ROI_lineDetection_folder, filename)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to read {image_path}")
        continue
    
    # Calculate the rotation angle (in degrees) from the negative gradient
    rotation_angle_degrees = np.degrees(np.arctan(gradient))

    # Rotate the image
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle_degrees, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))

    # Save the rotated image
    rotated_image_path = os.path.join(rotated_images_folder, f"rotated_{filename}")
    cv2.imwrite(rotated_image_path, rotated_image)
    print(f"Rotated image saved to: {rotated_image_path}")

    # Display the rotated image
    plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# # Save rotated images with line detection on the image

# Iterate over each image and its respective gradient
for filename, gradient in zip(images_w_detectedLines, gradients):
    # Read the image
    image_path = os.path.join(superimposed_folder, filename)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to read {image_path}")
        continue
    
    # Calculate the rotation angle (in degrees) from the negative gradient
    rotation_angle_degrees = np.degrees(np.arctan(gradient))

    # Rotate the image
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle_degrees, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))

    # Save the rotated image
    rotated_image_path = os.path.join(rotated_images_line_folder, f"rotated_{filename}")
    cv2.imwrite(rotated_image_path, rotated_image)
    print(f"Rotated image saved to: {rotated_image_path}")

    # Display the rotated image
    plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# # Expand all these images to get rid of the black edges - without line detection


for filename in os.listdir(rotated_images_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Read the rotated image
        image_path = os.path.join(rotated_images_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Unable to read {image_path}")
            continue

        # Display the rotated image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


# In[384]:


for filename in os.listdir(rotated_images_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Read the rotated image
        image_path = os.path.join(rotated_images_folder, filename)
        image = cv2.imread(image_path)

        # Get original dimensions
        original_height, original_width = image.shape[:2]

        # Define the desired dimensions for the final image
        final_height = original_height
        final_width = original_width

        # Calculate scaling factors to make the image bigger
        scale_factor = 1.2  # You can adjust this value according to your needs

        # Resize the image while maintaining the aspect ratio
        resized_width = int(original_width * scale_factor)
        resized_height = int(original_height * scale_factor)
        resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        # Calculate the cropping dimensions based on the desired final size
        crop_start_x = (resized_width - final_width) // 2
        crop_end_x = crop_start_x + final_width
        crop_start_y = (resized_height - final_height) // 2
        crop_end_y = crop_start_y + final_height

        # Perform the cropping directly from the resized image
        cropped_image = resized_image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

        # Display original and cropped images inline
        plt.figure(figsize=(10, 10))

        # Original Image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        plt.title('Resized Image')
        plt.axis('off')

        # Cropped Image
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title('Expanded Image')
        plt.axis('off')

        plt.show()
        
        
        #Save them into folder
        output_path = os.path.join(expanded_images_folder, filename)
        cv2.imwrite(output_path, cropped_image)
        print(f"Cropped image saved to: {output_path}")


# In[385]:


#Now, show all the cropped and expanded images!!
for filename in os.listdir(expanded_images_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Read the rotated image
        image_path = os.path.join(expanded_images_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Unable to read {image_path}")
            continue

        # Display the rotated image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


# # Expand the images to get rid of black edges with line detection

# In[416]:


for filename in os.listdir(rotated_images_line_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Read the rotated image
        image_path = os.path.join(rotated_images_line_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Unable to read {image_path}")
            continue
            
         # Get original dimensions
        original_height, original_width = image.shape[:2]

        # Define the desired dimensions for the final image
        final_height = original_height
        final_width = original_width

        # Calculate scaling factors to make the image bigger
        scale_factor = 1.2  # You can adjust this value according to your needs

        # Resize the image while maintaining the aspect ratio
        resized_width = int(original_width * scale_factor)
        resized_height = int(original_height * scale_factor)
        resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        # Calculate the cropping dimensions based on the desired final size
        crop_start_x = (resized_width - final_width) // 2
        crop_end_x = crop_start_x + final_width
        crop_start_y = (resized_height - final_height) // 2
        crop_end_y = crop_start_y + final_height

        # Perform the cropping directly from the resized image
        cropped_image = resized_image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

        # Display original and cropped images inline
        plt.figure(figsize=(10, 10))

        # Original Image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        plt.title('Resized Image')
        plt.axis('off')

        # Cropped Image
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title('Expanded Image')
        plt.axis('off')

        plt.show()
        
        
        #Save them into folder
        output_path = os.path.join(expanded_images_line_folder, filename)
        cv2.imwrite(output_path, cropped_image)
        print(f"Cropped image saved to: {output_path}")


# # Perform Hough Transform again to find the y height of all the lines of all the expanded images (with line detection first, to better determine where the line is)

# In[417]:


#Create an array of all the y-coordinates of all the lines of the expanded images

y_coordinate_of_lines = []


# In[418]:


for filename in os.listdir(expanded_images_line_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Read the rotated image
        image_path = os.path.join(expanded_images_line_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Unable to read {image_path}")
            continue
            
         # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Apply Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        # Check if any lines are detected
        if lines is None:
            print(f"No lines detected in {image_path}. Moving on to the next image.")

        # Find the strongest line (longest) among the detected lines
        # to ensure that there is only ONE line that shows up
        longest_line = None
        max_length = 0


        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > max_length:
                max_length = length
                longest_line = line
                
                
             # Calculate y-coordinate of the line (can be either y1 or y2)
            y_coordinate_of_line = int((y1 + y2) / 2)  # Choose the appropriate y-coordinate
            
            # Draw the detected line on the expanded image
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw line in green

            
        #Find the height of each line, and append to the array
        print("y-coordinate of line: " + str(y_coordinate_of_line))
        y_coordinate_of_lines.append(y_coordinate_of_line)
        
        # Display the image with detected lines
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Hough Line Detection")
        plt.axis('off')
        plt.show()
        


# In[419]:


print(y_coordinate_of_lines)


# In[420]:


# Find the average of all the y-coordinates
sum_of_heights = sum(y_coordinate_of_lines)
average = sum_of_heights/ len(y_coordinate_of_lines)

print("Sum of heights: ", sum_of_heights)
print("average of numbers: ", average)


# # Move all the expanded images WITHOUT line detection such that all of their line detected is at the average height

# In[421]:


# Initialize an array to store shifts needed for each image

shifts_needed = []

# Initialize a counter for the number of images
num_images = 0

# Count the number of image files (JPEG or JPG)
total_images = sum(1 for file in files if file.lower().endswith(('.jpg', '.jpeg')))
print("total_images: ", total_images)
print()

for filename in os.listdir(expanded_images_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Read the rotated image
        image_path = os.path.join(expanded_images_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Unable to read {image_path}")
            continue
        
        num_images += 1
        
        # Calculate the shift needed
        shift_needed = y_coordinate_of_lines[num_images-1] - average

        # Append the shift needed to the list
        shifts_needed.append(shift_needed)

        print(f"Height of line in '{filename}': {y_coordinate_of_lines[num]}")
        print(f"Shift needed: {shift_needed}")
        print()

        #Okay, now move each image up and down according to the shift_needed
        shifted_image = np.roll(image, int(-shift_needed), axis=0)

        # Display the image with detected lines
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Shifted Image")
        plt.axis('off')
        plt.show()

        #Save them into folder
        output_path = os.path.join(shifted_images_folder, filename)
        cv2.imwrite(output_path, shifted_image)
        print(f"Shifted image saved to: {output_path}")
        print()


# # Okay, I have to expand the images, such that there is no gap at the top/ bottom of the images after moving them up/ down, while still keeping the line at exactly the same level
# - Expanding all the images by the same amount

# In[422]:


for filename in os.listdir(shifted_images_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Read the rotated image
        image_path = os.path.join(shifted_images_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Unable to read {image_path}")
            continue
            
         # Get original dimensions
        original_height, original_width = image.shape[:2]

        # Define the desired dimensions for the final image
        final_height = original_height
        final_width = original_width

        # Calculate scaling factors to make the image bigger
        scale_factor = 1.2  # You can adjust this value according to your needs

        # Resize the image while maintaining the aspect ratio
        resized_width = int(original_width * scale_factor)
        resized_height = int(original_height * scale_factor)
        resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        # Calculate the cropping dimensions based on the desired final size
        crop_start_x = (resized_width - final_width) // 2
        crop_end_x = crop_start_x + final_width
        crop_start_y = (resized_height - final_height) // 2
        crop_end_y = crop_start_y + final_height

        # Perform the cropping directly from the resized image
        cropped_image = resized_image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

        # Display original and cropped images inline
        plt.figure(figsize=(10, 10))

        # Original Image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        plt.title('Resized Image')
        plt.axis('off')

        # Cropped Image
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title('Expanded Image')
        plt.axis('off')

        plt.show()
        
        
        #Save them into folder
        output_path = os.path.join(shiftedExpanded_folder, filename)
        cv2.imwrite(output_path, cropped_image)
        print(f"Cropped image saved to: {output_path}")


# # Trying to animate the images #1: Gif
# Images to fade into one another

# In[463]:


import imageio


# In[464]:


#Create list to store image paths
image_paths = [os.path.join(shiftedExpanded_folder, file) for file in os.listdir(expanded_images_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

image_paths


# In[465]:


# Sort the image paths to ensure proper order
image_paths.sort()

images_list = []

for path in image_paths:
    img = Image.open(path)
    images_list.append(np.array(img))

print(images_list)


# In[466]:


len(images_list)


# In[467]:


# duration = 1
# num_frames = int(len(images_list)/duration)

# We want 6 frames per 10 mins = 600s
# That means 1 frame per 1.67 min = 100.2 seconds
# Number of frames = the number of images in the list

num_frames = len(images_list)
print(num_frames)


# In[468]:


faded_images = []

for i in range(len(images_list)-1): #iterate through range of images
    for j in range(num_frames):
        alpha = j / num_frames
        blended_frame = (1 - alpha) * images_list[i] + alpha * images_list[i+1]
        faded_images.append(blended_frame.astype(np.uint8))


# In[469]:


output_path = f"{finalOutputs_folder}/ny_gif.gif"

imageio.mimsave(output_path, faded_images, duration = num_frames * 20)


# In[470]:


# Display gif created
# Show the frames
gif = imageio.mimread(output_path)

for frame in gif:
    plt.imshow(frame)
    plt.show()


# # Create Video

# In[471]:


pip install 'imageio[ffmpeg]'


#Num of frames = number of images in the images_list

#6 frames every 10min = 600s
#0.01 frames per 1s
#Currently 18 images > 30mins
# 1 image = 1.67min

videoDuration = num_frames * 10 #in seconds
fps = 1

numFramesVideo = int(videoDuration * fps)


# In[473]:


faded_images_video = []
for i in range(len(images_list) - 1):
    for j in range(numFramesVideo):
        alpha = j / numFramesVideo
        blended_frame = (1 - alpha) * images_list[i] + alpha * images_list[i + 1]
        faded_images_video.append(blended_frame.astype(np.uint8))


# In[474]:


#save animation video
output_path_video = f"{finalOutputs_folder}/ny_video.mp4"

writer = imageio.get_writer(output_path_video, fps=10, codec='libx264', quality=8)
for frame in faded_images_video:
    writer.append_data(frame)
writer.close()


# In[475]:


# Display the video in the Jupyter Notebook cell

from IPython.display import display, Video

video = Video(output_path_video, embed=True)
display(video)


# # Attempting to create a video with frames of lowered opacity (17%) 
# Each image has opacity of 17% and layered on top of one another
# - Image 1 fade in from 0 to 17% opacity, for 1.67 min
# - Image 2 fade in from 0 to 17% opacity ON TOP of image 1, for the next 1.67 min
# - Image 3 fade in from 0 to 17% opacity ON TOP OF image 1 and 2, for the next 1.67 min
# 
# and so on and so forth

# In[476]:


pip install 'imageio[ffmpeg]'


# In[477]:


import numpy as np
import imageio


# In[478]:


videoDuration = num_frames * 5  # in seconds
fade_duration = 1.67 * 5  # in seconds
fps = 10
numFramesVideo = int(fps * fade_duration)
faded_images_video = []


# In[479]:


# Initialize the composite image as a blank canvas
composite_image = np.zeros_like(images_list[0])


# In[480]:


# Loop through each image in the list
for i, image in enumerate(images_list):
    # Calculate the start and end frame indices for fading in each image
    start_frame = i * numFramesVideo
    end_frame = start_frame + numFramesVideo
    
    # Fading in each image over specified duration
    for j in range(start_frame, end_frame):
        # Calculate alpha for the current frame to achieve a gradual fade
        alpha = ((j - start_frame) / numFramesVideo) * 0.17  # Gradual fade from 0 to 17% opacity
        # Blend the current image with the composite image
        blended_frame = (1 - alpha) * composite_image + alpha * image
        # Append the blended frame to the list
        faded_images_video.append(blended_frame.astype(np.uint8))

    # Update the composite image by stacking the current image on top
    composite_image = (1 - alpha) * composite_image + alpha * image


# In[481]:


# Write the blended frames to a video file
output_path_video = f"{finalOutputs_folder}/17%opacity.mp4"
writer = imageio.get_writer(output_path_video, fps=fps, codec='libx264', quality=8)
for frame in faded_images_video:
    writer.append_data(frame)
writer.close()


# In[482]:


# Display the video
from IPython.display import display, Video
video = Video(output_path_video, embed=True)
display(video)


# # Trying with opacity = 10% instead
# - No kicking out of any images

pip install 'imageio[ffmpeg]'


# In[484]:


import numpy as np
import imageio


# In[485]:


videoDuration = num_frames * 5  # in seconds
fade_duration = 1.67 * 5  # in seconds
fps = 10
numFramesVideo = int(fps * fade_duration)
faded_images_video = []


# In[486]:


# Initialize the composite image as a blank canvas
composite_image = np.zeros_like(images_list[0])


# In[487]:


# Loop through each image in the list
for i, image in enumerate(images_list):
    # Calculate the start and end frame indices for fading in each image
    start_frame = i * numFramesVideo
    end_frame = start_frame + numFramesVideo
    
    # Fading in each image over specified duration
    for j in range(start_frame, end_frame):
        # Calculate alpha for the current frame to achieve a gradual fade
        alpha = ((j - start_frame) / numFramesVideo) * 0.10  # Gradual fade from 0 to 10% opacity
        # Blend the current image with the composite image
        blended_frame = (1 - alpha) * composite_image + alpha * image
        # Append the blended frame to the list
        faded_images_video.append(blended_frame.astype(np.uint8))

    # Update the composite image by stacking the current image on top
    composite_image = (1 - alpha) * composite_image + alpha * image


# In[488]:


# Write the blended frames to a video file
output_path_video = f"{finalOutputs_folder}/10%.mp4"
writer = imageio.get_writer(output_path_video, fps=fps, codec='libx264', quality=8)
for frame in faded_images_video:
    writer.append_data(frame)
writer.close()


# In[489]:


# Display the video
from IPython.display import display, Video
video = Video(output_path_video, embed=True)
display(video)


# # Attempting to create a video with frames of lowered opacity - Only 6 layers at one time.
# 
# - Image 1 fade in from 0 to 17% opacity, for 1.67 min
# - Image 2 fade in from 0 to 17% opacity ON TOP of image 1, for the next 1.67 min
# - Image 3 fade in from 0 to 17% opacity ON TOP OF image 1 and 2, for the next 1.67 min
# - Image 4  fade in from 0 to 17% opacity ON TOP OF image 1,2,3, for the next 1.67 min
# - Image 5 fade in from 0 to 17% opacity ON TOP OF image 1,2,3,4, for the next 1.67 min
# - Image 6  fade in from 0 to 17% opacity ON TOP OF image 1,2,3,4,5 for the next 1.67 min
# - Image 7  fade in from 0 to 17% opacity ON TOP OF ONLY images 2,3,4,5,6 for the next 1.67 min
# - Image 8 fade  in from 0 to 17% opacity ON TOP OF ONLY images 3,4,5,6,7 for the next 1.67 min
# and so on
# 
# So essentially, only 6 images layered on top of one another at one time. 
# When the 7th image comes about, kick the 1st image out 
# When the 8th image comes about, kick the 1st and 2nd image out 
# etc

# In[490]:


pip install 'imageio[ffmpeg]'


# In[491]:


pip install --upgrade imageio[ffmpeg]


import numpy as np
import imageio

videoDuration = num_frames * 5  # in seconds
fade_duration = 1.67 * 5  # in seconds
fps = 10
numFramesVideo = int(fps * fade_duration)
faded_images_video = []


# Initialize variables to track the composite image and the images to be blended together
composite_images = [np.zeros_like(images_list[0])] * 6

# Loop through each image in the list
for i, image in enumerate(images_list):
    # Calculate the start and end frame indices for fading in each image
    start_frame = i * numFramesVideo
    end_frame = start_frame + numFramesVideo
    
    # Fading in each image over specified duration
    for j in range(start_frame, end_frame):
        # Calculate alpha for the current frame to achieve a gradual fade
        alpha = ((j - start_frame) / numFramesVideo) * 0.17  # Gradual fade from 0 to 17% opacity
        
        # Blend the current image with the composite image stack
        blended_frame = np.zeros_like(composite_images[0], dtype=np.float32)
        for img in composite_images:
            blended_frame += img.astype(np.float32)
        blended_frame = (1 - alpha) * blended_frame + alpha * image.astype(np.float32)
        
        # Append the blended frame to the list after converting back to uint8
        faded_images_video.append(blended_frame.astype(np.uint8))

    # Update the composite image stack by removing the oldest image and adding the current image
    composite_images.pop(0)
    composite_images.append(image)


# Write the blended frames to a video file
output_path_video = f"{finalOutputs_folder}/17%opacity_6Frames.mp4"
writer = imageio.get_writer(output_path_video, fps=fps, codec='libx264', quality=8)
for frame in faded_images_video:
    writer.append_data(frame)
writer.close()


# Display the video
from IPython.display import display, Video
video = Video(output_path_video, embed=True)
display(video)


# # Second attempt at creating a video with only 6 frames at each time. 
# Blend the first 6 images first.
# When it comes to the next set of 6, 

# In[498]:


pip install 'imageio[ffmpeg]'


# In[499]:


import numpy as np
import imageio


# In[500]:


video_duration = 30 * 60  # 30 minutes in seconds
fade_duration = 1.67 * 60  # 1.67 minutes in seconds
fps = 30
num_frames_per_image = int(fps * fade_duration)
total_frames = int(fps * video_duration)
faded_images_video = []


# In[501]:


# Initialize variables to track the composite image and the images to be blended together
composite_images = []


# In[ ]:


# Loop through each frame
for frame_index in range(total_frames):
    # Determine the index of the image to use based on the current frame
    image_index = frame_index // num_frames_per_image
    
    # If the image_index is greater than 5, start kicking out the first images
    if image_index >= 6:
        composite_images.pop(0)
    
    # Blend the appropriate set of images to form the current frame
    blended_frame = np.zeros_like(images_list[0], dtype=np.float32)
    for i, img in enumerate(images_list[max(0, image_index - 5):image_index + 1]):
        alpha = min(1, ((frame_index % num_frames_per_image) / num_frames_per_image) * (i + 1) * 0.17)
        blended_frame += alpha * img.astype(np.float32)
    
    # Append the blended frame to the list after converting back to uint8
    faded_images_video.append(np.clip(blended_frame, 0, 255).astype(np.uint8))
    
    # If the image_index is less than the total number of images, add the current image to the composite stack
    if image_index < len(images_list):
        composite_images.append(images_list[image_index])


# In[ ]:


# Write the blended frames to a video file
output_path_video = f"{finalOutputs_folder}/17%opacity_6FramesV2.mp4"
writer = imageio.get_writer(output_path_video, fps=fps, codec='libx264', quality=8)
for frame in faded_images_video:
    writer.append_data(frame)
writer.close()


# In[ ]:


# Display the video
from IPython.display import display, Video
video = Video(output_path_video, embed=True)
display(video)

