import time
from flask import Flask, render_template
from buoyant import Buoy
from datetime import datetime
import os
import urllib.request
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
import sys
import cv2
import shutil


app = Flask(__name__)

# Directory paths
base_path_ny = "downloadedImages/NY/raw_images" 
save_folder_path = "downloadedImages/NY/processed_images"
detectedROI_folder = "downloadedImages/NY/processed_images/detectedROI"
cropped_folder = "static/downloadedImages/NY/processed_images/cropped_images"
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


## TESTING FLASK
@app.route("/")
def hello():
    return "<h1>Hello World!</h1>"

#FUNCTION TO OBTAIN THE IMAGES IN THE RAW_IMAGES FOLDER
def get_image_paths(base_path):
    image_paths = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):  # Adjust extensions as needed
                image_paths.append(os.path.join(root.replace('static/', ''), file))
    return image_paths


## FUNCTION TO CREATE A NEW FOLDER EVERY DAY
def create_folder_everyday(base_path):
    current_date = datetime.now().date()
    current_folder_path = os.path.join(base_path, str(current_date))
    
    # if folder doesn't exist yet
    if not os.path.exists(current_folder_path):
        os.makedirs(current_folder_path)
        print(f"New folder created: {current_folder_path}")
        
    return current_folder_path


## FUNCTION TO CROP IMAGE INTO 6 PARTS
def crop_image(image_path, save_folder_path):
    # Open the image
    image = Image.open(image_path)

    # Cut the bottom descriptive strip away
    width, height = image.size
    topPortion = image.crop((0, 0, width, height - 31))  # Assume bottom strip height is 31px

    part_width = width // 6
    cropped_image_paths = []

    # Crop the image into 6 parts
    for i in range(6):
        part = topPortion.crop((i * part_width, 0, (i + 1) * part_width, height - 31))
        cropped_image_name = f"part_{i+1}.jpg"
        cropped_image_path = os.path.join('static', save_folder_path, cropped_image_name)
        part.save(cropped_image_path)
        # Store only the relative path from the static directory
        relative_image_path = os.path.join(save_folder_path, cropped_image_name)
        cropped_image_paths.append(relative_image_path)

    return cropped_image_paths


## FUNCTION TO CROP ALL IMAGES IN THE RAW_IMAGES FOLDER
def crop_all_images(base_path, save_folder_path):
    image_paths = get_image_paths(base_path)
    for image_path in image_paths:
        cropped_image_paths = crop_image(image_path.replace('static/', ''), save_folder_path)



## FUNCTION TO AUTOMATE ROI
def automateROI(image_path, save_path=None):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([50, 10, 10])
    upper_blue = np.array([150, 255, 150])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_segmented, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    h += 20  # Extend the height a bit for more context
    roi_rectangle = np.array([[0, y], [image.shape[1], y], [image.shape[1], y + h], [0, y + h]])
    result_image = cv2.drawContours(image.copy(), [roi_rectangle], -1, (0, 255, 0), 2)
    if save_path:
        cv2.imwrite(save_path, result_image)
    return result_image, y, h


## FUNCTION TO DETECT AND VISUALIZE HOUGH TRANSFORM LINES
def detect_and_visualize_lines(image_path, save_folder):
    print(f"Processing {image_path}")

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
        print(f"No lines detected in {image_path}.")
        return None, None
    
    # Draw lines on the image
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
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Using green color for lines

    # Calculate gradient of the strongest line (assuming the first line is the strongest)
    rho, theta = lines[0][0]
    gradient = -np.cos(theta) / np.sin(theta)
    print(f"Gradient = {gradient}")
    
    # Save the image with the detected lines
    output_image_path = os.path.join('static', save_folder, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)
    print(f"Image with detected line saved to: {output_image_path}")
    
    return output_image_path, gradient

    print(f"Processing processing {image_path}")


## FUNCTION TO CROP IMAGES INTO THEIR ROI RECTANGLES
def crop_images_into_roi(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Read the image
            image = cv2.imread(os.path.join(folder, filename))
            
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



@app.route("/buoy_data")
def buoy_data():
    ny_station = Buoy(44065)
    image_url = ny_station.image_url

    current_datetime = datetime.now()
    current_date_time_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # Call the function to crop all images in the raw_images folder
    crop_all_images(base_path_ny, save_folder_path)

    # Prepare all the raw images
    raw_images_paths = get_image_paths(base_path_ny)

    # Prepare list of existing processed images to display
    processed_images_folder = "static/downloadedImages/NY/processed_images"
    processed_image_files = [os.path.join(processed_images_folder.replace('static/', ''), f)
                             for f in os.listdir(processed_images_folder)
                             if f.endswith('.jpg') or f.endswith('.jpeg')]

    global cropped_folder
    image_files = []
    roi_images = []
    roi_coordinates = {}

    try: 
        # Creating new folder every day for each buoy
        new_folder_ny = create_folder_everyday(base_path_ny)  # Ensure base_path_ny is correct

        # Save image as a file as {time}.jpg in the date folder
        local_filename_ny = f"{current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.jpg" 

        # Get the full path to save the image
        local_path_ny = os.path.join(new_folder_ny, local_filename_ny) 
        # Download the image
        urllib.request.urlretrieve(image_url, local_path_ny)
        image_save_message = f"Image saved as {local_filename_ny} in {local_path_ny}"
        print(image_save_message)

        # Crop the image
        cropped_image_paths = crop_image(local_path_ny, save_folder_path)

        roi_images = []
        for path in cropped_image_paths:
            full_path = os.path.join('static', path)  # Ensure 'static' is used correctly
            save_path = os.path.join('static', detectedROI_folder, os.path.basename(path))
            result, y, h = automateROI(full_path, save_path)  # Corrected call to automateROI
            if result is not None:
                roi_image_path = os.path.join(detectedROI_folder.replace('static/', ''), os.path.basename(path))
                roi_images.append(roi_image_path)
                print("This is ROI Image path" + roi_image_path)

        # Check and append files if they are images
        for filename in os.listdir(cropped_folder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                file_path = os.path.join('static', cropped_folder, filename)
                # Assuming you are storing static files in a folder named 'static'
                image_files.append(file_path.replace('static/', '')) # Modify path for web access

    except Exception as e:
        image_save_message = f"Error downloading image: {e}"
        cropped_image_paths = []

    processed_images_folder = "static/downloadedImages/NY/processed_images"
    processed_image_files = [os.path.join(processed_images_folder.replace('static/', ''), f)
                             for f in os.listdir(processed_images_folder)
                             if f.endswith('.jpg') or f.endswith('.jpeg')]

    for cropped_image_path in cropped_image_paths:
        full_image_path = os.path.join('static', cropped_image_path)
        detect_and_visualize_lines(full_image_path, detectedLines_folder)

    return render_template('buoy_data.html', raw_images_paths=raw_images_paths, image_url=image_url, current_date_time=current_date_time_str, local_filename_ny=local_filename_ny, local_path_ny=local_path_ny, image_save_message=image_save_message, cropped_image_paths=cropped_image_paths, processed_image_files=processed_image_files, roi_images=roi_images, image_files=image_files)

        

if __name__ == "__main__":
    app.run(debug=True)
