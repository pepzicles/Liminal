# Liminal
by [Katherine Moriwaki](https://kakirine.com/)

<img src="Media/final.gif" alt="demo" width="600"/>

## lim·i·nal
/ˈlimənəl/

adjectiveTECHNICAL
adjective: liminal
1. occupying a position at, or on both sides of, a boundary or threshold. "I was in the liminal space between past and present"
2. relating to a transitional or initial stage of a process.
"that liminal period when a child is old enough to begin following basic rules but is still too young to do so consistently"

Liminal: A space in between. A transition from one stage to another. 


## Concept
This project explores liminality through the capture, transformation, and re-presentation of images captured by sea buoys within the the NOAA BuoyCAM network.

NOAA Buoys in the BuoyCAM network publish a strip of six photos every ten minutes as part of the data stream each buoy produces. The code checks the buoy and pulls the most recent strip of images. Each image in the strip is separated into its own standalone image. Horizon detection for each image is established, then each image is manipulated in code so that the horizon on each image is aligned and averaged together. 

In order to do this we used strategies outlined in academic papers on horizon line detection for autonomous marine vehicles and horizon detection using a ROI - or region of interest. 

The blended images when layered over each other with merged opacity create a single image that slowly updates as more captured and processed images are added to it. The end effect is one of an unsettling indistinguishability. In other words, the view of a horizon, across the vast expanse of ocean that surrounds us might be variable, but is also unchanging, and this tension is something we also experience in our experience of our own existence and humanity. 

Two buoys 44025 (Santa Monica Basin) and 44065 (New York Harbor) are monitored by the system. Each buoy has its own data and image set. The work is presented in two forms: 
1. As a website. The single page site presents the images from each buoy side by side. The images from the buoys might change slowly, or they may look the same depending on weather conditions and time of day. Regardless, they present a kind of liminality, a sense of however between states as represented by each view captured by the buoyCAM.
2. As an installation. Each buoy’s images are combined into a projectable image. Each image is projected onto opposite sides of a room. 

## Navigating this repository
In the [Code folder](Code), you will see:
1. [downloadedImages folder](Code/downloadedImages)- This is the folder where all the buoy images are stored and processed from. You can also find all the processed buoy images in this folder. This is the folder which all your images from running Liminal.ipynb will be stored.
2. [Flask Application folder](Code/flaskApplication) - This is the folder where you will find the Flask Application being built to turn this project into a web application. Unfortunately, as of Aug 2024, the Flask Application is not complete.
3. [Liminal.ipynb](Code/Liminal.ipynb) - This is the Jupyter Notebook file where all the code runs from, and all the images are being generated from. See below to find out how to run the code in Jupyter Notebook.
4. [run_notebook.py](Code/run_notebook.py) - This is the python script that automatically runs Liminal.ipynb every 10 minutes, so that a new video with updated, real-time buoy images can be generated every 10 minutes (the time it takes for a new buoy image to upload is also 10 minutes, so this is the fastest rate we can go to get new buoy images in real time). See below to find out how to run this python script.

## How to run code
As of August 2024, this project is not complete yet. We have yet to turn this project into a full-fledged, web-application instead of a local one. For now, the project works works by running locally, meaning all the images and media will be stored locally, on your device. 

Running code locally: 
1. Under the folder "Code", download [Liminal.ipynb](Code/Liminal.ipynb).
2. If you don't have it already, download [Anaconda](https://www.anaconda.com/download). Then, install Juypter Notebook in Anaconda.
3. In the Anaconda-Navigator, launch Jupyter Notebook. Navigate to where you have downloaded Liminal.ipynb, and open it in Jupyter Notebook.
4. Run all the cells. All images, media should be downloaded locally in a folder called "downloadedImages".
<br>

To automatically run the code every 10 minutes: 
1. Under the folder "Code", download [run_notebook.py](Code/run_notebook.py) in the same folder as Liminal.ipynb. 
2. Open terminal. Navigate (cd) into the folder where you have Liminal.ipynb and run_notebook.py
3. In terminal, run: _python run_notebook.py_
This would automatically make Liminal.ipynb restart and run all its cells every 10 mins.

## References
We got the idea of cropping the images into their respective **regions of interest (ROI)** before trying to align their horizons from these research papers: 
1. [Fast horizon detection in maritime images using region-of-interest](https://journals.sagepub.com/doi/10.1177/1550147718790753)
2. [An Overview on Horizon Detection Methods in Maritime Video Surveillance ](https://www.researchgate.net/publication/340796087_An_Overview_on_Horizon_Detection_Methods_in_Maritime_Video_Surveillance)
3. [Research on a Horizon Line Detection Method for Unmanned Surface Vehicles in Complex Environments](https://www.researchgate.net/publication/371118691_Research_on_a_Horizon_Line_Detection_Method_for_Unmanned_Surface_Vehicles_in_Complex_Environments)
4. [Automatic detection of small surface targets with Electro-Optical sensors in a harbor environment](https://www.researchgate.net/publication/228697970_Automatic_detection_of_small_surface_targets_with_Electro-Optical_sensors_in_a_harbor_environment)

## Contact 
If you have any questions, thoughts or comments about this piece, do contact [Katherine Moriwaki](https://kakirine.com/) at moriwakk@newschool.edu
