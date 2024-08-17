# Liminal
by [Katherine Moriwaki](https://kakirine.com/)

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


## How to run code
As of August 2024, this project is not complete yet. We have yet to turn this project into a full-fledged, web-application instead of a local one. For now, the project works works by running locally, meaning all the images and media will be stored locally, on your device. 

Running code locally: 
1. If you don't have it already, download [Anaconda](https://www.anaconda.com/download). Then, download Juypter Notebook in Anaconda.
2. 
3. Under the folder "Code", download into [Liminal.ipynb](Code/Liminal.ipynb). Open this in Jupyter Notebook.
4. 


## Contact 
If you have any questions, thoughts or comments about this piece, do contact [Katherine Moriwaki](https://kakirine.com/) at moriwakk@newschool.edu
