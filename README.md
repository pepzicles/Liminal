# Liminal
by [Katherine Moriwaki](https://kakirine.com/)

## About
This project explores liminality through the capture, transformation, and re-presentation of images captured by sea buoys within the the NOAA BuoyCAM network.

NOAA Buoys in the BuoyCAM network publish a strip of six photos every ten minutes as part of the data stream each buoy produces. The code checks the buoy and pulls the most recent strip of images. Each image in the strip is separated into its own standalone image. Horizon detection for each image is established, then each image is manipulated in code so that the horizon on each image is aligned and averaged together. 

In order to do this we used strategies outlined in academic papers on horizon line detection for autonomous marine vehicles and horizon detection using a ROI - or region of interest. 

The blended images when layered over each other with merged opacity create a single image that slowly updates as more captured and processed images are added to it. The end effect is one of an unsettling indistinguishability. In other words, the view of a horizon, across the vast expanse of ocean that surrounds us might be variable, but is also unchanging, and this tension is something we also experience in our experience of our own existence and humanity. 

Two buoys 44025 (Santa Monica Basin) and 44065 (New York Harbor) are monitored by the system. Each buoy has its own data and image set. The work is presented in two forms: 
1. As a website. The single page site presents the images from each buoy side by side. The images from the buoys might change slowly, or they may look the same depending on weather conditions and time of day. Regardless, they present a kind of liminality, a sense of however between states as represented by each view captured by the buoyCAM.
2. As an installation. Each buoy’s images are combined into a projectable image. Each image is projected onto opposite sides of a room. 


## How to run code
