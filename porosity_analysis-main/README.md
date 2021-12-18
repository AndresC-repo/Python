# porosity_analysis
Little code done as a favor to a friend, done is short time

TODO:
Better user interface to modify thresholds individualy for every image (maybe calculate threshold automatically)
Analyse rad of circles

**Table of Contents**
 * [Installation](#installation)
 * [Project Structure](#Project_Structure)

## Installation
Packages used:
-pillow
-albumentations
-cv2
-numpy
-matplotlib.pyplot



## Project_Structure
```
├── root_images/                    <<  
│   └── image.tif                   <<  Raw images
│
├── [main.py]/                       <<  Main code of the project
│
├── ---results/                       <<  Creates individual folders for every image
|    └── Original.jpg                     <<  Original image with the useless bottom cut
|    └── median_filter.jpg                <<  Original - median filter
|    └── THRESH_BINARY.jpg                <<  median filter -> Otsu  *** USED
|    └── Adaptive Gaussian Thresh.jpg     <<  median filter -> adaptive guass trsh
|    └── Adaptive Mean Thresh.jpg         <<  median filter -> adaptive mean trsh
|    └── Distance Transform.jpg           <<  THRESH_BINARY -> distance_transform
|    └── overlay.jpg                      <<  Tresh_binary + original
|    └── bin_cirlces.jpg                <<  Sets circles on the inverse of the Binary_image using the distance transform as reference
|    └── support_bin_circles.jpg                <<  Support image to show how it works
|    └── histogram media_filter.png                <<  Pixel histogram of median filter img
|    └── histo_rad_circles.png                <<  Histogram of radius of circles created
|    └── radius.txt                    <<  list of radius of cicles created
|
└── .gitignore                      <<  To personalize what is uploaded into the Git
