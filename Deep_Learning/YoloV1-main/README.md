# YoloV1
[YOLO_V1](https://arxiv.org/abs/1506.02640) : You Only Look Once: Unified, Real-Time Object Detection as presented in original paper.


"It frames object detection as a re-gression problem to spatially separated bounding boxes and associated class probabilities"

Goal: Predict bounding boxes and class probabilities directly from full  images  IN ONE EVALUATION.

This is important as previous techniques as Sliding windows would result in much computation or Regional based networks could be really tricky to implement and optimize but always at least a two step process.

 With YOLO, a single neural network can be optimized end-to-end directly.
 
# What makes Yolo unique:
It splits the image into SxS number of boxes for which B number of boxes are predicted and evaluated.
So with a single forward pass there is a backward loss related to every Bounding Box related to each image frame.
This loss is obtained via an specific YOLO loss function which takes into consideration the presence of an object and its coordinates.

# Labels and Predictions

Labels are represented by a vector of the form \[Class0, ... ClassN, Pc, X, Y, W, H\]

Predictions are \[Class0, ... ClassN, Pb1, X1, Y1, W1, H1, Pb2, X2, Y2, W2, H2]  Note it produces two bounding boxes

* note: a cell can only detect/have one class

Where:
- Class0...ClassN are each of the classes. In this case 20 as it is trained with [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset.
- Pc: probaility of each class for labels its ether 1 or 0
- Pb1 and Pb2: output probability of the predicted class hence the output of the NN softmax function in last layer.
- X,Y: oject's midpoint coordinates
- W, H: Weight and Height of the bounding box 

## YOLO loss
The loss function presented in the paper is as showed in the image (taken from the paper).

<img src="https://user-images.githubusercontent.com/75669936/135240745-6b34bb6a-104e-45b1-a936-4433b07ae78d.png" width="150" height="100">

Split image into SxS grid -> eacg cells outputs a prediction with corresponding bounding box
Find one cell responsible for each object -> the one that contains its midpoint.
Each output label is relative to each cell.

Lambda multipliers are priority constants
- l_coord: gives more weight to the loss related to the box coordinates (5)
- l_noobj: gives more weight to the loss related to the misprescence of an object in the cell (0.5)

<img src="https://latex.codecogs.com/svg.image?\sum_{i=0}^{S^{2}}" title="\sum_{i=0}^{S^{2}}" /> sum of all the cells (S=7)

<img src="https://latex.codecogs.com/svg.image?\sum_{j=0}^{B}" title="\sum_{j=0}^{B}" /> all the boxes (B=2)

<img src="https://latex.codecogs.com/svg.image?\bg_white&space;{I}_{ij}^{obj}" title="\bg_white {I}_{ij}^{obj}" /> Enabler. 1 when there is an object in cell i 0 otherwise

![CodeCogsEqn](https://user-images.githubusercontent.com/75669936/135246629-d4c747ea-db7c-48f8-b1a5-75255136c65f.png) Enabler. 1 no object in cell. 0 object in cell 

As it can be seen, the loss function is a sum of Mean Squared Errors for:
1) Box coordinates when there is an object in the cell
2) height and width of the box when there is an object in the cell
3) Probability that there in an object
4) If there is no object in cell
5) Class loss if there is an object in cell

## Dataloader
Target should be output as \[S,S, (C+5*B)] which in this context means: \[7,7,30]
Formed as:
- \[S,S] = the row and column as the cell splitted label
- \[..., 0:20] from 0 to 19 the total classes
- \[..., 20] class proabability (output of the sofmatx function)
- \[..., 21:25] from 21 to 24 X, Y W, H
- \[..., 26:] Unused for label but predictions has the other boudningbox

Data Augumentation can be applied to both but resizing can be just aplied to the image as label is already cell proportional.

----
## TODO:
- Show images with corresponding boundingboxes and labels
- checkpoint and model loader

