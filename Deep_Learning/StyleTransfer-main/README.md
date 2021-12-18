# StyleTransfer
Implementation of "[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)" paper using Pytorch.
---
"Artificial system based on a Deep Neural Network that creates artistic images
of high perceptual quality"

Goal: Output a high-resolution image which is the combination of two input images, one that gives the shape and the style is taken from the other.



---
What is **Content** in NN? The normal flow of a nn is done to learn the content of an image. We can visualize the information each layer contains, by trying to reconstruct the input image only from the feature maps in that layer.
Higher layers in the network capture high-level content in terms of objects and their arraingment in the input image.

What is **Style**? In NN we obtain this using a fetaure space desinged to capture texture built on top of the filter responses in each layer of the network.
Style consists in the correlation between the different filter responses over the spatial extent of the feature maps.

---
<img src="https://user-images.githubusercontent.com/75669936/136451835-6980b617-d70a-46ab-97f6-4648a3b50bb1.png" width="550" height="400">

## loss
The loss function is as presented in the paper which is the sum of a ContentLoss (Lcontent) and StyleLoss (Lstyle).
<img src="https://latex.codecogs.com/svg.image?\bg_black&space;\L_{total}\left&space;(&space;G&space;\right&space;)&space;=\alpha&space;\L&space;_{content}\left&space;(&space;C,G&space;\right&space;)&plus;\beta&space;\L&space;_{style}\left&space;(&space;S,G&space;\right&space;)" title="\bg_black \L_{total}\left ( G \right ) =\alpha \L _{content}\left ( C,G \right )+\beta \L _{style}\left ( S,G \right )" /> 

Where:
**C**: content/original image
**G**: Generated image
**S**: Style image

 **α** and **β** are the weighting factors for content and style reconstruction respectively.

<img src="https://latex.codecogs.com/svg.image?\bg_black&space;\L&space;_{content}\left&space;(&space;C,G&space;\right&space;)=\frac{1}{2}\left\|&space;a^{\left&space;[&space;l&space;\right&space;]&space;\left&space;(&space;C&space;\right&space;)}&space;-a^{\left&space;[&space;l&space;\right&space;]&space;\left&space;(&space;G&space;\right&space;)}&space;\right\|^{2}" title="\bg_black \L _{content}\left ( C,G \right )=\frac{1}{2}\left\| a^{\left [ l \right ] \left ( C \right )} -a^{\left [ l \right ] \left ( G \right )} \right\|^{2}" /> Takes the norm of every layer (a is the output of the layer "l")

Style Loss
Gram Matrices: Multiply the output of the layer with its transpose

<img src="https://latex.codecogs.com/svg.image?\bg_black&space;G_{ij}^{l}=\sum_{k}g_{ik}^{l}g_{jk}^{l}" title="\bg_black G_{ij}^{l}=\sum_{k}g_{ik}^{l}g_{jk}^{l}" /> Gram matrix for the Generated image 

<img src="https://latex.codecogs.com/svg.image?\bg_black&space;S_{ij}^{l}=\sum_{k}s_{ik}^{l}s_{jk}^{l}" title="\bg_black S_{ij}^{l}=\sum_{k}s_{ik}^{l}s_{jk}^{l}" /> Gram matrix for Style

Substract them both and sum the square results
<img src="https://latex.codecogs.com/svg.image?\bg_black&space;\L&space;\left&space;(&space;g,s&space;\right&space;)=\sum_{ij}\left&space;(&space;&space;G_{ij}^{l}-S_{ij}^{l}\right&space;)^{2}" title="\bg_black \L \left ( g,s \right )=\sum_{ij}\left ( G_{ij}^{l}-S_{ij}^{l}\right )^{2}" />


---
## Model 
This work uses the [VGG19](https://pytorch.org/hub/pytorch_vision_vgg/) as out of the box network

### Forward pass
Remove Dense layers of network.
For Original image, Style imagae and Generated Image (noise at the beginning)
- Take the ouput of: ‘conv1 1’ (a), ‘conv2 1’ (b), ‘conv3 1’ (c), ‘conv4 1’ (d) and ‘conv5 1’ (e)  -> with C, H, W and pass them to the loss function





## Dataloader
Pillow: Load Style image, Original image
- One batch at a time. BS=1
- Augumentation: Resize and Normalize
Generated image is random noise of the shame of the other images IMPORTANT to add requires_grad=True

----
## TODO:




