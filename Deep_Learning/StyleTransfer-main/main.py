import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import os

seed = 42
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
im_size = 448
max_epochs = 6000
lr = 0.001
alpha = 1
beta = 0.01

# ----------------------------------------------- #
# Pretrained VGG19   							  #
# ----------------------------------------------- #
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # ‘conv1 1’, ‘conv2 1’, ‘conv3 1’, ‘conv4 1’,‘conv5 1’
        self.layers = ['0', '5', '10', '19', '28'] 
        # Not the dense (classifier) part of the network
        self.vgg = models.vgg19(pretrained=True).features

        
    def forward(self, x):
        # Get the convolutional feature maps for selected layers
        # key and value, if key is one of the selected ones
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features.append(x)
        return features

# ----------------------------------------------- #
# Load images									  #
# ----------------------------------------------- #
transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]),])

def loadImg(path):
    # load image add Batch dimension and send to device
    img = Image.open(path)
    return transform(img).unsqueeze(0).to(device)

def get_images():
    # Content, Style
    content = loadImg("image/style_1.jpg")
    style = loadImg("image/content_1.png")

    # Generated image: noise
    generated = torch.randn(content.data.shape, device=device, requires_grad=True)

    return content, style, generated

# ----------------------------------------------- #
# Training   									  #
# ----------------------------------------------- #

def train():
    # load model to device
    vgg = VGG().to(device).eval()

    content, style, generated = get_images()
    # important: optimize generated image
    optimizer = optim.Adam([generated], lr=lr)

    content_loss = 0
    style_loss = 0
    for ep in range(max_epochs):
        # Extract multiple(5) conv feature vectors
        generated_features = vgg(generated)
        content_features = vgg(content)
        style_features = vgg(style)

        for g, c, s in zip(generated_features, content_features, style_features):
            # Content Loss
            content_loss += torch.mean((c - g)**2)

            # Reshape feature maps to (C, H*W) 2D matrix
            _, c, h, w = g.size()
            g = g.view(c, h * w)
            s = s.view(c, h * w)

            # Gram matrix matmul(matrix with its transpose)
            GramGen = torch.mm(g, g.t())
            GramStyle = torch.mm(s, s.t())
           	# Style Loss
            style_loss += torch.mean((GramGen - GramStyle)**2)

       # Total loss
        total_loss = alpha * content_loss + beta * style_loss
        # Optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if ep % 200 == 0:
            print(total_loss)
            denormalize = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = generated.clone().squeeze()
            img = denormalize(img).clamp_(0, 1)
            if not os.path.exists("gen"):
                os.makedirs("gen")
            save_image(img, "gen/generated_" + str(ep) + ".png")


# ----------------------------------------------- #
# Main       									  #
# ----------------------------------------------- #
if __name__ == "__main__":
    train()