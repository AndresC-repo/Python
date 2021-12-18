# Inference
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from dataloader import get_transform
from main import MyLightingModule
import os
import pandas as pd


def add_text(image, text):
    draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 25)
    draw.text((0, 0), str(text), (26, 100, 0), font=fnt)
    # image.save('sample-out.jpg')
    return image


def evaluation(path, cats):
    # Load your image and preprocess it
    sample_img = Image.open("./imgs/test/000000003808.jpg").convert('RGB')

    transform = get_transform(244, 'test')

    img_tensor = transform(sample_img)

    model_ft = MyLightingModule.load_from_checkpoint(path)

    # Set model to evaluation mode
    model_ft.eval()
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0)  # add batchsize 1
        outputs = model_ft(img_tensor)
        outputs = torch.sigmoid(outputs)
        preds = np.around(outputs.cpu().detach().numpy())

    # Get categories classified as true
    cats_list = [cats[i]
                 for i, x in enumerate(list(preds.squeeze())) if x == 1]

    print(cats_list)
    image = add_text(sample_img, cats_list)
    image.show()


if __name__ == "__main__":

    DATA_DIR = './labels'
    cats = pd.read_csv(os.path.join(DATA_DIR, "categories.csv"), header=None)
    cats = list(cats[0])

    # path = './training/default/version_0/checkpoints/epoch=5-step=32999.ckpt'
    path = './training/default/version_3/checkpoints/epoch=9-step=54999.ckpt'

    # ------------------------
    # ARGUMENTS
    # ------------------------
    evaluation(path, cats)
