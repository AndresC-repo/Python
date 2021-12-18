# Multi-Label Image Classification
Multilabel image classification challenge, using a modified version of Microsoft COCO 2017 dataset.


Kaggle competition description:

The competition is linked to DSEG660: Applied Deep Learning, taught by Dr David at HBKU.

Task:
Multilabel classification problem which is a simplified version of the original COCO Object Detection Task which involved also predicting the segmentation/bounding box of the detected class.

Uses Pytorch lightning framework to train and predict over a multilael classification problem.

Model: EfficientNet https://arxiv.org/pdf/1905.11946.pdf
Criterion:
- BCEWithLogitsLoss
- AsymmetricLoss https://arxiv.org/pdf/2009.14119.pdf
- Focal loss https://arxiv.org/pdf/1708.02002v2.pdf

Optimizer: Adam
Scheduler: StepLR


## Just commands
List all just commands with the following command:
```bash
just <COMMAND NAME>
```

### Tensorboard
While training your model you can observe its progress with tensorboard:

Example:
```bash
tensorboard --logdir "./training/"
```



## Project Structure

```
├── <DATASET>/                  -
│     └── imgs/                 -
│     └── labels/               -
│
├── <parts>                     -  
│   ├── loss.py                 <<  get criterion
│   ├── dataloader.py           <<  Construction of dataloader
│   ├── model.py                <<  Building of EfficientNET
│   └── optim.py                <<  get Optimizer
│
│
├── <MAIN>                      -  
│   ├── main.py                 <<  Training -> pytorch lightning
│   └── inference.py            <<  for individual images
│
├── <extras>                    -
│   ├── trainer_lr.py           <<  uses lightning trainer to obtain lr
│   └── weighter.py             <<  Weighter sampler
│
│
├── justfile                    <<  Just commands
└── .gitignore                  <<  Personalize what is uploaded into the Git


```

## License

TODO: Update!