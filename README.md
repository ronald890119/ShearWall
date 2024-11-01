# ShearWall
This is a repo for 2024 S2 Capstone Project.

## Dataset
The images for this project are from this [repo](https://github.com/fyangneil/pavement-crack-detection). The datasets include training set, validation set, and testing set. All of them have two versions, which include high resolution and cropped images. 

## Virtual Environment
It is recommended that installing packages in a virtual environment.
To setup virtual environment for the first time please execute: `python3 -m venv venv`

To Activate the virtual environment:
- Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

To Deactivate the virtual environment:
- Linux: `deactivate`

## Packages
For future reference we will need a **requirements.txt** so that people can install the required packages via `pip install -r requirements.txt`.

Whenever you add a new package, don't forget to run the following command: `pip freeze > requirements.txt`.
This will update the requirements.txt file and will make sure other people can update their packages appropriately

## Directory Structure
Please follow the structure below to ensure consistency and convenience
```bash
ShearWall/
├── regression/
│   ├── concrete.csv
│   └── *.py
├── segmentation/
│   ├── dataset/
│   │   └── images
│   └── *.py
└── README.md
├── predict.py
├── *.pth
└── requirements.txt
```

## Trained Models
Due to the file limitation, the trained models are available from [Google Drive](https://drive.google.com/drive/u/0/folders/1t4K7JDZUPHxSTxJVII-lFqUv-Uc4znv5).
⚠️**The link will be available until 12/31/2024, please download them if needed**⚠️

## How to Run
### Training
1. For model training, please execute `python segmentation/train_*.py` in the root directory.
2. Batch size can be adjusted based on the device.

### Segmentation
1. For crack segmentation, please execute `python predict.py [Type of model] [Image file name]`.
2. Current available models include FCN and STDC.
3. An example of usage is `python predict.py FCN crack.png`