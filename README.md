
# ShearWall
This is a repo for 2024 S2 Capstone Project. More information will be added.

## Virtual Environment
It is recommended that we install packages in a virtual environment.
To setup virtual environment for the first time: `python3 -m venv venv`

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

## How to Run
### Training
1. For model training, please execute `python segmentation/train_*.py` in the root directory.
2. Batch size can be adjusted based on the device.

### Segmentation
1. For crack segmentation, please execute `python predict.py [Type of model] [Image file name]`.
2. Current available model includes FCN.
3. An example of usage is `python predict.py FCN crack.png`