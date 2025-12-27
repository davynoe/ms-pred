# ms-pred
An LSTM model trained for predicting stock prices for Microsoft using Microsoft's stock price data<br>
Data is completely free and is acquired from kaggle.com

## Prerequisites
- Have Python 3.12+ installed

## Installation and Preparation
Clone the repo and install the required libraries
```
git clone https://github.com/davynoe/ms-pred
cd ms-pred
pip install -r requirements.txt
```

## Running
Then either run `train.py` which trains the model and runs the program<br>
Or `load.py` that just runs the program over the model that is pre-trained
```
python train.py

python load.py
```