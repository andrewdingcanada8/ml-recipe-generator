# ml-recipe-generator

This repository is home to salad party's recipe generator model! For the term 
project for our Brown Deep Learning course, we've decided to create an LSTM 
based deep learning model that generates novel recipes given any starting str.

It is trained on the meal-masters dataset and ... [WIP]

## Setup
```bash
# run ./create_venv.sh to setup the python venv
./create_venv

# activate the venv
source env/bin/activate
```
## Dataset
We've already downloaded, parsed, tokenized and pickled our meal-master data via
the included jupyter notebook. If you want to see how we did it you can re-run
the included `data_mmf.ipynb`

## Running the Model
In order to train the data, just run `python code/main.py --train "<num_epochs>"`. There are more command line arguments, such as embedding_size, etc, that you can find in the argparser.

We also added functionality to continue training vs retraining the entire model, so if there is already a saved model, the program will ask you if you want to retrain the model or contninue. 

After training, you will be presented with a REPL, which you can use to feed the model sample inputs and have it generate recipes. You can re-run this REPL by running `python code/main.py --test_only`. This will skip the training part of the code and just give you the REPL.

If you do not wish to train the data and just use the tests, you can download the trained model from [this GDrive link](https://drive.google.com/drive/folders/1cDZdXIcptLIYd7BRw_PKr6TCTLPeyivl?usp=sharing) and put them in `code/models`

## Repository Structure
It is broken up into two parts: data and code. Data has all the data and data processing code, while code has the model, and functions to train and test.
