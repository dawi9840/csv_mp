# csv_mp
The dataset is recorded some pose coordinates, I want to use it to classify the pose.  

My purpose is classified the pose in **model_train_pose.py**.  

The idea is from Keras documents tutorial: [Structured data classification from scratch](https://keras.io/examples/structured_data/structured_data_classification_from_scratch/), I learn how to use csv file to be classified.  

# Scripts 

**model_train_pose.py** - This file is training and saving model for classifying pose.  

**model_inference_mp.py** - Input a sample pose data from **numerical_coords_dataset_test.csv** to test the saved model.

# Installation

**Conda virtual env**

```bash

conda create --name [env_name]  python=3.8
conda activate [env_name]
pip install pandas==1.1.3
pip install numpy
pip install tensorflow-gpu==2.6.0
conda install cudnn==8.2.0.53
pip install pydot
pip install mediapipe==0.8.6.2
```
