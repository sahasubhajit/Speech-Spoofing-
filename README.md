# Speech-Spoofing-
A Machine Learning project aimed for a countermeasures of speech spoofing. 

A [basic lecture video for self supervised learning](https://www.youtube.com/watch?v=6N3OAWIsUOU).

Download ASVspoof 2019 dataset from [here](https://datashare.ed.ac.uk/handle/10283/3336). Run 
'''
pip install -r requirements. txt
''' 
in your terminal to install all required packages.

[wave2vec2_representation.py](https://github.com/sahasubhajit/Speech-Spoofing-/blob/main/wave2vec2_representation.py) is to get feature embedding by wave2vec 2.0 pretrained model. Adjust the file/directory path to read from train/validation/test dataset and write to designated csv file. Run this script after navigated to the directory of audio files (possible three directories for train/dev/eval).
Or you may directly get the embedding of [train](https://drive.google.com/file/d/1s5FteLqqMvqCZrjCxG3wOWhbkiIgbyX_/view?usp=sharing), [test](https://drive.google.com/file/d/1s5FteLqqMvqCZrjCxG3wOWhbkiIgbyX_/view?usp=sharing), [validation](https://drive.google.com/file/d/1sO88GHIM_T4FLD-szl3e-Olw6jCwN3pf/view?usp=sharing) datasets in csv format. 

[ml_models.ipynb](https://github.com/sahasubhajit/Speech-Spoofing-/blob/main/ml_models.ipynb) contains the implementation of all the ML models on the embedded reprsentation of ASVspoof 2019 dataset. 




