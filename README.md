### Brain Tumor: dataset
Dataset for the analysis is provided at Kaggle: 
https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset?resource=download<br>
Dataset that could be good to use for testing after learning process:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data

#### Contents
- *model.py* - Stores model class and helper functions for learning process. Run, when need to execute learning process;
- *main.py* - Contains a terminal utility for making a sample prediction. The script picks a random image from Data2 dataset (refer to its desc. pls), preprocesses it and model makes a prediction. The softmax'ed model output and the real label of the chosen image are the file result;
- *.pth model files* - see below;

#### Datasets
The datasets mentioned above (further refered to as Data and Data2 respectively) are sets of images split onto 4 classes - Normal and 3 types of brain tumors. The datasets must be downloaded from the source and extracted into *Data* and *Data1* folders resp.<br>
1. The *Data* is used for learning process
2. The *Data2* is used for sample predictions in main.py. The image path must be ./Data2/Testing/<randomly_picked_class>/<filename.jpg>

#### Models
Available models are following the VisualTransformer architecture and the filenames are self-describing.<br>
L - number of layers (encoder blocks)<br>
E - embedding vector length<br>
H - number of heads for Multi-Head attention<br>
w8L is a working copy of 8L-384E-12H which has the approx. accuracy of 92-97%

