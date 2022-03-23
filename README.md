# MLP-Project
Repository for Project for the course 'Machine Learning Project' at the 
Rijksuniversiteit Groningen.

### instructions for data 
After pulling this repository, please download the folder fingers with train 
and test data from https://www.kaggle.com/koryakinp/fingers and place the 
test and train directory into the data folder. The MLP-Project directory should 
then look like:
```
MLP-Project
|   .gitignore
|   LICENSE
|   main.py
|   README.md
|
└─── fingers
    |   test
    |   train
```

The main.py should then print the number of files of that directory.

## preprocessing
Labels:
```
[['1' 'L']
 ['4' 'L']
 ['2' 'R']
 ['6' 'L']
 ['5' 'R']]
```
Images:
Numpy array with an image per row and images in 1 numpy array.

### updates 
**23-03-2022** changed the folder name from data to fingers 

**23-03-2022** created preprocessing file