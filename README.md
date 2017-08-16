# Spread-out Local Feature Descriptor

This code is the training and evaluation code for our ICCV 2017 paper.

@inproceedings{zhang2017learningb,
  title={Learning Spread-out Local Feature Descriptors},
  author={Zhang, Xu and Yu, Felix X. and Kumar, Sanjiv and Chang, Shih-Fu},
  booktitle={ICCV},
  year={2017}
}

The code is tested on Ubuntu 14.04

### Requirement
Python package:

tensorflow>1.0.0, tqdm, cv2, exifread, skimage, glob

### Usage

#### Get the data

Download data from 
https://www.dropbox.com/s/l7a8zvni6ia5f9g/datasets.tar.gz?dl=0

and put the extract the data to ./data/

#### Run the code

Change Matlab link in all the files in `./script/`

`cd ./script`

Generate transformed patch and train the model

`./batch\_run_train.sh`

Extract local feature point

`./batch\_run_test.sh`

Evaluate the performance

`./batch\_run_eval.sh`

### Acknowledgement 

We would like to thank

VLfeat [1], http://www.vlfeat.org/ 

Tilde [2], https://github.com/kmyid/TILDE

Karel Lenc etal [3], https://github.com/lenck/ddet

for offering the implementations of their methods. 

and

UBC dataset [3]

for providing the image data.

[1] A. Vedaldi and B. Fulkerson, VLFeat: An Open and Portable Library of Computer Vision Algorithms
