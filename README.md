# AGSCC
 An adaptive graph and structure cycle consistency (AGSCC) based image regression method for heterogeneous change detection
 
## Introduction
MATLAB Code: AGSCC-2022
This is a test program for the adaptive graph and structure cycle consistency (AGSCC) method for heterogeneous change detection problem.

the proposed method first constructs an adaptive graph to represent the structure of pre-event image, which connects each superpixel with its truly similar neighbors by using a $k$-selection strategy and adaptive-weighted distance metric. Based on the fact that the similarity relationships based structure can be well preserved across different imaging modalities, the adaptive graph can be used to translate the pre-event image to the domain of post-event image with three types of regularization: forward transformation term, cycle transformation term and sparse regularization term.

Please refer to the paper for details. You are more than welcome to use the code! 

===================================================

## Available datasets

#6-California is download from Dr. Luigi Tommaso Luppino's webpage (https://sites.google.com/view/luppino/data) and it was downsampled to 875*500 as shown in our paper.

#7-Texas is download from Professor Michele Volpi's webpage at https://sites.google.com/site/michelevolpiresearch/home.

===================================================

## Citation

If you use this code for your research, please cite our paper. Thank you!

Sun, Yuli, et al. "Image Regression with Structure Cycle Consistency for Heterogeneous Change Detection."
IEEE Transactions on Neural Networks and Learning Systems, 2022

## Q & A

If you have any queries, please do not hesitate to contact me (yulisun@mail.ustc.edu.cn ).
