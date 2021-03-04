
---

<div align="center">    
 
# TRES-Net   

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/2006.03677)

<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://arxiv.org/abs/2006.03677)
-->



</div>
 
## Description   
This is an implementation of a visual transformer.
I used resneSt(50-101-152) as backbone to extract feature maps and tokens from the images.
The model converges very fast and reaches reasonable accuracy(95%++) on most publicly available small-medium datasets.

![arch](images/archhhh2-770x388.png)

## How to run   

First, install dependencies   
```bash
# clone project   
git clone https://github.com/Zyarra/Visual_transformer

# install project
pip install -r requirements.txt
python project\main.py
 ```   
python tresnet.py
All default arguments are added for imagenet, but stuff like data_dir has to be adjusted to your own structure.
see --help for the rest of arguments

 

