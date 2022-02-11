Texture Information Boosts Video Quality Assessment
===
Introduction
---

<img src="https://github.com/GZHU-DVL/TiVQA/blob/main/Framework.jpg" width="650" height="310" /><br/>

Usage
---
First, you need to run Features extraction.py to extract content-dependency features and texture features. Note that you need to specify specific datasets and corresponding paths, where the default dataset is koNViD-1k. Then run TiVQA.py to test the results. Because the LSVQ dataset contains too many videos, the code is different from the other three datasets, and you need to run TiVQA_LSVQ.py to test the results.

## Requirment
* torch==1.6.0
* torchvision==0.7.0
* h5py==2.10.0
* scikit-video==1.1.11
* scikit-image==0.18.1
* scikit-learn==0.24.1
* scipy==1.4.1
* tensorboardX==2.2
* tensorflow-gpu==2.2.0
* numpy==1.19.5
