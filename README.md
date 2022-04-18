Texture Information Boosts Video Quality Assessment
===
Introduction
---

<img src="https://github.com/GZHU-DVL/TiVQA/blob/main/Framework.jpg" width="650" height="310" /><br/>
Abstract: Automatically evaluating the quality of in-the-wild videos is challenging since both the distortion types and reference videos are unknown. In general, humans can make a fast and accurate judgment for video quality. Fortunately, deep neural networks have been developed to effectively model the human visual system (HVS). In this paper, we deeply investigate three elements of HVS, including texture masking, content-dependency, and temporal-memory effects from an experimental perspective. Based on the investigation, we propose to make full use of texture information to boost the performance of video quality assessment (VQA), termed TiVQA in this paper. To be specific, TiVQA first uses the local binary pattern (LBP) operator to detect texture information of each video frame. Then a two-stream ResNet is employed to extract the texture masking and content-dependency embeddings, respectively. Finally, TiVQA integrates both the gated recurrent unit and subjectively-inspired temporal pooling layer to model the temporal-memory effects. Extensiveexperiments on benchmark datasets including KoNViD-1k, CVD2014, LIVE-Qualcomm, and LSVQ show that the proposed TiVQA obtains state-of-the-art performance in terms of SRCC and PLCC.

Usage
---
First, you need to run Features Extraction.py to extract content-dependency features and texture features. Note that you need to specify specific datasets and corresponding paths, where the default dataset is koNViD-1k. Then run TiVQA.py to test the results. 

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

License
---
This source code is made available for research purpose only.
