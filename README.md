Texture Information Boosts Video Quality Assessment
===
Introduction
---

<img src="https://github.com/GZHU-DVL/TiVQA/blob/main/Framework.jpg" width="650" height="310" /><br/>
Abstract: Automatically evaluating the quality of in-the-wild videos is challenging since both the distortion types and reference videos are unknown. In general, humans can make a fast and accurate judgment for video quality. Fortunately, deep neural networks have been developed to effectively model the human visual system (HVS). In this paper, we deeply investigate three elements of HVS, including texture masking, content-dependency, and temporal-memory effects from an experimental perspective. Based on the investigation, we propose to make full use of texture information to boost the performance of video quality assessment (VQA), termed TiVQA in this paper. To be specific, TiVQA first uses the local binary pattern (LBP) operator to detect texture information of each video frame. Then a two-stream ResNet is employed to extract the texture masking and content-dependency embeddings, respectively. Finally, TiVQA integrates both the gated recurrent unit and subjectively-inspired temporal pooling layer to model the temporal-memory effects. Extensiveexperiments on benchmark datasets including KoNViD-1k, CVD2014, LIVE-Qualcomm, and LSVQ show that the proposed TiVQA obtains state-of-the-art performance in terms of SRCC and PLCC.

Usage
---
## Dataset Preparation
**VQA Datasets.**
We test TiVQA on four datasets, including [KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html), [CVD2014](https://www.mv.helsinki.fi/home/msjnuuti/CVD2014/), [LIVE-Qualcomm](http://live.ece.utexas.edu/research/incaptureDatabase/index.html), and [LSVQ](https://github.com/baidut/PatchVQ), download the datasets from the official website. 

## Extract the content-aware and texture features
First, you need to download the dataset and copy the local address into the videos_dir of [Features Extraction.py](https://github.com/GZHU-DVL/TiVQA/blob/main/Features%20Extraction.py). Due to the particularity of the LSVQ dataset, we give a version for extracting LSVQ in [Features Extraction_LSVQ.py](https://github.com/GZHU-DVL/TiVQA/blob/main/Features%20Extraction_LSVQ.py). In it, we mark the video sequence numbers that do not exist in the current version of LSVQ.

```
python Feature Extraction.py --database=database --frame_batch_size=16 \
python Feature Extraction_LSVQ.py --database=LSVQ --frame_batch_size=16 \
```
Please note that when extracting the content-aware and texture features, you can choose the size of frame_batch_size according to your GPU. After running the [Features Extraction.py](https://github.com/GZHU-DVL/TiVQA/blob/main/Features%20Extraction.py) or [Features Extraction_LSVQ.py](https://github.com/GZHU-DVL/TiVQA/blob/main/Features%20Extraction_LSVQ.py), you can get the content-aware and texture features of each video in the directory "LBP_P10_R4_std_CNN_features_dataset/".

## Training and Evaluating
```
python main.py  --database==database --reduced_size=reduced_size --hidden_size==hidden_size \
                --batch_size==batch_size  --tau==tau  --beta==beta
```
Please note that our parameter tuning for different datasets is only to verify the conjecture in this paper that different datasets apply to different parameters.

## Testing
After extracting the content-aware and texture features of the dataset, you can use the models already trained in the directory "/models/" to test the performance.
```
python test_performance.py  --database==database --reduced_size=reduced_size --hidden_size==hidden_size \
                --batch_size==batch_size  --tau==tau  --beta==beta
```

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

Correct
---
Due to experimental error, we apologize for the error in the experimental data of KonNViD-1k dataset, and hereby correct it.

KonNViD-1k     SRCC: 0.818
               PLCC: 0.816
             
Acknowledgement
---
This cobebase is heavily inspired by [VSFA](https://github.com/lidq92/VSFA) (Li et al., ACM2019).

Great appreciation for their excellent works.

If you find this repo helpful to your research, we sincerely appreciate it if you cite our paper.:smiley::smiley:

```
A.-X. Zhang and Y.-G Wang, “Texture information boosts video quality assessment,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 2050-2054, 2022.
```
Meanwhile, you may also consider citing Li's work,
```
D. Li, T. Jiang, and M. Jiang, “Quality assessment of in-the-wild videos,” in ACM International Conference on Multimedia (MM), pp. 2351-2359, 2019.
```

License
---
This source code is made available for research purpose only.


