# Medical-Segmentation-Wavelet-Transform
A repo of a project in DS503 - Spring 2024 in KAIST.
## Installation
```
conda env create -f environment.yml
conda activate Wavelet_MedSeg
```
## Run the Code

```
# Run the Wavelet model for segmentation #
python train.py --num_epoch [# of epochs] --save_content --batch_size [# of batch size] --config [# name of the config file]

### example ### 
python train.py --num_epoch 100 --save_content --batch_size 14 --config monuseg_wavelet_NoGAN.yml 

# Run the Unet model for segmentation # 
python train_Unet.py --num_epoch [# of epochs] --save_content --batch_size [# of batch size] --config monuseg.yml

```
## Inference 
```
# Run the Wavelet model for segmentation #
python infer.py --config [# name of the config file]

# Run the Unet model for segmentation # 
python infer_Unet.py --config monuseg.yml

### The output of inferencing is located in the experment folder ###
```
## BF-Score 
```
# Change the root folder is the file that contain the inference output images in the file 'bf-score.py' then run #

python bf-score.py
```

## Dataset 
Data setlink: 
- [Monuseg in this project](https://drive.google.com/drive/folders/1-MmUO-3C6cuFUhcRpZl6tKl2sUsSnKyc?usp=sharing)
- [Original of Monuseg](https://monuseg.grand-challenge.org/Data/)

Please download then change the data path in the config file.
## Citation
This wavelet idea comes from [Wavelet Diffusion](https://arxiv.org/pdf/2211.16152). If you use this wavelet idea, please cite below:
```
@inproceedings{phung2023wavelet,
  title={Wavelet diffusion models are fast and scalable image generators},
  author={Phung, Hao and Dao, Quan and Tran, Anh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10199--10208},
  year={2023}
}
```

