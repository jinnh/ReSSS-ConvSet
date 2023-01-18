ReSSS-ConvSet
======
**This is an implementation of Deep Diversity-Enhanced Feature Representation of Hyperspectral Images.**
[[arXiv]](https://arxiv.org/abs/2301.06132 "arXiv")

Requirement
---------
* python 3.7, pytorch 1.7.0, and cuda 11.0

* Matlab 

HS image denoising
--------
### Dataset

You can refer to the following links to download the dataset, [ICVL](http://icvl.cs.bgu.ac.il/hyperspectral/ "ICVL"). Following [QRNN3D](https://github.com/Vandermode/QRNN3D "QRNN3D"), we generated the noisy images for training and testing. You can run the matlab programs in the folder 'datasets' to get the pre-processed training and testing data.


### Training the model for Gaussian noise

	bash train_gaussian.sh

### Training the model for Complex noise

	bash train_complex.sh

### Testing

	python test.py --cuda --gpu "0" --dataset "ICVL" --noiseType "gaussian" --model_name "res3net" --checkpoint checkpoints/CAVE_x4/res3net_gaussian_epoch_25.pth

	python test.py --cuda --gpu "0" --dataset "ICVL" --noiseType "complex" --model_name "res3net" --checkpoint checkpoints/CAVE_x4/res3net_complex_epoch_25.pth

HS image super-resolution
--------
### Dataset

You can refer to the following links to download the dataset, [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE"). And run the matlab programs in the folder 'datasets' to get the pre-processed training and testing data.


### Training

	bash train.sh


### Testing

	python test.py --cuda --gpu "0" --dataset "CAVE" --model_name "res3net" --upscale_factor 4 --checkpoint checkpoints/CAVE_x4/res3net_4_epoch_50.pth'

HS image classification
--------
### Dataset

### Training

	python demo5.py --dataset="Indian" --method="RE_F1" --epoch=1000 --patches=7 --weight_decay=1e-2 --learning_rate=1e-3 --gpu_id=0 --loss_weight=1e-4

### Testing

	python test.py --flag_test="test" --dataset="Indian" --method="RE_F1" --model_name="checkpoints/RE_F1_Indian/RE_F1_best.pth" --patches=7

Citation 
--------
**Please cite our work if you find it helpful.**

	@misc{hou23deep,
		title={Deep Diversity-Enhanced Feature Representation of Hyperspectral Images},
		author={Hou, Jinhui and Zhu, Zhiyu and Hou, Junhui and Liu, Hui and Zeng, Huanqiang and Meng, Deyu},
		eprint={2301.06132},
      	archivePrefix={arXiv},
      	primaryClass={cs.CV},
		year={2023}
	}
  
