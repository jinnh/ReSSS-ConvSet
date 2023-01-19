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
Enter the HSID folder and run

	bash train_gaussian.sh

### Training the model for Complex noise
Enter the HSID folder and run

	bash train_complex.sh

### Testing

	python test.py --cuda --gpu "0" --dataset "ICVL" --noiseType "gaussian" --model_name "res3net" --checkpoint checkpoints/ICVL/res3net_gaussian_epoch_25.pth

	python test.py --cuda --gpu "0" --dataset "ICVL" --noiseType "complex" --model_name "res3net" --checkpoint checkpoints/ICVL/res3net_complex_epoch_25.pth

HS image super-resolution
--------
### Dataset

You can refer to the following links to download the dataset, [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE"). And run the matlab programs in the folder 'datasets' to get the pre-processed training and testing data.


### Training
Enter the HSISR folder and run

	bash train.sh


### Testing

	python test.py --cuda --gpu "0" --dataset "CAVE" --model_name "res3net" --upscale_factor 4 --checkpoint checkpoints/CAVE_x4/res3net_4_epoch_50.pth'

HS image classification
--------
This codebase borrows from [Spectralformer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer 'Spectralformer') and [3D-CNN](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565 '3D-CNN').

### Dataset

You can refer to the following link to download the datasets, [IndianPine and Pavia](https://drive.google.com/drive/folders/1YLGWvMUdYzRoKmThpN83n0wpapAIeloV "IndianPine and Pavia").

### Training
Enter the HSIC folder and run

	python main.py --dataset="Indian" --method="res3net" --epoch=1000 --patches=7 --weight_decay=1e-2 --learning_rate=1e-3 --gpu_id=0 --loss_weight=1e-4

	python main.py --dataset="Pavia" --method="res3net" --epoch=160 --patches=7 --weight_decay=1e-3 --learning_rate=1e-3 --gpu_id=0 --loss_weight=3e-4

### Testing

	python main.py --flag_test="test" --dataset="Indian" --method="res3net" --model_name="checkpoints/res3net_Indian/res3net_best.pt" --patches=7

	python main.py --flag_test="test" --dataset="Pavia" --method="res3net" --model_name="checkpoints/res3net_Pavia/res3net_best.pt" --patches=7

Citation 
--------
**Please kindly cite our work if you find it helpful.**

	@article{hou23deep,
		title={Deep Diversity-Enhanced Feature Representation of Hyperspectral Images},
		author={Hou, Jinhui and Zhu, Zhiyu and Hou, Junhui and Liu, Hui and Zeng, Huanqiang and Meng, Deyu},
      	journal={arXiv preprint arXiv:2301.06132}
		year={2023}
	}
  
