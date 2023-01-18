# warm up
python train.py --cuda --gpu "0" --dataset "ICVL" --noiseType "gaussian" --model_name "res3conv" --nEpochs 10 --sigma 30 

# res3conv
python train.py --cuda --gpu "0" --dataset "ICVL" --noiseType "gaussian" --model_name "res3conv" --nEpochs 25 --sigma 10 20 30 --resume checkpoints/ICVL/res3conv_gaussian_epoch_10.pth

# res3net
python train.py --cuda --gpu "0" --dataset "ICVL" --noiseType "gaussian" --model_name "res3net" --nEpochs 25 --sigma 10 20 30 --resume checkpoints/ICVL/res3conv_gaussian_epoch_25.pth
