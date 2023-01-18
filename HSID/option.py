import argparse
# Training settings
parser = argparse.ArgumentParser(description="Hyperspectral Image Denoising")
parser.add_argument("--noiseType", default="gaussian", type=str, help="noise type (Gaussian, complex)")
parser.add_argument("--sigma", default=30, type=int, nargs="+", help="guassion noise sigma")
parser.add_argument('--seed', type=int, default=1,  help='random seed (default: 1)')
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=25, help="maximum number of epochs to train")
parser.add_argument('--model_name', default='res3conv', type=str, help="model name (res3conv, res3net)")
parser.add_argument("--svdLossWeight", type=float, default=1e-4, help="initial svdLossWeight")
parser.add_argument("--lr", type=int, default=5e-4, help="initial learning rate")
parser.add_argument("--cuda", action="store_true", help="Use cuda")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")
parser.add_argument("--threads", type=int, default=8, help="number of threads for dataloader to use")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)") 
parser.add_argument("--dataset", default="ICVL", type=str, help="data name (ICVL,)")

# Test image
parser.add_argument('--checkpoint', default='checkpoint/model_gaussian_20.pth', type=str, help='denoising model name')
opt = parser.parse_args() 
