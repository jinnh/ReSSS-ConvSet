import argparse
# Training
parser = argparse.ArgumentParser(description="Hyperspectral Image Super-Resolution")
parser.add_argument("--upscale_factor", default=4, type=int, help="super resolution upscale factor")
parser.add_argument('--seed', type=int, default=1,  help='random seed (default: 1)')
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="maximum number of epochs to train")
parser.add_argument('--model_name', default='res3conv', type=str, help="model name (res3conv, res3net)")
parser.add_argument("--dataset", default="CAVE", type=str, help="data name (CAVE, HARVARD)")
parser.add_argument("--lr", type=int, default=5e-4, help="initial  lerning rate")
parser.add_argument("--svdLossWeight", type=float, default=1e-4, help="initial svdLossWeight")
parser.add_argument("--cuda", action="store_true", help="Use cuda")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")
parser.add_argument("--threads", type=int, default=8, help="number of threads for dataloader to use")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")               

# Test
parser.add_argument('--checkpoint', default='checkpoints/CAVE_x4/res3net_4_epoch_50.pth', type=str, help='sr model')
opt = parser.parse_args() 
