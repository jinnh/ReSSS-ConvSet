input = imread('input.png');
output = imread('Ours_vgg_31.png');

D = cat(4, input, output);
togetGif(D, './unaligned_ours', 0, 1)