clear 
clc
close all

dataset = 'CAVE';
upscale = 4;
mode = {'train', 'test'};
train = ''; % all train file name
test = '';  % all test file name

savePath = ['./HSI/CAVE/TrainTestMAT/',num2str(upscale)];
if ~exist(savePath, 'dir')
    mkdir(savePath)
end

%% obtian all the original hyperspectral image
for d = 1:2

    % download the CAVE dataset and split the dataset into train_data and test_data
    if strcmp(mode(d), 'train')
        srPath = './HSI/CAVE/Train';
        disp('-----deal with: training data');
    else
        srPath = './HSI/CAVE/Test';
        disp('-----deal with: testing data');
    end

    srFile=fullfile(srPath);
    srdirOutput=dir(fullfile(srFile));
    srfileNames={srdirOutput.name}';
    number = length(srfileNames)-2;

    for index = 1 : length(srfileNames)
        name = char(srfileNames(index));
        if(isequal(name,'.')||... % remove the two hidden folders that come with the system
               isequal(name,'..'))
                   continue;
        end
        disp(['-----deal with:',num2str(index-2),'----name:',name]);     

        singlePath= [srPath,'\', name, '\', name];
        singleFile=fullfile(singlePath);
        srdirOutput=dir(fullfile(singleFile,'/*.png'));
        singlefileNames={srdirOutput.name}';
        Band = length(singlefileNames);
        source = zeros(512*512, Band);
        for i = 1:Band
            srName = char(singlefileNames(i));
            if strcmp(name, 'watercolors_ms')
                srImage = rgb2gray(imread([singlePath,'/',srName]));
            else
                srImage = imread([singlePath,'/',srName]);
            end
            if i == 1
                width = size(srImage,1);
                height = size(srImage,2);
            end
            source(:,i) = srImage(:);   
        end

       %% normalization
        imgz=double(source(:));
        imgz=imgz./65535;
        img=reshape(imgz,width*height, Band);

       %% obtian HR and LR hyperspectral image
        hrImage = reshape(img, width, height, Band);

        HR = modcrop(hrImage, upscale);
        LR = imresize(HR,1/upscale,'bicubic'); %LR  
        save([savePath,'/',name,'.mat'], 'HR', 'LR')

        if strcmp(mode(d), 'train')
            train = strvcat(char(train), char([name,'.mat']));
        else
            test = strvcat(char(test), char([name,'.mat']));
        end

        clear source
        clear HR
        clear LR
    end
end

%% save training and testing filename
save('./HSI/CAVE/cave_train_test_filename.mat', 'train', 'test')

