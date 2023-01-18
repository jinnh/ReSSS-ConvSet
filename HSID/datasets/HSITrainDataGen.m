clear 
clc
close all

dataset = 'ICVL';
savePath = './ICVL/ICVL_train_64';
if ~exist(savePath, 'dir')
    mkdir(savePath)
end

%% obtian all the training hyperspectral image

% training data
train_fns = './ICVL/train_fns';
t1 = load(train_fns);
srfileNames = t1.fns;

% download icvl data and store the data in this folder
srPath = './ICVL/data_all/';
number = length(srfileNames);

train = ''; % all train file name

for index = 1 : 1
    name = char(srfileNames(index));

    disp(['-----deal with:',num2str(index),'----name:',name]);  
    
    %% normalization
    singlePath= [srPath,'\', name];
 
    source = load(singlePath);
    source = source.rad;
    Clean = normalized(source);
    
    width = size(source,1);

    %% obtian hyperspectral patches
 
    stride = (width-1024)/2;
    DATA = Clean(stride+1:width-stride,25:1368,:);

    num=0;
    for i=1:16
        for j=1:21
           Clean = DATA(1+64*(i-1):64*i,1+64*(j-1):64*j,:); 
           save([savePath,'/',name(1:size(name,2)-4), '_' ,int2str(num), '.mat'], 'Clean')
           train = strvcat(char(train), char([name,'_',num2str(num),'.mat']));
           num = num + 1;
        end
    end  
    clear Clean
end

%% save training and testing filename
g1 = load(fullfile('./ICVL/_meta_gauss.mat'));
g2 = load(fullfile('./ICVL/_meta_gauss_2.mat'));
test_gaussian = [g1.fns;g2.fns];
test_gaussian = char(test_gaussian);

c1 = load(fullfile('./ICVL/_meta_complex.mat')); % load fns
c2 = load(fullfile('./ICVL/_meta_complex_2.mat'));
test_complex = [c1.fns;c2.fns];
test_complex = char(test_complex);

save('./ICVL/icvl_train_test_filename.mat', 'train', 'test_gaussian', 'test_complex')
