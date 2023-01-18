%% generate dataset
rng(0)
addpath(genpath('lib'));

basedir = './ICVL/';
sz = 512;
preprocess = @(x)(center_crop(rot90(x), sz, sz));

%% for iid gaussian

% download icvl data and store the data in this folder
datadir = './ICVL/data_all/';
g1 = load(fullfile(basedir, '_meta_gauss.mat')); 
g2 = load(fullfile(basedir, '_meta_gauss_2.mat'));
fns = [g1.fns;g2.fns];

% gaussian noise 30 50 70
for sigma = [30 50 70]
    newdir = fullfile(basedir, ['icvl_', num2str(sz), '_', num2str(sigma)]);
    generate_dataset(datadir, fns, newdir, sigma, 'rad', preprocess);
end

% gaussian blind noise
newdir = fullfile(basedir, ['icvl_', num2str(sz), '_', 'blind']);
generate_dataset_blind(datadir, fns, newdir, 'rad', preprocess);

%% for non-iid gaussian
% g1 = load(fullfile(basedir, '_meta_complex.mat')); 
% g2 = load(fullfile(basedir, '_meta_complex_2.mat'));
% fns = [g1.fns;g2.fns];

% datadir = './ICVL/data_all/';
% sigmas = [10 30 50 70];
% newdir = fullfile(basedir, ['icvl_', num2str(sz), '_', 'noniid']);
% generate_dataset_noniid(datadir, fns, newdir, sigmas, 'rad', preprocess);
%%% for non-iid gaussian + stripe
% newdir = fullfile(basedir, ['icvl_', num2str(sz), '_', 'stripe']);
% generate_dataset_stripe(datadir, fns, newdir, sigmas, 'rad', preprocess);
% %%% for non-iid gaussian + deadline
% newdir = fullfile(basedir, ['icvl_', num2str(sz), '_', 'deadline']);
% generate_dataset_deadline(datadir, fns, newdir, sigmas, 'rad', preprocess);
% %%% for non-iid gaussian + impluse
% newdir = fullfile(basedir, ['icvl_', num2str(sz), '_', 'impulse']);
% generate_dataset_impulse(datadir, fns, newdir, sigmas, 'rad', preprocess);
%%% for mixture noise
% newdir = fullfile(basedir, ['icvl_', num2str(sz), '_','mixture']);
% generate_dataset_mixture(datadir, fns, newdir, sigmas, 'rad', preprocess);
