%%  Image Regression with Structure Cycle Consistency for Heterogeneous Change Detection
%{
Code: AGSCC - 2022
This is a test program for the Adaptive Graph and Structure Cycle Consistency (AGSCC) for heterogeneous change detection.

If you use this code for your research, please cite our paper. Thank you!

Sun, Yuli, et al. "Image Regression with Structure Cycle Consistency for Heterogeneous Change Detection."
IEEE Transactions on Neural Networks and Learning Systems, 2022
===================================================
%}

clear;
close all
addpath('auxi_funcs')
%% load dataset

addpath('datasets')

% Please note that the forward and backward detection results are not the same. 
% When the forward result is not satisfactory, try swapping the input order of image_t1 and image_t2 to get the backward change detection result. 
% In the future we will consider fusing the forward and backward results to improve detection performance.
% #2-Img7, #3-Img17, and #4-Img5 can be found at Professor Max Mignotte's webpage (http://www-labs.iro.umontreal.ca/~mignotte/).

% #1-Italy, #2-Img7, #3-Img17, #4-Img5, #5-Shuguang, #6-California, #7-Texas

dataset = '#1-Italy';

if strcmp(dataset,'others') == 0
    Load_dataset
elseif strcmp(dataset,'others') == 1
    image_t1 = imread('Italy_1.bmp');
    image_t2 = imread('Italy_2.bmp');
    gt = imread('Italy_gt.bmp');
    Ref_gt = double(gt(:,:,1));
    figure;
    subplot(131);imshow(image_t1,[]);title('imaget1')
    subplot(132);imshow(image_t2,[]);title('imaget2')
    subplot(133);imshow(Ref_gt,[]);title('Refgt')
    image_t1 = double (image_t1);
    image_t2 = double (image_t2);
end
fprintf(['\n Data loading is completed...... ' '\n'])
%% Parameter setting

% With different parameter settings, the results will be a little different
% Ns: the number of superpxiels,  A larger Ns will improve the detection granularity, but also increase the running time. 5000 <= Ns <= 10000 is recommended.
% Niter_AGL: the maximum iteration number of Adaptive Graph Learning, Niter_AGL = 5 is recommended.
% Niter_SCC: the maximum iteration number of Structure Cycle Consistencyand , Niter_SCC =10 is recommended.
% eta: 0 < eta < 1, eta = 0.5 is recommended.
% beta, gamma, lambda: regularization parametes; beta = gamma = 5, lambda = 0.1  are recommended.
% seg: balanced parameter of the MRF segmentation. The smaller the seg, the smoother the CM. 0.025<= seg <=0.1 is recommended.

opt.Ns = 5000;
opt.Niter_AGL = 5;
opt.Niter_SCC = 10;
opt.eta = 0.5; % for #6-California, opt.eta = 0.3 is better.
opt.beta = 5;
opt.gamma  = 5;
opt.lambda  = 0.1;
opt.seg = 0.05;  % for #7-Texas, opt.seg = 0.01 is better.
%% AGSCC

t_o = clock;
fprintf(['\n AGSCC is running...... ' '\n'])

%------------- Preprocessing: Superpixel segmentation and feature extraction---------------%

t_p1 = clock;
Compactness = 1;
[sup_img,Ns] =  SuperpixelSegmentation(image_t1,opt.Ns,Compactness);
[t1_feature,t2_feature,norm_par] = Feature_extraction(sup_img,image_t1,image_t2) ;% MVE;MSM
fprintf('\n');fprintf('The computational time of Preprocessing (t_p1) is %i \n', etime(clock, t_p1));
fprintf(['\n' '====================================================================== ' '\n'])

%------------- Algorithm 1: Adaptive Graph Learning---------------%
t_p2 = clock;
[Sx,Kmat,wm_x,wm_x_rel] = AGSCC_AdaptiveGraphLearning(t1_feature,opt);
fprintf('\n');fprintf('The computational time of Adaptive Graph Learning (t_p2) is %i \n', etime(clock, t_p2));
fprintf(['\n' '====================================================================== ' '\n'])

%------------- Algorithm 2: Structure Cycle Consistency based Image Regression---------------%
% image_t1 ----> image_t2

t_p3 = clock;
sum_wmx = sum(wm_x);
opt.beta = sum_wmx * opt.beta;
opt.lambda = sum_wmx * opt.lambda;
opt.gamma  =sum_wmx * opt.gamma ; 
opt.mu = sum_wmx * 0.4;
[Xpie,regression_t1,delt,wm_y,RelDiff] = AGSCC_StructureCycleConsistencyRegression(t1_feature,t2_feature,Sx,Kmat,wm_x,opt);
fprintf('\n');fprintf('The computational time of Image Regression (t_p3) is %i \n', etime(clock, t_p3));
fprintf(['\n' '====================================================================== ' '\n'])

%------------- Algorithm 3: MRF segmentation---------------%

t_p4 = clock;
delt_new (1:size(image_t2,3),:) = delt(:,:,1);
delt_new (size(image_t2,3)+1:2*size(image_t2,3),:) = delt(:,:,2);
delt_new (2*size(image_t2,3)+1:3*size(image_t2,3),:) = delt(:,:,3);

[CM_map,labels] = MRFsegmentation(sup_img,opt.seg,delt_new);
fprintf('\n');fprintf('The computational time of MRF segmentation (t_p4) is %i \n', etime(clock, t_p4));
fprintf(['\n' '====================================================================== ' '\n'])

fprintf('\n');fprintf('The total computational time of AGSCC (t_total) is %i \n', etime(clock, t_o));
%% Displaying results

fprintf(['\n' '====================================================================== ' '\n'])
fprintf(['\n Displaying the results...... ' '\n'])

%---------------------AUC PCC F1 KC ----------------------%

n=500;
Ref_gt = Ref_gt/max(Ref_gt(:));
DI_tmep = sum(delt_new.^2,1);
DI  = suplabel2DI(sup_img,DI_tmep);
[TPR, FPR]= Roc_plot(DI,Ref_gt,n);
[AUC, Ddist] = AUC_Diagdistance(TPR, FPR);
[tp,fp,tn,fn,fplv,fnlv,~,~,pcc,kappa,imw]=performance(CM_map,Ref_gt);
F1 = 2*tp/(2*tp + fp + fn);
result = 'AUC is %4.3f; PCC is %4.3f; F1 is %4.3f; KC is %4.3f \n';
fprintf(result,AUC,pcc,F1,kappa)

%------------Regression image,  Difference imag and Change map --------------%

figure; plot(FPR,TPR);title('ROC curves');
[RegImg,~,~] = suplabel2ImFeature(sup_img,regression_t1,size(image_t2,3));% t1--->t2
RegImg = DenormImage(RegImg,norm_par(size(image_t1,3)+1:end));
figure;
if strcmp(dataset,'#7-Texas') == 1
    subplot(131);imshow(6*uint16(RegImg(:,:,[5 4 3])));title('Regression image')
elseif strcmp(dataset,'#6-California') == 1
    load('California.mat')
    image_t1_temp =image_t2;
    image_t2 = image_t1;
    image_t1 = image_t1_temp;
    min_t2 = min(image_t2(:));
    max_t2 = max(image_t2(:));
    subplot(131);imshow((max_t2-min_t2)*RegImg+min_t2,[]);title('Regression image')
elseif strcmp(dataset,'#4-Img5') == 1
    subplot(131);imshow(uint8(exp(RegImg*3.75+1.8)));title('Regression image')
else
    subplot(131);imshow(uint8(RegImg));title('Regression image')
end
subplot(132);imshow(remove_outlier(DI),[]);title('Difference image')
subplot(133);imshow(CM_map,[]);title('Change mape')

if F1 < 0.3
   fprintf('\n');disp('Please exchange the order of the input images OR select the appropriate opt.seg for AGSCC!')
end  
