clc;
clear;
close all;

tic;
imStr = 'Sqr.png';
prm = getPrm();
I = im2double(imread(imStr));
%I = imresize(I,[200 250]);
if ndims(I) == 3
    I = rgb2gray(I);
end
E = runIm(imStr,prm);
E = E./max(E(:));
figure;
subplot(1,2,1);
imshow(I);
subplot(1,2,2);
imshow(E);
toc;