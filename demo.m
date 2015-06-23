clc;
clear;
close all;

tic;
prm = getPrm();
I = imread('Sines.png');
E = runIm(I,prm);
E = E./max(E(:));
figure;
subplot(1,2,1);
imshow(I);
subplot(1,2,2);
imshow(E);
toc;