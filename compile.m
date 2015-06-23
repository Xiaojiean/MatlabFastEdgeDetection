clc;
clear;
close all;

mex EdgeDetection.cpp Detector.cpp tools.cpp -I./ -L./lib/ -lopencv_imgproc2410.lib -lopencv_core2410.lib -lopencv_highgui2410.lib