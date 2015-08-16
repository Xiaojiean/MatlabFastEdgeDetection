#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tools.h"
#include "Detector.h"
#include "mex.h"


using namespace std;
using namespace cv;
/*
int main( int argc, char** argv )
{
	if( argc != 2)
    {
		println("Usage: EdgeDetection ImagePath");
		return endRun(-1);
    }

	Mat I;
	I = readImage(argv[1]);
	resize(I, I, Size(150,150));
	I.convertTo(I, TYPE);
	I = I / 255;

	MyParam prm;
	Detector d(I,prm);
	Mat E = d.runIm();
	E = E / maxValue(E);
	Mat H;
	hconcat(I, d.getPixelScores(), H);
	hconcat(H, E, H); 

	showImage(H, 1, 3, true);
	println("Finished");
	return 0;
}
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mexPrintf("Run Edge Detection\n");
	if (nrhs != 6) {
		mexErrMsgIdAndTxt("MATLAB:mexcpp:nargin", "MEXCPP requires six input arguments.");
	}
	else if (nlhs != 1) {
		mexErrMsgIdAndTxt("MATLAB:mexcpp:nargout", "MEXCPP requires one output argument.");
	}

	if (!mxIsDouble(prhs[0])) {
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble", "Input Matrix must be a double.");
	}

	for (int i = 1; i < 6; ++i){
		if (!mxIsDouble(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1) {
			mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notScalar", "Input multiplier must be a scalar.");
		}
	}

	MyParam prm;
	double* img1 = (double *)mxGetPr(prhs[0]);
	int cols = (int)mxGetN(prhs[0]);
	int rows = (int)mxGetM(prhs[0]);
	mexPrintf(format("Image Size: %d, %d\n", rows,cols).c_str());
	prm.removeEpsilon = mxGetScalar(prhs[1]);
	prm.maxTurn = mxGetScalar(prhs[2]);
	prm.nmsFact = mxGetScalar(prhs[3]);
	prm.splitPoints = (int)mxGetScalar(prhs[4]);
	prm.minContrast = (int)mxGetScalar(prhs[5]);

	mexPrintf(format("Params: %2.2f, %2.2f, %2.2f, %d, %d\n", prm.removeEpsilon, prm.maxTurn, prm.nmsFact, prm.splitPoints, prm.minContrast).c_str());
	Mat I(rows, cols, TYPE);
	memcpy(I.data, img1, I.rows * I.cols * sizeof(double));
	Detector d(I, prm);
	Mat E = d.runIm();
	plhs[0] = mxCreateDoubleMatrix(E.rows, E.cols, mxREAL);
	memcpy(mxGetPr(plhs[0]), E.data, E.rows * E.cols * sizeof(double));
}