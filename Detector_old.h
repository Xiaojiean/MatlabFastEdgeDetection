#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mutex>
#include "tools.h"

using namespace std;
using namespace cv;

#define TYPE CV_64F
#define BOOL CV_8U

#define ZERO double(0)
#define MINUS double(-1)
#define FALSE Scalar(0)

typedef unsigned int uint;

typedef struct MyParam{
	double removeEpsilon = 0.248;
	double maxTurn = 35;
	double nmsFact = 0.75;
	int splitPoints = 40;
	uint minContrast = 8;
	uint maxNumOfEdges = 100;

	void function(const Mat& x, const Mat& y, const Mat& dst){
		assert(x.size() == y.size());
		Mat m1 = Mat(x.rows, x.cols, TYPE, ZERO);
		Mat m2 = Mat(y.rows, y.cols, TYPE, ZERO);
		cv::max(x, m1, m1);
		cv::max(y, m2, m2);
		Mat m3 = (m1+m2);
		m3.copyTo(dst);
	}
	uint w = 2;
	double sigma = 0.1;
	uint patchSize = 5;
	bool parallel = true;
} MyParam;

typedef struct Bottom{
	Mat lineVec;
	Mat leftVec;
	Mat rightVec;
	Mat lengthVec;
	Mat p0;
	Mat p1;
	Mat indices;
} Bottom;

typedef struct Handle{
	uint n;
	uint m;
	uint N;
	Mat S;
	Size rSize;
	uint R = 0;
	uint L = 1;
	uint C = 2;
	uint SC = 3;
	uint minC = 4;
	uint maxC = 5;
	uint I0S0 = 6;
	uint S0I1 = 7;	
	uint TOTAL = 8;

} Handle;

typedef struct Container{
	Mat S;
	Mat I;
} Container;

class Detector {
	private:
		Mat _I;
		Mat _E;
		MyParam _prm;
		Bottom _bot;
		Handle _handle;
		uint _pBig;
		unordered_map<uint, Mat>* _data;
		Mat* _pixelScores;
		unordered_map<uint, Mat> _pixels;
		Container* _cArr;
		bool _debug = false;
		std::mutex _mtx;

		/* Bottom Level Processing */
		void extractBottomLevelData();
		void getBottomLevel(Container& c, uint index);
		int getLine(uint n, int x0, int y0, int x1, int y1, Mat& P);
		void getVerticesFromPatchIndices(uint e, uint  v, uint n, uint& x, uint& y);

		/* Core Functions*/
		void getBestSplittingPoints(Mat& split, Mat& dst, uint index);
		void threshold(Mat& L, Mat& t);
		void mergeTilesSimple(const Mat& S1, const Mat& S2, uint index, uint level);
		void findBestResponse(Mat& edge1, Mat& split, Mat& edge2, unordered_map<uint, Mat>& data1, unordered_map<uint, Mat>& data2, uint index, uint level);

		/* Sub Functions */
		void subIm(const Container& cSrc, uint x0, uint y0, uint x1, uint y1, uint w, Container& cDst);
		void getEdgeIndices(const Mat& S, vector<Mat>& v);
		void assignNewPixelScores(Mat& resp, Mat& ind0, Mat& ind1, uint index);
		void putValuesInMat(Mat& resp, Mat& len, Mat& con, Mat& scores, Mat& minC, Mat& maxC, Mat& i0s0, Mat& s0i1, Mat& dest);
		void setNewValues(Mat& ind01, Mat& ind10, Mat& values01, Mat& values10, uint index);
		void setNewValues2(Mat& ind, Mat& values, uint index);
		void setNewPixels(Mat& ind01, Mat& ind10, Mat& curPixels);
		void addIndices(Mat& table, Mat& ind0, Mat& s0, Mat& ind1);
		bool angleInRange(uint ind0, uint s0, uint ind1, uint level);
		bool angleInRange(double ang0, double ang1, uint level);

		/* Post Processing */
		void getScores();
		void removeKey(unordered_map<uint, Mat>& data, uint key);
		bool addEdge(unordered_map<uint, Mat>& data, uint curKey, Mat& E, uint level);

	public:
		Detector(Mat I, MyParam prm);
		~Detector();
		Mat runIm();
		Mat getE(){ return _E;};
		Mat getPixelScores(){
			Mat P = _pixelScores[0];
			P = P / maxValue(P);
			return _pixelScores[0];
		};
		void beamCurves(uint index, uint level, Container* c);
};
