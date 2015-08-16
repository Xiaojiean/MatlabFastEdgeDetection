#include <iostream>
#include <string>
#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Detector.h"
#include "tools.h"
#include <math.h>
#include <thread>


using namespace std;
using namespace cv;

Detector::Detector(Mat I, MyParam prm){
	_prm = prm;
	I.convertTo(_I, TYPE);
	_E = Mat(_I.rows, _I.cols, TYPE, ZERO);

	uint n = _I.rows;
	uint m = _I.cols;
	uint N = n*m;
	_handle.n = n;
	_handle.m = m;
	_handle.N = N;
	_handle.rSize = Size(_handle.N, _handle.N);
	uint w = _prm.w;
	_pBig = _prm.patchSize + 2 * w;
	uint PBig = (uint)pow(_pBig,2);
	uint P = (uint)pow(_prm.patchSize,2);

	uint pairs = nchoosek(4, 2)*P;
	_bot.lineVec = Mat(pairs, P, TYPE, ZERO);
	_bot.leftVec = Mat(pairs, PBig, TYPE, ZERO);
	_bot.rightVec = Mat(pairs, PBig, TYPE, ZERO);
	_bot.lengthVec = Mat(1, pairs, TYPE, MINUS);
	_bot.p0 = Mat(1, pairs, TYPE, MINUS);
	_bot.p1 = Mat(1, pairs, TYPE, MINUS);
	_bot.indices = Mat(pairs, _prm.patchSize, TYPE, MINUS);

	int maxSize = (int)ceil(_handle.N / 8.0);
	_pixelScores = new Mat[maxSize];
	_data = new unordered_map<uint, Mat>[maxSize];
	_cArr = new Container[maxSize];
}

Detector::~Detector(){
	delete[] _pixelScores;
	delete[] _data;
	delete[] _cArr;
}

Mat Detector::runIm(){
	uint n = _handle.n;
	uint m = _handle.m;
	double j = log2(n - 1);
	if (n != m){
		println("Non square image");
	}
	else if (j - floor(j) != 0){
		println("Image size is not a power of 2 plus 1");
	}
	else{
		println("Extract Bottom Level Data");
		double start = tic();
		extractBottomLevelData();
		toc(start);
		
		println("Build Binary Tree");
		start = tic();
		Mat S(_handle.n,_handle.n,TYPE);
		assert(S.isContinuous());
		double* p = (double*)S.data;
		for (int i = 0; i < S.size().area(); ++i){
			*p++ = i;
		}

		uint w = _prm.w;
		Container c;
		Mat padded(_I.rows+2*w,_I.cols+2*w,_I.depth());
		copyMakeBorder(_I, padded, w, w, w, w, IPL_BORDER_REFLECT);
		padded.copyTo(c.I);
		S.copyTo(c.S);
		S.copyTo(_handle.S);
		beamCurves(0,0,&c);
		toc(start);

		println("Create Edge Image");
		start = tic();
		getScores();
		toc(start);
	}
	return _E;
}

void Detector::beamCurves(uint index, uint level, Container* c){
	uint m = c->I.rows;
	uint n = c->I.cols;
	uint w = _prm.w;
	n = n - 2 * w;
	m = m - 2 * w;

	_pixelScores[index] = Mat(_handle.n, _handle.n, TYPE,ZERO);

	if (m == _prm.patchSize && n == _prm.patchSize){
		getBottomLevel(*c, index);
	}
	else{
		Container c0, c1;
		uint mid = (uint)floor(m / 2);
		if (m==n){
			subIm(*c, 0, 0, m-1, mid, w, c0);
			subIm(*c, 0, mid, m-1, n-1, w, c1);
		}
		else if(m>n){
			subIm(*c, 0, 0, mid, mid, w, c0);
			subIm(*c, mid, 0, m-1, n-1, w, c1);
		}
		
		uint t[] = { 2 * index + 1, 2 * index + 2 };
		Container* cArr[] = { &c0, &c1 };

		if (_prm.parallel && level%2 == 0){
			vector<std::thread> tasks;
			for (uint i = 0; i < 2; ++i)
				tasks.push_back(std::thread(std::bind(&Detector::beamCurves, this, t[i], level+1, cArr[i])));
			
			for (uint i = 0; i < tasks.size(); ++i)
				tasks[i].join();
		}
		else{
			for (uint i = 0; i < 2; ++i)
				beamCurves(t[i], level+1, cArr[i]);
		}
		_prm.function(_pixelScores[t[0]], _pixelScores[t[1]], _pixelScores[index]);
		for (uint i = 0; i < 2; ++i){
			_pixelScores[t[i]].release();
		}
		mergeTilesSimple(c0.S, c1.S, index, level);
		for (uint i = 0; i < 2; ++i){
			_data[index].insert(_data[t[i]].begin(), _data[t[i]].end());
			_data[t[i]].clear();
		}
		double j = log2(index + 2);
		if (ceil(j) == j){
			println(format("Level %d Complete", (int)j));
		}
	}
}

void Detector::mergeTilesSimple(const Mat& S1, const Mat& S2, uint index, uint level){

	uint index1 = 2 * index + 1;
	uint index2 = 2 * index + 2;
	unordered_map<uint, Mat>& data1 = _data[index1];
	unordered_map<uint, Mat>& data2 = _data[index2];

	vector<Mat> edgeS1; getEdgeIndices(S1, edgeS1);
	vector<Mat> edgeS2; getEdgeIndices(S2, edgeS2);
	Mat split;

	vector<std::thread> tasks;

	if (S1.rows != S1.cols){
		getBestSplittingPoints(edgeS1[2], split, index);

		for (int i = 0; i < 4; ++i){
			if (i == 2) continue;
			for (int j = 1; j < 4; ++j){
				if (i == j) continue;
				findBestResponse(edgeS1[i], split, edgeS2[j], data1, data2, index, level);
			}
		}
	}
	else{
		getBestSplittingPoints(edgeS1[3], split, index);
		for (uint i = 0; i < 3; ++i){
			for (uint j = 0; j<4; ++j){
				if (i == j || j == 1) continue;
				findBestResponse(edgeS1[i], split, edgeS2[j], data1, data2, index, level);
			}
		}
	}
}

void Detector::findBestResponse(Mat& edge1, Mat& split, Mat& edge2, unordered_map<uint, Mat>& data1, unordered_map<uint, Mat>& data2, uint index, uint level){
	double* e1, *e2, *s;
	int i, j, k;
	double ind0, ind1, s0;
	for (e1 = (double*)edge1.data, i = 0; i < edge1.size().area(); ++i){
		ind0 = *e1++;
		for (e2 = (double*)edge2.data, j = 0; j < edge2.size().area(); ++j){
			ind1 = *e2++;
			if (ind0 == ind1){ continue; }

			Mat bestData(_handle.TOTAL, 1, TYPE, ZERO);
			Mat bestValues1;
			Mat bestValues2;
			assert(bestData.isContinuous());
			double* bd = (double*)bestData.data;
			double bestScore = 0;
			double bestS0 = -1;

			for (s = (double*)split.data, k = 0; k < split.size().area(); ++k){
				s0 = *s++;
				if (ind0 == s0 || ind1 == s0){ continue; }
				
				if (!angleInRange((uint)ind0, (uint)s0, (uint)ind1, level)){
					continue;
				}

				double ind0s0 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)ind0, (int)s0);
				double s0ind1 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)s0, (int)ind1);
				if (data1.count((uint)ind0s0) != 1 || (data2.count((uint)s0ind1) != 1)){continue;}

				Mat indValues1 = data1.at((int)ind0s0).clone();
				Mat indValues2 = data2.at((int)s0ind1).clone();

				double len1 = indValues1.at<double>(_handle.L);
				double len2 = indValues2.at<double>(_handle.L);
				double resp1 = indValues1.at<double>(_handle.R);
				double resp2 = indValues2.at<double>(_handle.R);

				double len = len1 + len2;
				assert(len1 >=1 && len2>=1);
				double resp = resp1 + resp2;
				double con;
				if (_prm.w == 0) con = resp/len;
				else con = resp/len/2/_prm.w;
				double thresh = _prm.sigma*sqrt(2 * log(6 * _handle.N) / _prm.w / len / 2);
				double score = abs(con) - thresh;

				if (score > bestScore || bestS0<0){

					bd[_handle.C] = con;
					bd[_handle.I0S0] = ind0s0;
					bd[_handle.L] = len;
					bd[_handle.R] = resp;
					bd[_handle.S0I1] = s0ind1;
					bd[_handle.SC] = score;
					bestValues1 = indValues1.clone();
					bestValues2 = indValues2.clone();
					bestS0 = s0;
					bestScore = score;
				}
			}

			if (bestS0 >= 0){
				double ind01 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)ind0, (int)ind1);
				double ind10 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)ind1, (int)ind0);
				double ind1s0 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)ind1, (int)bestS0);
				double s0ind0 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)bestS0, (int)ind0);

				if (bd[_handle.L] <= _prm.minContrast){
					bd[_handle.minC] = bd[_handle.C];
					bd[_handle.maxC] = bd[_handle.C];
				}
				else{
					double minC = std::min(bestValues1.at<double>(_handle.minC), bestValues2.at<double>(_handle.minC));
					double maxC = std::max(bestValues1.at<double>(_handle.maxC), bestValues2.at<double>(_handle.maxC));
					bd[_handle.minC] = minC;
					bd[_handle.maxC] = maxC;
				}
				double bestValue = abs(bd[_handle.R]);
				if (bestValue > 0){
					Mat& px = _pixelScores[index];
					assert(px.isContinuous());
					double* p = (double*)px.data;
					uint i0 = (int)ind0, i1 = (uint)ind1;
					p[i0] = std::max(p[i0], bestValue);
					p[i1] = std::max(p[i1], bestValue);
				}

				Mat bestData2 = bestData.clone();
				assert(bestData2.isContinuous());
				double* bd2 = (double*)bestData2.data;
				bd2[_handle.R] = -bd[_handle.R];
				bd2[_handle.C] = -bd[_handle.C];
				bd2[_handle.minC] = -bd[_handle.maxC];
				bd2[_handle.maxC] = -bd[_handle.minC];
				bd2[_handle.I0S0] = ind1s0;
				bd2[_handle.S0I1] = s0ind0;

				_data[index].insert(pair<uint, Mat>((uint)ind01, bestData.clone()));
				_data[index].insert(pair<uint, Mat>((uint)ind10, bestData2.clone()));
			}
		}
	}

}

void Detector::getBestSplittingPoints(Mat& split,Mat& dst,uint index){
	int len = split.size().area();
	if ( (_prm.splitPoints >= len) || (_prm.splitPoints == 0) ){
		split.copyTo(dst);
		return;
	}

	Mat splitScore;
	split.convertTo(split, TYPE);
	copyIndices(_pixelScores[index], split, splitScore);

	int P = _prm.splitPoints;
	Mat idx, dst2;
	getHighestKValues(splitScore, dst2, idx, P);
	
	for (int i = 0; i < idx.size().area(); ++i){
		dst.push_back(split.at<double>((int)idx.at<double>(i)));
	}
}

void Detector::getEdgeIndices(const Mat& S, vector<Mat>& v){
	uint m = S.rows;
	uint n = S.cols;

	v.push_back(S.col(0).clone().reshape(0,1));
	v.push_back(S.row(0).clone());
	v.push_back(S.col(n - 1).clone().reshape(0,1));
	v.push_back(S.row(m - 1).clone());
}

void Detector::subIm(const Container& cSrc, uint x0, uint y0, uint x1, uint y1, uint w, Container& cDst){
	cDst.I = cSrc.I(Range(x0,2 * w + x1+1), Range(y0,2 * w + y1+1)).clone();
	cDst.S = cSrc.S(Range(x0,x1+1), Range(y0,y1+1)).clone();
}

void Detector::getBottomLevel(Container& c, uint index){
	Mat I = c.I.clone();
	Mat S = c.S.clone();

	uint m = I.rows;
	uint n = I.cols;
	uint N = m*n;
	I = I.reshape(0, N);

	Mat curPixels;
	copyIndices(S, _bot.indices, curPixels);
	Mat resp;
	if (_prm.w == 0){
		resp = _bot.lineVec *I;
	}
	else{
		resp = (_bot.leftVec - _bot.rightVec)* I;
	}
	resp = resp.t();
	Mat ind0,ind1,len; 
	copyIndices(S,_bot.p0,ind0);
	copyIndices(S,_bot.p1,ind1);
	len = _bot.lengthVec.clone();
	
	
	Mat ind01, ind10, con;
	matSub2ind(_handle.rSize, ind0, ind1,ind01);
	matSub2ind(_handle.rSize, ind1, ind0,ind10);
	if (_prm.w == 0){
		divide(resp,len,con);
	}
	else{
		divide(resp, 2*_prm.w*len, con);
	}

	setValueIfTrue(ZERO , con, len <= 0);
	
	if (_prm.removeEpsilon > 0){
		Mat good = abs(con) >= (_prm.removeEpsilon*_prm.sigma);
		if (sum(good)[0] == 0){
			return;
		}
		keepTrue(con, good, con);
		keepTrue(len, good, len);
		keepTrue(resp, good, resp);
		keepTrue(ind01, good, ind01);
		keepTrue(ind10, good, ind10);
		keepTrue(ind0, good, ind0);
		keepTrue(ind1, good, ind1);
		keepSelectedRows(curPixels, good, curPixels);
	}
	
	Mat minC = con.clone(), maxC = con.clone(), thresh, scores;
	threshold(len, thresh);
	scores = abs(con) - thresh;
	setValueIfTrue(numeric_limits<double>::min(), scores, len <= 0);

	assignNewPixelScores(resp, ind0, ind1, index);
	
	Mat s0 = Mat(resp.size(), TYPE, ZERO);
	
	Mat data01,data10; 
	putValuesInMat(resp, len, con, scores, minC, maxC, s0, s0, data01);
	Mat respB = -resp, conB = -con, minCB = -maxC, maxCB = -minC;
	putValuesInMat(respB, len, conB, scores, minCB, maxCB, s0, s0, data10);

	setNewValues(ind01, ind10, data01, data10, index);
	setNewPixels(ind01, ind10, curPixels);
}

void Detector::setNewPixels(Mat& ind01, Mat& ind10, Mat& curPixels){
	assert(ind01.cols == ind10.cols && ind01.cols == curPixels.rows);

	double* p01 = (double*)ind01.data;
	double* p10 = (double*)ind10.data;

	assert(ind01.isContinuous() && ind10.isContinuous());
	std::chrono::milliseconds interval(0);
	while (true){
		if (_mtx.try_lock()){
			break;
		}
		else{
			this_thread::sleep_for(interval);
		}
	}
	for (int i = 0; i < ind01.cols; ++i){
		_pixels.insert(pair<uint, Mat>((uint)*p01++, curPixels.row(i).clone()));
		_pixels.insert(pair<uint, Mat>((uint)*p10++, curPixels.row(i).clone()));
	}
	_mtx.unlock();
}

void Detector::setNewValues(Mat& ind01, Mat& ind10, Mat& values01, Mat& values10, uint index){
	Mat ind;
	hconcat(ind01, ind10, ind);
	Mat values;
	hconcat(values01, values10, values);
	setNewValues2(ind, values, index);
}

void Detector::setNewValues2(Mat& ind, Mat& values, uint index){
	assert(ind.isContinuous());
	double* p = (double*)ind.data;
	for (int i = 0; i < ind.size().area(); ++i){
		_data[index].insert(pair<uint, Mat>((uint)*p++, values.col(i).clone()));
	}
}

void Detector::putValuesInMat(Mat& resp, Mat& len, Mat& con, Mat& scores, Mat& minC, Mat& maxC, Mat& i0s0, Mat& s0i1, Mat& dest){
	Mat A, B, C, D;
	vconcat(resp, len, A);
	vconcat(con, scores, B);
	vconcat(minC, maxC, C);
	vconcat(i0s0, s0i1, D);
	Mat E, F;
	vconcat(A, B, E);
	vconcat(C, D, F);
	vconcat(E, F, dest);
}

void Detector::assignNewPixelScores(Mat& resp, Mat& ind0, Mat& ind1,uint index){
	Mat values = resp.clone();

	Mat newPixelScores0 = Mat(_handle.n,_handle.n, TYPE, ZERO);
	Mat newPixelScores1 = Mat(_handle.n, _handle.n, TYPE, ZERO);

	Mat idx;
	sortIdx(values, idx, CV_SORT_ASCENDING);

	reorder(idx, values, values);
	reorder(idx, ind0, ind0);
	reorder(idx, ind1, ind1);

	setValuesInInd(values, ind0, newPixelScores0);
	setValuesInInd(values, ind1, newPixelScores1);
	Mat newPixelScores;
	max(newPixelScores0, newPixelScores1, newPixelScores);
	_prm.function(newPixelScores, _pixelScores[index], _pixelScores[index]);
}

void Detector::threshold(Mat& L,Mat& T){
	uint w = 2 * _prm.w;
	Mat wL = w*L;
	Mat div; divide(2 * log(6 * _handle.N), wL, div);
	Mat sq; sqrt(div, sq);
	T = _prm.sigma*sq;
}

void Detector::extractBottomLevelData(){
	uint index = 0;
	uint pSize = _prm.patchSize;
	Mat curIndices;

	for (uint e0 = 1; e0 <= 3; ++e0){
		for (uint e1 = e0 + 1; e1 <= 4; ++e1){
			for (uint v0 = 0; v0 <pSize; ++v0){
				uint x0, y0;
				getVerticesFromPatchIndices(e0, v0, pSize, x0, y0);
				for (uint v1 = 0; v1 < pSize; ++v1){
					uint x1, y1;
					getVerticesFromPatchIndices(e1, v1, pSize, x1, y1);
					_bot.p0.at<double>(index) = sub2ind(pSize, pSize, x0, y0);
					_bot.p1.at<double>(index) = sub2ind(pSize, pSize, x1, y1);
					Mat P(pSize, pSize,TYPE, ZERO);
					int L = getLine(pSize, x0, y0, x1, y1, P);
					curIndices.release();
					findIndices(P, curIndices);
					curIndices.copyTo(_bot.indices(Range(index, index + 1), Range(0, curIndices.cols)));

					Mat line = P;
					Mat left(_pBig,_pBig,TYPE,ZERO);
					Mat right(_pBig,_pBig,TYPE,ZERO);

					int dx = x1 - x0;
					int dy = y1 - y0;

					if (abs(dx) > abs(dy)){
						dy = -sign(dx);
						dx = 0;
					}
					else{
						dx = sign(dy);
						dy = 0;
					}
					dx = sign(dx);
					dy = sign(dy);

					int x, y;
					int w = _prm.w;
					for (int k = 1; k <= w; ++k){
						x = w + dx*k;
						y = w + dy*k;
						right(Range(x, x + pSize), Range(y, y + pSize)) += line;
						x = w - dx*k; 
						y = w - dy*k;
						left(Range(x, x + pSize), Range(y, y + pSize)) += line;
					}

					right = min(right,1);
					left = min(left,1);
					line.reshape(0, 1).copyTo(_bot.lineVec.row(index));
					left.reshape(0, 1).copyTo(_bot.leftVec.row(index));
					right.reshape(0, 1).copyTo(_bot.rightVec.row(index));
					_bot.lengthVec.at<double>(index) = L;
					index = index + 1;
				}
			}
		}
	}
} 

int Detector::getLine(uint n, int x0, int y0, int x1, int y1, Mat& P){
	int dx = abs(x1 - x0);
	int dy = abs(y1 - y0);
	int L = max(dx, dy);

	int sx;
	if (x0 < x1) sx = 1;
	else sx = -1;
	int sy;
	if(y0 < y1) sy = 1;
	else sy = -1;

	int err = dx - dy;
	bool first = true;

	while(true){
		P.at<double>(x0, y0) = 1;
		if (first){
			P.at<double>(x0, y0) = 0.5;
			first = false;
		}
		if (x0 == x1 && y0 == y1){
			P.at<double>(x0, y0) = 0.5;
			break;
		}
		int e2 = 2 * err;
		if (e2 > -dy){
			err = err - dy;
			x0 = x0 + sx;
		}
		if (e2 < dx){
			err = err + dx;
			y0 = y0 + sy;
		}
	}
	return L;
}

void Detector::getVerticesFromPatchIndices(uint e, uint  v, uint n, uint& x, uint& y){
	switch (e){
	case 1:
		x = v;
		y = 0;
		break;
	case 2:
		x = 0;
		y = v;
		break;
	case 3:
		x = v;
		y = n-1;
		break;
	case 4:
		x = n-1;
		y = v;
		break;
	default:
		x = -1;
		y = -1;
	}
}

void Detector::getScores(){
	uint n = _handle.n;
	
	Mat selected(n,n,BOOL,FALSE);

	unordered_map<uint,Mat>& data = _data[0];
	priority_queue<pair<double,uint>> q;
	unordered_map<uint, Mat>::iterator it;
	for (it = data.begin(); it != data.end(); ++it){
		Mat tuple = it->second;
		double* p = (double*)tuple.data;


		double sc = p[_handle.SC], con = p[_handle.C], minC = p[_handle.minC], maxC = p[_handle.maxC];
		if (sc > 0){
			bool minTest = _prm.minContrast == 0 || (con > 0 && minC >= (con / 2)) || (con < 0 && maxC <= (con / 2));
			if (minTest){
				q.push(pair<double, uint>(sc, it->first));
			}
		}
	}

	uint counter = 0;
	size_t before = q.size();
	while (!q.empty()){
		double curScore = q.top().first;
		uint curKey = q.top().second;
		q.pop();
		Mat E(n, n, BOOL, FALSE);

		if (!addEdge(data, curKey, E, 1)){
			continue;
		}
		else if (_prm.nmsFact == 0){
			Mat t;
			E.convertTo(t, TYPE);
			Mat cur = t*curScore;
			max(_E, cur, _E);
			++counter;
		}
		else{
			if (_debug){
				Mat e(_handle.n, _handle.n, TYPE, double(1));
				int r, c;
				ind2sub(curKey, _handle.N, _handle.N, r, c);
				double* p = (double*)e.data;
				p[r] = 0;
				p[c] = 0;
				Mat H;
				Mat k;
				E.convertTo(k, TYPE);
				hconcat(k, e, H);
				showImage(H, 1, 3, true);
			}

			Mat curI = E.clone();
			Size imSize = curI.size();
			Mat sub1, sub2, sub3, sub4;
			Mat horFalse(1, imSize.width, BOOL, FALSE);
			Mat verFalse(imSize.height, 1, BOOL, FALSE);
			
			vconcat(horFalse, curI(Range(0,imSize.height-1),Range::all()), sub1);
			vconcat(curI(Range(1, imSize.height), Range::all()), horFalse, sub2);
			hconcat(curI(Range::all(), Range(1, imSize.width)), verFalse, sub3);
			hconcat(verFalse , curI(Range::all(), Range(0, imSize.width-1)), sub4);

			Mat curIdialate = curI | sub1 | sub2 | sub3 | sub4;

			double L = sum(curI)[0];
			Mat coor;
			bitwise_and(curIdialate, selected, coor);
			double nmsScore = sum(coor)[0]/L;
			if (nmsScore < _prm.nmsFact){
				removeKey(data, curKey);
				++counter;
				selected = selected | curIdialate;
				Mat t;
				E.convertTo(t, TYPE);
				Mat cur = t*curScore;
				max(_E, cur , _E);
				if (counter > _prm.maxNumOfEdges){
					return;
				}
			}
		}
	}
	println(format("EdgesBeforeNMS = %d\nEdgesAfterNMS = %d", before, counter));
}

void Detector::removeKey(unordered_map<uint, Mat>& data, uint key){
	int a1, a2;
	uint rows = _handle.rSize.height, cols = _handle.rSize.width;

	ind2sub(key, rows, cols, a1, a2);
	uint negKey = sub2ind(rows, cols, a2, a1);
	if (data.count(key) == 1){
		data.erase(key);
	}
	if (data.count(negKey)){
		data.erase(negKey);
	}
}

bool Detector::addEdge(unordered_map<uint, Mat>& data, uint curKey, Mat& E, uint level){
	int maxLevel = (int)(2*log2(_handle.n)-log2(_prm.patchSize));
	if (data.count(curKey) == 0 || level == maxLevel){
		return false;
	}
	Mat curData = data.at(curKey).clone();
	uint i0s0 = (uint)curData.at<double>(_handle.I0S0);
	uint s0i1 = (uint)curData.at<double>(_handle.S0I1);

	i0s0 = (1 - (curKey == i0s0))*i0s0;
	s0i1 = (1 - (curKey == s0i1))*s0i1;
	if (i0s0 > 0 && s0i1 > 0){
		if (!addEdge(data, (uint)i0s0, E, level + 1)) return false;
		if (!addEdge(data, (uint)s0i1, E, level + 1)) return false;		
	}
	else{
		if (_pixels.count(curKey) == 0){
			println("Pixels Problem");
			return false;
		}
		Mat& pixels = _pixels.at(curKey);
		assert(pixels.isContinuous());
		double* p = (double*)pixels.data;
		bool* e = (bool*)E.data;

		bool flag = false;
		for (int i = 0; i < pixels.size().area(); ++i){
			int curPixel = (int)*p++;
			if (curPixel >= 0){
				flag = true;
				e[curPixel] = true;
			}
		}
		if (!flag){
			return false;
		}
	}
	return true;
}

bool Detector::angleInRange(uint ind0, uint s0, uint ind1, uint level){
	int ang0 = indToAngle(_handle.n, _handle.n, ind0, s0);
	int ang1 = indToAngle(_handle.n, _handle.n, s0, ind1);
	return(angleInRange(ang0, ang1, level));
}

bool Detector::angleInRange(double ang0, double ang1, uint level){
	int diff = (int)(ang0 - ang1)+360;
	diff %= 360;
	assert(diff >= 0 && diff < 360);
	double J = floor(log2(_handle.N)) - 2*log2(_prm.patchSize - 1)-1;
	double curMaxTurn = _prm.maxTurn*(2 - level / J);
	assert(curMaxTurn >= _prm.maxTurn && curMaxTurn <= _prm.maxTurn * 2);
	return (diff <= curMaxTurn) || ((360 - diff) <= curMaxTurn);
}