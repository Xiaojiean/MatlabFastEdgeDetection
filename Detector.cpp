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

Detector::Detector(Mat& I, MyParam& prm){
	_prm = prm;
	Mat filter = (Mat_<double>(1, 5) << 1, 1, 0, -1, -1);
	I.convertTo(_I, TYPE);
	filter.convertTo(_filter, TYPE);
	_w = (_filter.size().area() - 1) / 2;
	_E = Mat(_I.rows, _I.cols, TYPE, ZERO);
	filter2D(_I, _dX, TYPE, _filter);
	filter2D(_I, _dY, TYPE, _filter.t());
	//cout << _filter << endl;
	//Mat a = abs(_dY);
	//showImage(a, 1, 3, true);
	uint m = _I.rows;
	uint n = _I.cols;
	uint N = n*m;
	_handle.m = m;
	_handle.n = n;
	_handle.N = N;
	_prm.patchSize = min(m, n)*5/129;
	cout << _prm.patchSize << endl;
	_handle.rSize = Size(_handle.N, _handle.N);
	int maxSize = (int)ceil(_handle.N / 3.0);
	_pixelScores = new Mat[maxSize];
	_data = new unordered_map<uint, Mat>[maxSize];
}

Detector::~Detector(){
	delete[] _pixelScores;
	delete[] _data;
}

Mat Detector::runIm(){
	uint m = _handle.m;
	uint n = _handle.n;
	//double j = log2(n - 1);

	println("Build Binary Tree");
	double start = tic();
	Mat S(_handle.m,_handle.n,TYPE);
	assert(S.isContinuous());
	double* p = (double*)S.data;
	for (int i = 0; i < S.size().area(); ++i){
		*p++ = i;
	}

	uint w = _w;
	S.copyTo(_handle.S);
	beamCurves(0,0,&S);
	toc(start);

	println("Create Edge Image");
	start = tic();
	getScores();
	toc(start);
	return _E;
}

void Detector::beamCurves(uint index, uint level, Mat* S){
	uint m = S->rows;
	uint n = S->cols;

	_pixelScores[index] = Mat(_handle.m, _handle.n, TYPE,ZERO);

	if ( max(m,n) <= _prm.patchSize ){
		_maxLevel = level;
		getBottomLevelSimple(*S, index);
	}
	else{
		Mat S0, S1;
		bool verticalSplit;
		if (n>=m) {
			verticalSplit = true;
			uint mid = (uint)floor(n / 2);
			subIm(*S, 0, 0, m-1, mid, S0);
			subIm(*S, 0, mid, m-1, n-1, S1);
		}
		else {
			verticalSplit = false;
			uint mid = (uint)floor(m / 2);
			subIm(*S, 0, 0, mid, n-1, S0);
			subIm(*S, mid, 0, m-1, n-1, S1);
		}
		
		uint t[] = { 2 * index + 1, 2 * index + 2 };
		Mat* SArr[] = { &S0, &S1 };

		if (_prm.parallel && level%2 == 0){
			vector<std::thread> tasks;
			for (uint i = 0; i < 2; ++i)
				tasks.push_back(std::thread(std::bind(&Detector::beamCurves, this, t[i], level+1, SArr[i])));
			
			for (uint i = 0; i < tasks.size(); ++i)
				tasks[i].join();
		}
		else{
			for (uint i = 0; i < 2; ++i)
				beamCurves(t[i], level+1, SArr[i]);
		}
		_prm.function(_pixelScores[t[0]], _pixelScores[t[1]], _pixelScores[index]);
		for (uint i = 0; i < 2; ++i){
			_pixelScores[t[i]].release();
		}
		mergeTilesSimple(S0, S1, index, level, verticalSplit);
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

void Detector::mergeTilesSimple(const Mat& S1, const Mat& S2, uint index, uint level, bool verticalSplit){

	uint index1 = 2 * index + 1;
	uint index2 = 2 * index + 2;
	unordered_map<uint, Mat>& data1 = _data[index1];
	unordered_map<uint, Mat>& data2 = _data[index2];

	vector<Mat> edgeS1; getEdgeIndices(S1, edgeS1);
	vector<Mat> edgeS2; getEdgeIndices(S2, edgeS2);
	Mat split;

	vector<std::thread> tasks;

	if (verticalSplit){
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
				if (_w == 0) con = resp/len;
				else con = resp/len/2/_w;
				double thresh = getThreshold(len);
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

void Detector::subIm(const Mat& Ssrc, uint x0, uint y0, uint x1, uint y1, Mat& Sdst){
	Sdst = Ssrc(Range(x0,x1+1), Range(y0,y1+1)).clone();
}

uint Detector::getSideLength(uint m, uint n, uint e){
	if (e % 2 == 1)
		return m;
	else
		return n;
}

void Detector::getBottomLevelSimple(Mat& S, uint index){
	//cout << S << endl;
	uint m = S.rows;
	uint n = S.cols;
	assert(S.isContinuous());
	int baseInd = (int)S.at<double>(0,0);
	int row, col;
	ind2sub(baseInd, _I.cols, _I.rows, row, col);
	//cout << _handle.S << endl;
	//cout << S << endl;
	//cout << row << "," << col << endl;
	Mat gx, gy, ss;
	subIm(_dX, row, col, row+m-1, col+n-1, gx);
	subIm(_dY, row, col, row+m-1, col+n-1, gy);
	subIm(_handle.S, row, col, row + m - 1, col + n - 1, ss);
	//cout << ss << endl;
	
	for (uint e0 = 1; e0 <= 3; ++e0){
		for (uint e1 = e0 + 1; e1 <= 4; ++e1){
			uint len0 = getSideLength(m, n, e0);
			for (uint v0 = 0; v0 < len0; ++v0){
				uint x0, y0;
				getVerticesFromPatchIndices(e0, v0, m, n, x0, y0);
				uint len1 = getSideLength(m, n, e1);
				for (uint v1 = 0; v1 < len1; ++v1){
					uint x1, y1;
					getVerticesFromPatchIndices(e1, v1, m, n, x1, y1);
					uint ind0 = (uint)S.at<double>(x0, y0);
					uint ind1 = (uint)S.at<double>(x1, y1);

					if (ind0 == ind1) continue;

					Mat P(m, n, TYPE, ZERO);
					// TODO: if expensive, keep line images
					int len = getLine(x0, y0, x1, y1, P);
					//cout << P << endl;
					Mat curIndices;
					findIndices(P, curIndices);
					//cout << curIndices << endl;
					Mat curPixels;
					curIndices.convertTo(curIndices, TYPE);
					copyIndices(S, curIndices, curPixels);
					//cout << curPixels << endl;
					int dx = x1 - x0;
					int dy = y1 - y0;

					double respX = sign(dx)*P.dot(gx);
					double respY = -sign(dy)*P.dot(gy);
					double resp;
					if (abs(dx) == abs(dy)){
						resp = 0.5*(respX + respY); 
					}
					else if (abs(dx) > abs(dy)){ 
						resp = respX;
					}
					else{ 
						resp = respY;
					}
					double con = resp/(2 * _w*len);
					bool good = abs(con) >= (_prm.removeEpsilon*_prm.sigma);
					if (!good) continue;
					double minC = con, maxC = con, score;
					if (len <= 0){
						score = numeric_limits<double>::min();
					}
					else{
						double thresh = getThreshold(len);
						score = abs(con) - thresh;
					}

					double ind01 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)ind0, (int)ind1);
					double ind10 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)ind1, (int)ind0);
					Mat bestData(_handle.TOTAL, 1, TYPE, ZERO);
					assert(bestData.isContinuous());
					double* bd = (double*)bestData.data;
					bd[_handle.C] = con;
					bd[_handle.I0S0] = 0;
					bd[_handle.L] = len;
					bd[_handle.R] = resp;
					bd[_handle.S0I1] = 0;
					bd[_handle.SC] = score;
					bd[_handle.minC] = con;
					bd[_handle.maxC] = con;

					Mat bestData2 = bestData.clone();
					assert(bestData2.isContinuous());
					double* bd2 = (double*)bestData2.data;
					bd2[_handle.R] = -bd[_handle.R];
					bd2[_handle.C] = -bd[_handle.C];
					bd2[_handle.minC] = -bd[_handle.maxC];
					bd2[_handle.maxC] = -bd[_handle.minC];

					_data[index].insert(pair<uint, Mat>((uint)ind01, bestData.clone()));
					_data[index].insert(pair<uint, Mat>((uint)ind10, bestData2.clone()));

					double value = abs(resp);
					if (value > 0){
						Mat& px = _pixelScores[index];
						assert(px.isContinuous());
						double* p = (double*)px.data;
						uint i0 = (int)ind0, i1 = (uint)ind1;
						p[i0] = std::max(p[i0], value);
						p[i1] = std::max(p[i1], value);
					}

					std::chrono::milliseconds interval(0);
					while (true){
						if (_mtx.try_lock()){
							break;
						}
						else{
							this_thread::sleep_for(interval);
						}
					}
					_pixels.insert(pair<uint, Mat>((uint)ind01, curPixels.clone()));
					_pixels.insert(pair<uint, Mat>((uint)ind10, curPixels.clone()));
					_mtx.unlock();
				}
			}
		}
	}
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

	Mat newPixelScores0 = Mat(_handle.m,_handle.n, TYPE, ZERO);
	Mat newPixelScores1 = Mat(_handle.m, _handle.n, TYPE, ZERO);

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
	uint w = 2 * _w;
	Mat wL = w*L;
	Mat div; divide(2 * log(6 * _handle.N), wL, div);
	Mat sq; sqrt(div, sq);
	T = _prm.sigma*sq;
}

int Detector::getLine(int x0, int y0, int x1, int y1, Mat& P){
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

	double corner = 2*0.5;

	while(true){
		P.at<double>(x0, y0) = 1;
		if (first){
			P.at<double>(x0, y0) = corner;
			first = false;
		}
		if (x0 == x1 && y0 == y1){
			P.at<double>(x0, y0) = corner;
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

void Detector::getVerticesFromPatchIndices(uint e, uint  v, uint m, uint n, uint& x, uint& y){
	/*
	e is the edge index, v is the vertex index
	the function converet the pair(e,v) to the coordinates (x,y)
	e = 1 is the left patch side
	e = 2 is the up side
	*/

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
		x = m-1;
		y = v;
		break;
	default:
		x = -1;
		y = -1;
	}
}

void Detector::getScores(){
	uint m = _handle.m;
	uint n = _handle.n;
	
	Mat selected(m,n,BOOL,FALSE);

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
		Mat E(m, n, BOOL, FALSE);

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
	//int maxLevel = (int)(2*log2(_handle.n)-log2(_prm.patchSize));
	if (data.count(curKey) == 0 || level == _maxLevel){
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
	int ang0 = indToAngle(_handle.m, _handle.n, ind0, s0);
	int ang1 = indToAngle(_handle.m, _handle.n, s0, ind1);
	return(angleInRange(ang0, ang1, level));
}

bool Detector::angleInRange(double ang0, double ang1, uint level){
	int diff = (int)(ang0 - ang1)+360;
	diff %= 360;
	assert(diff >= 0 && diff < 360);
	//double J = floor(log2(_handle.N)) - 2*log2(_prm.patchSize - 1)-1;
	double J = _maxLevel-1;
	double curMaxTurn = _prm.maxTurn*(2.0 - (double)level / J);
	assert(curMaxTurn >= _prm.maxTurn && curMaxTurn <= _prm.maxTurn * 2);
	return (diff <= curMaxTurn) || ((360 - diff) <= curMaxTurn);
}