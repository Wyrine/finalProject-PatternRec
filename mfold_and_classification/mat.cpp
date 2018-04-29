#include <iostream>
#include <ctime>
#include <functional>
#include "mat.hpp"

using namespace std;

Mat::Mat(const char* tr, const char* te, const uint& _features, 
				const uint& _classes, double (*_compFunc)(const string &))
{
		classes = _classes;
		features = _features;
		compFunc = _compFunc;
		readFile(tr, X);
		readFile(te, Xte);
		nX = subMatrix(X, 0, 0, X.getRow() -1, features - 1);
		nXte = subMatrix(Xte, 0, 0, Xte.getRow() -1, features -1);
		normalize(nX, nXte, features, 1);

		setParams(mu, sig, nX, nXte);
		PCA(); FLD();
}
		void
Mat::generateEvals(const Matrix & results, const void* arg, FILE* out, const uint flag)
{
		double accuracy, precision, sens, spec, tp=0, tn=0, fp=0, fn=0;

		for(int i = 0; i < results.getRow(); i++)
		{
				if(results(i, 0) == results(i, 1))
				{
						if( results(i, 0) ) tp++;
						else tn++;
				}
				else
				{
						if( results(i, 0) ) fp++;
						else fn++;
				}
		}
		tp /= results.getRow();
		tn /= results.getRow();
		fp /= results.getRow();
		fn /= results.getRow();
		accuracy = (tp + tn) / (tp + tn + fp + fn);
		sens = tp / (tp + fn);
		spec = tn / (tn + fp);
		precision = tp / (tp + fp);
		switch(flag)
		{
				case 0: //prior0,prior1,
						fprintf(out, "%lf,%lf,", ((double*)arg)[0], ((double*)arg)[1]);
						break;
				case 1://k,dist,
						fprintf(out, "%d,%d,", *((int*)arg), ((int*)arg)[1] );
						break;
				case 2: //validationStep,k,dist,
						fprintf(out, "%d,%d,%d,", *((int*)arg), ((int*)arg)[1], ((int*)arg)[2]);
						break;
				case 3://L1,L2,
						fprintf(out, "%s,%s,", classNames[*((int*)arg)].c_str(), 
										classNames[((int*)arg)[1]].c_str());
				default:
						break;
		}
		fprintf(out, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", tp, tn, fp, fn,
						accuracy, sens, spec, precision);
}
Matrix
Mat::getProbMatrix(const Matrix & result) const
{
		Matrix rv(classes, classes);
		int actual[classes];
		memset(actual, 0, sizeof(int) * classes);

		for(int i = 0; i < result.getRow(); i++)
		{
				++actual[(int)result(i, 1)];
				++rv(result(i, 1), result(i, 0));
		}
		for(int i = 0; i < classes; i++)
				for(int j = 0; j < classes; j++)
						rv(i, j) /= actual[i];
		return rv;
}
		void
Mat::fuseNB_All()
{
		Matrix fusedTable, pred1, pred2, fusedPred(predictions[0].getRow(), 1);
		FILE* out; out = openFile("fusionNB_Exhaustive.csv");
		fprintf(out, "L1_Name,L2_Name,");
		writeHeader(classes, out, -1);
		addLabels(fusedPred, predictions[0]);
		for(uint i = 0; i < 12; i++)
		{
				for(uint j = 0; j < 12; j++)
				{
						if( i == j ) continue;
						int arg[2]; arg[0] = i; arg[1] = j;
						pred1 = predictions[i]; pred2 = predictions[j];
						fusedTable = fuseProbMatrix(getProbMatrix(pred1), getProbMatrix(pred2));
							for(uint k = 0; k < pred1.getRow(); k++)
									fusedPred(k, 0) = fusedTable((int)pred1(k, 0), (int)pred2(k, 0));
						generateEvals(fusedPred, arg, out, 3);
				}
		}
		fclose(out);
}
Matrix
Mat::fuseProbMatrix(const Matrix & L1, const Matrix & L2) const
{
		Matrix rv(classes, classes);
		Matrix curMul(classes, 1);
		int maxInd;

		for( int i = 0; i < classes; i++)//classifier L1 chose i (col)
				for( int j = 0; j < classes; j++) //classifier L2 chose j (col)
				{
						for(int k = maxInd = 0; k < classes; k++) 
						{		//element-wise multiplication and take max
								//class in the resulting column
								curMul(k, 0) = L1(k, i) * L2(k, j);
								if(k > 0 && (curMul(maxInd, 0) < curMul(k, 0)))
										maxInd = k;
						}
						rv(i, j) = maxInd;
				}
		return rv;
}
		void
Mat::readFile(const char* fName, Matrix &_X)
{
		vector<Sample> d;
		double samp[features+1];
		string tmp;
		ifstream input(fName);

		if(! input.is_open() )
		{
				fprintf(stderr, "Unable to open %s, exiting.\n", fName);
				exit(1);
		}
		getline(input, tmp);
		while( getSamp(input, samp) )
		{
				Sample s(features, samp);
				d.push_back(s);
		}
		input.close();
		buildMatrix(d, _X);
}
		bool
Mat::getSamp(ifstream &input, double samp[])
{
		int i;
		string tmp;

		for(i = 0; i < features && (input >> samp[i]); i++);
		if( i < features) return false;

		input >> tmp;
		samp[i] = compFunc(tmp);

		return true;
}
		void
Mat::buildMatrix(vector<Sample> &c, Matrix & _X)
{
		int sz = c.size();
		_X.createMatrix(sz, features+1);
		for(int i = 0; i < sz; i++)
				for(int j = 0; j < features + 1; j++)
						_X(i, j) = c[i][j];
}
		void
Mat::PCA(float maxErr)
{
		pX = nX;
		pXte = nXte;
		int numDrop = pca(pX, pXte, features, maxErr, 1);
		pX = subMatrix(pX, 0, 0, pX.getRow() - 1, numDrop - 1);
		pXte = subMatrix(pXte, 0, 0, pXte.getRow() - 1, numDrop - 1);
		setParams(pMu, pSig, pX, pXte);
}
		void
Mat::addLabels(Matrix &dest, const Matrix & src)
{
		Matrix tmp(dest.getRow(), dest.getCol() + 1);
		uint lab_ind = src.getCol() - 1;
		uint i, j;
		for(i = 0; i < dest.getRow(); i++)
		{
				for(j = 0; j < dest.getCol(); j++)
						tmp(i, j) = dest(i, j);
				tmp(i, j) = src(i, lab_ind);
		}
		dest = tmp;
}
		void
Mat::FLD()
{
		Matrix mVec;
		mVec = mu[0];
		fW = (sig[0].getRow() - 1) * sig[0];
		for(uint i = 1; i < classes; i++)
		{
				mVec = mVec - mu[i];
				fW = fW + (sig[i].getRow() - 1) * sig[i];
		}
		Matrix sW = fW;
		fW = inverse(fW) ->* mVec;
		mVec = mVec ->* transpose(mVec);
		Matrix lossFunc = ( transpose(fW) ->* mVec ->* fW)
				/ ( transpose(fW) ->* sW ->* fW);

		//projecting
		fX = subMatrix(nX, 0, 0, nX.getRow() - 1, features - 1) ->* fW;
		fXte = subMatrix(nXte, 0, 0, nXte.getRow() - 1, features - 1) ->* fW;
		//getting means
		setParams(fMu, fSig, fX, fXte);
}

		void
Mat::setParams(vector<Matrix> & Mean, vector<Matrix> & Cov, Matrix & _X, Matrix & _Xte)
{
		Matrix m;

		addLabels(_Xte, Xte);
		addLabels(_X, X);
		for(int i = 0; i < classes; i++)
		{
				m = getType(_X, i);
				Mean.push_back( mean(m, m.getCol() ) );
				Cov.push_back( cov(m, m.getCol() ) );
		}
}
		FILE*
Mat::openFile(const char* fName)
{
		string tmp = string(STORAGE_PATH) + string(fName);
		FILE* out = fopen(tmp.c_str(), "w");

		if(out == NULL){ perror(tmp.c_str()); exit(1); }
		return out;
}
		void
Mat::writeHeader(const uint classes, FILE* out, const uint flag)
{
		switch(flag)
		{
				case 0:
						for(int i = 0; i < classes; i++)
								fprintf(out, "PriorClass%d,", i);
						break;
				case 1:
						fprintf(out, "k,dist,");
						break;
				default:
						break;
		}
		fprintf(out,"TP,TN,FP,FN,Accuracy,Sensitivity(Recall),Specificity,Precision\n");
}
		void
Mat::runCase1(const double prior[])
{
		cout << "Case 1 Scores:\n";
		writeHeader(classes);
		vector<Matrix> tmpSig;
		Matrix identity = Identity(features);
		for(int i = 0; i < classes; i++)
				tmpSig.push_back( sqrt(sig[1](0, 0)) * identity );
		Matrix rv(nXte.getRow(), 1);
		MPP(nXte, prior, tmpSig, mu, rv);
		cout << "Normalized X: \n";
		generateEvals(rv, prior);
		predictions.push_back(rv);

		tmpSig.clear();
		rv.createMatrix(pXte.getRow(), 1);
		identity = Identity(pXte.getCol() -1);

		for(int i = 0; i < classes; i++)
				tmpSig.push_back( sqrt(pSig[1](0, 0) ) * identity );
		MPP(pXte, prior, tmpSig, pMu, rv);
		cout << "PCA: \n";
		generateEvals(rv, prior);
		predictions.push_back(rv);


		tmpSig.clear();
		rv.createMatrix(fXte.getRow(), 1);
		identity = Identity(1);
		for(int i = 0; i < classes; i++)
				tmpSig.push_back( sqrt( mtod(fSig[1]) ) * identity );
		MPP(fXte, prior, tmpSig, fMu, rv);
		cout << "FLD: \n";
		generateEvals(rv, prior);
		predictions.push_back(rv);

		cout << "\n\n\n\n\n";
}
		void
Mat::varyNorm1()
{
		FILE* out = openFile("case1_normalized.csv");
		double prior[2];
		vector<Matrix> tmpSig;
		Matrix identity = Identity(features);

		//assumptions with this classifier
		for(int i = 0; i < classes; i++)
				tmpSig.push_back( sqrt(sig[1](0, 0)) * identity );

		writeHeader(classes, out);
		for(prior[0] = STEP_SIZE; prior[0] <= 1.0 - STEP_SIZE; prior[0] += STEP_SIZE)
		{
				prior[1] = 1 - prior[0];
				Matrix rv(nXte.getRow(), 1);
				MPP(nXte, prior, tmpSig, mu, rv);
				generateEvals(rv, prior, out);
		}
		fclose(out);
}
		void
Mat::varyPCA1()
{
		FILE* out = openFile("case1_PCA.csv");
		double prior[2];
		vector<Matrix> tmpSig;
		Matrix identity = Identity(pXte.getCol() -1);
		for(int i = 0; i < classes; i++)
				tmpSig.push_back( sqrt(pSig[1](0, 0) ) * identity );

		writeHeader(classes, out);
		for(prior[0] = STEP_SIZE; prior[0] <= 1.0 - STEP_SIZE; prior[0] += STEP_SIZE)
		{
				prior[1] = 1 - prior[0];
				Matrix rv(pXte.getRow(), 1);
				MPP(pXte, prior, tmpSig, pMu, rv);
				generateEvals(rv, prior, out);
		}
		fclose(out);
}
		void
Mat::varyFLD1()
{
		FILE* out = openFile("case1_FLD.csv");
		double prior[2];
		vector<Matrix> tmpSig;
		Matrix identity = Identity(1);
		for(int i = 0; i < classes; i++)
				tmpSig.push_back( sqrt( mtod(fSig[0]) ) * identity );

		writeHeader(classes, out);
		for(prior[0] = STEP_SIZE; prior[0] <= 1.0 - STEP_SIZE; prior[0] += STEP_SIZE)
		{
				prior[1] = 1 - prior[0];
				Matrix rv(fXte.getRow(), 1);
				MPP(fXte, prior, tmpSig, fMu, rv);
				generateEvals(rv, prior, out);
		}
		fclose(out);
}
		void
Mat::runCase2(const double prior[])
{
		cout << "Case 2 Scores:\n";
		writeHeader(classes);
		vector<Matrix> tmpSig;
		for(int i = 0; i < classes; i++)
				tmpSig.push_back(sig[0]);
		Matrix rv(nXte.getRow(), 1);
		MPP(nXte, prior, tmpSig, mu, rv);
		cout << "Normalized X: \n";
		generateEvals(rv, prior); 
		predictions.push_back(rv);

		tmpSig.clear();
		rv.createMatrix(pXte.getRow(), 1);

		for(int i = 0; i < classes; i++)
				tmpSig.push_back(pSig[0]);
		MPP(pXte, prior, tmpSig, pMu, rv);
		cout << "PCA: \n";
		generateEvals(rv, prior);
		predictions.push_back(rv);

		tmpSig.clear();
		rv.createMatrix(fXte.getRow(), 1);
		for(int i = 0; i < classes; i++)
				tmpSig.push_back( fSig[0] );
		MPP(fXte, prior, tmpSig, fMu, rv);
		cout << "FLD: \n";
		generateEvals(rv, prior);
		predictions.push_back(rv);
		cout << "\n\n\n\n\n";
}
		void
Mat::varyNorm2()
{
		FILE* out = openFile("case2_normalized.csv");
		double prior[2];
		vector<Matrix> tmpSig;

		for(int i = 0; i < classes; i++)
				tmpSig.push_back(sig[0]);

		writeHeader(classes, out);
		for(prior[0] = STEP_SIZE; prior[0] <= 1.0 - STEP_SIZE; prior[0] += STEP_SIZE)
		{
				prior[1] = 1 - prior[0];
				Matrix rv(nXte.getRow(), 1);
				MPP(nXte, prior, tmpSig, mu, rv);
				generateEvals(rv, prior, out);
		}
		fclose(out);
}
		void
Mat::varyPCA2()
{
		FILE* out = openFile("case2_PCA.csv");
		double prior[2];
		vector<Matrix> tmpSig;

		for(int i = 0; i < classes; i++)
				tmpSig.push_back(pSig[0]);

		writeHeader(classes, out);
		for(prior[0] = STEP_SIZE; prior[0] <= 1.0 - STEP_SIZE; prior[0] += STEP_SIZE)
		{
				prior[1] = 1 - prior[0];
				Matrix rv(pXte.getRow(), 1);
				MPP(pXte, prior, tmpSig, pMu, rv);
				generateEvals(rv, prior, out);
		}
		fclose(out);
}

		void
Mat::varyFLD2()
{
		FILE* out = openFile("case2_FLD.csv");
		double prior[2];
		vector<Matrix> tmpSig;

		for(int i = 0; i < classes; i++)
				tmpSig.push_back(fSig[0]);

		writeHeader(classes, out);
		for(prior[0] = STEP_SIZE; prior[0] <= 1.0 - STEP_SIZE; prior[0] += STEP_SIZE)
		{
				prior[1] = 1 - prior[0];
				Matrix rv(fXte.getRow(), 1);
				MPP(fXte, prior, tmpSig, fMu, rv);
				generateEvals(rv, prior, out);
		}
		fclose(out);
}

		void
Mat::runCase3(const double prior[])
{
		cout << "Case 3 Scores:\n";
		writeHeader(classes);
		Matrix rv(nXte.getRow(), 1);
		MPP(nXte, prior, sig, mu, rv);
		cout << "Normalized X: \n";
		generateEvals(rv, prior); 
		predictions.push_back(rv);

		rv.createMatrix(pXte.getRow(), 1);
		MPP(pXte, prior, pSig, pMu, rv);
		cout << "PCA: \n";
		generateEvals(rv, prior);
		predictions.push_back(rv);

		rv.createMatrix(fXte.getRow(), 1);
		MPP(fXte, prior, fSig, fMu, rv);
		cout << "FLD: \n";
		generateEvals(rv, prior);
		predictions.push_back(rv);
		cout << "\n\n\n\n\n";
}

		void
Mat::varyNorm3()
{
		FILE* out = openFile("case3_normalized.csv");
		double prior[2];
		vector<Matrix> tmpSig;
		for(int i = 0; i < classes; i++)
				tmpSig.push_back(sig[i]);

		writeHeader(classes, out);
		for(prior[0] = STEP_SIZE; prior[0] <= 1.0 - STEP_SIZE; prior[0] += STEP_SIZE)
		{
				prior[1] = 1 - prior[0];
				Matrix rv(nXte.getRow(), 1);
				MPP(nXte, prior, tmpSig, mu, rv);
				generateEvals(rv, prior, out);
		}
		fclose(out);
}
		void
Mat::varyPCA3()
{
		FILE* out = openFile("case3_PCA.csv");
		double prior[2];
		vector<Matrix> tmpSig;
		for(int i = 0; i < classes; i++)
				tmpSig.push_back(pSig[i]);

		writeHeader(classes, out);
		for(prior[0] = STEP_SIZE; prior[0] <= 1.0 - STEP_SIZE; prior[0] += STEP_SIZE)
		{
				prior[1] = 1 - prior[0];
				Matrix rv(pXte.getRow(), 1);
				MPP(pXte, prior, tmpSig, pMu, rv);
				generateEvals(rv, prior, out);
		}
		fclose(out);
}
		void
Mat::varyFLD3()
{
		FILE* out = openFile("case3_FLD.csv");
		double prior[2];
		vector<Matrix> tmpSig;
		for(int i = 0; i < classes; i++)
				tmpSig.push_back(fSig[i]);

		writeHeader(classes, out);
		for(prior[0] = STEP_SIZE; prior[0] <= 1.0 - STEP_SIZE; prior[0] += STEP_SIZE)
		{
				prior[1] = 1 - prior[0];
				Matrix rv(fXte.getRow(), 1);
				MPP(fXte, prior, tmpSig, fMu, rv);
				generateEvals(rv, prior, out);
		}
		fclose(out);
}
		Matrix &
Mat::MPP(Matrix &testData, const double prior[], const vector<Matrix> & _sig,
				const vector<Matrix>& _mean, Matrix & result)
{
		int samples = testData.getRow(), choice;
		Matrix xVec(testData.getCol() - 1, 1), diff;
		double post, tmp;

		for(int i = 0; i < samples; i++)
		{
				post = tmp = 0;
				choice = -1;
				for(int j = 0; j < testData.getCol() - 1; j++)
						xVec(j, 0) = testData(i, j);
				for(int j = 0; j < classes; j++)
				{
						diff = xVec - _mean[j];
						tmp = PI_CONST * 1 / sqrt(det(_sig[j])) * prior[j];
						tmp *= exp(- mtod(transpose(diff) ->* inverse(_sig[j]) ->* diff) / 2.0);
						if (tmp >= post) { post = tmp; choice = j; }
				}
				result(i, 0) = choice;
		}

		addLabels(result, Xte);
		return result;
}
		void
Mat::varyAllCases()
{
		thread t(&Mat::varyCase1, this);
		thread t1(&Mat::varyCase2, this);
		thread t2(&Mat::varyCase3, this);
		thread t3(&Mat::varyAllkNN, this, 2);
		t.join();
		t1.join();
		t2.join();
		t3.join();
}
		void
Mat::varyCase1()
{
		thread t(&Mat::varyNorm1, this);
		thread t1(&Mat::varyPCA1, this);
		thread t2(&Mat::varyFLD1, this);
		t.join();
		t1.join();
		t2.join();
}
		void
Mat::varyCase2()
{
		thread t(&Mat::varyNorm2, this);
		thread t1(&Mat::varyPCA2, this);
		thread t2(&Mat::varyFLD2, this);
		t.join();
		t1.join();
		t2.join();
}
		void
Mat::varyCase3()
{
		thread t(&Mat::varyNorm3, this);
		thread t1(&Mat::varyPCA3, this);
		thread t2(&Mat::varyFLD3, this);
		t.join();
		t1.join();
		t2.join();
}
		void
Mat::varyAllkNN(const uint dist)
{
		thread t1(&Mat::varykNN, this, 0, dist);
		thread t2(&Mat::varykNN, this, 1, dist);
		thread t3(&Mat::varykNN, this, 2, dist);
		t1.join();
		t2.join();
		t3.join();
}
		void
Mat::varykNN(const uint transType, const uint dist)
{
		FILE* out;
		switch(transType)
		{
				case 0:
						out = openFile("kNN_normalized.csv");
						break;
				case 1:
						out = openFile("kNN_PCA.csv");
						break;
				case 2:
						out = openFile("kNN_FLD.csv");
						break;
		}
		writeHeader(classes, out, 1);
		for(int k = 1; k <= MAX_K_NEIGHBORS; k+= 2)
				for(uint d = 1; d < MAX_DIST; d++)
						runkNN(transType, k, d, out);
		fclose(out);
}
		void
Mat::runkNN(const uint transType, const uint k, const uint dist, FILE* out)
{
		Matrix temp;
		clock_t t;
		switch(transType)
		{
				case 0:
						t = clock();
						temp = kNN(nXte, nX, k, dist);
						t = clock() - t;
						addLabels(temp, nXte);
						break;
				case 1:
						t = clock();
						temp = kNN(pXte, pX, k, dist);
						t = clock() - t;
						addLabels(temp, pXte);
						break;
				case 2:
						t = clock();
						temp = kNN(fXte, fX, k, dist);
						t = clock() - t;
						addLabels(temp, fXte);
						break;
		}
		if(out == stdout || out == stderr) 
		{
				printf("kNN score for fixed k and distance. Time taken in seconds: %lf:\n", 
						((double)t)/CLOCKS_PER_SEC);
				writeHeader(classes, out, 1);
		}
		int arg[2]; arg[0] = k; arg[1] = dist;
		generateEvals(temp, arg, out, 1);
		if(out == stdout || out == stderr)
		{
				predictions.push_back(temp);
				printf("\n\n\n\n\n\n");
		}
}
		Matrix
Mat::cropMatrix(const Matrix& m, const uint sR, const uint eR,
				const uint sC, const uint eC)
{
		Matrix rv(eR-sR, eC-sC);
		for(int i = sR; i < eR; i++)
				for(int j = sC; j < eC; j++)
						rv(i-sR, j-sC) = m(i, j);
		return rv;
}
void
Mat::kNN_Classify(const Matrix &_te, const Matrix &_tr, const uint k, const uint dist,
				const int i, Matrix &rv) const
{
		Matrix neighbors(k, 2), curTe, curTr;
		curTe = cropMatrix(_te, i, i+1, 0, _te.getCol()-1); 
		//initialize the k neighbors to the first k distances from the training set
		for(int j = 0; j < k; j++)
		{
				neighbors(j, 0) = Minkowski(curTe, 
								cropMatrix(_tr, j, j+1, 0, _tr.getCol()-1), dist);
				neighbors(j, 1) = _tr(j, _tr.getCol()-1);
		}
		//and loop through the rest of the training set
		for(int j = k; j < _tr.getRow(); j++)
		{
				curTr = cropMatrix(_tr, j, j+1, 0, _tr.getCol()-1); 
				double curDist = Minkowski(curTe, curTr, dist);
				int max_dist_index = 0;
				for(int t = 1; t < k; t++)
						if( neighbors(t, 0) > neighbors(max_dist_index, 0) )
								max_dist_index = t;
				//replace the maximum distance if it is longer than
				//the current distance
				if( neighbors(max_dist_index, 0) > curDist )
				{
						neighbors(max_dist_index, 0) = curDist;
						neighbors(max_dist_index, 1) = _tr(j, _tr.getCol()-1);
				}
		}
		//classifying the class with most neighbors
		rv(i, 0) = neighborVoting(neighbors);
}
Matrix
Mat::kNN(const Matrix &_te, const Matrix &_tr, const uint k, const uint dist) const
{

		Matrix rv(_te.getRow(), 1), neighbors(k, 2), curTe, curTr;
		for(int i = 0; i < _te.getRow(); i++)	
				kNN_Classify(_te, _tr, k, dist, i, rv);
		return rv;
}
Matrix
Mat::Parallel_kNN(const Matrix &_te, const Matrix &_tr, const uint k, 
				const uint dist) const
{
		thread t[_te.getRow()];
		Matrix rv(_te.getRow(), 1), neighbors(k, 2), curTe, curTr;
		for(int i = 0; i < _te.getRow(); i++)
				t[i] = thread(&Mat::kNN_Classify, this, cref(_te), cref(_tr), k, dist, i, ref(rv));
		for(int i = 0; i < _te.getRow(); i++)
				t[i].join();
		return rv;
}
short
Mat::neighborVoting(const Matrix& neighbors) const
{
		int neighborCounts[classes];
		memset(neighborCounts, 0, sizeof(int) * classes);
		for(int i = 0; i < neighbors.getRow(); i++)
				neighborCounts[ (int)neighbors(i, 1) ]+= 1;
		int rv = 0;
		for(int i = 1; i < classes; i++)
				if( neighborCounts[i] > neighborCounts[rv] )
						rv = i;
		return rv;
}
		double
Mat::Minkowski(const Matrix & teSample, const Matrix & trSample, const uint dist)
{
		double rv = 0.0;
		for(int i=0; i< teSample.getCol(); i++)
				rv += pow(abs(teSample(0, i) - trSample(0, i)), dist);
		return pow(rv, 1.0/dist);
}
		Matrix
Identity(const uint n)
{
		Matrix ident(n,n);
		for(uint i = 0; i < n; i++)
				ident(i,i) = 1;
		return ident;
}
		ostream& 
operator<<(ostream &os, const Sample &s)
{
		for(int i = 0; i < s.sample.size(); i++)
				os << s.sample[i] << " ";
		os << endl;
		return os;
}
