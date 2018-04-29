#pragma once

#define _USE_MATH_DEFINES

#include <iostream>
#include <cstdlib>
#include <string>
#include <cmath>
#include <fstream>
#include <thread>
#include <vector>
#include "Matrix.h"
#include "Pr.h"


#define STEP_SIZE 0.1
#define MAX_K_NEIGHBORS sqrt(X.getRow())
#define MAX_DIST 101
#define PI_CONST 1.0 / pow(2*M_PI, (testData.getCol() - 1 )/2)
#define STORAGE_PATH "./performance/"
typedef unsigned int uint;
static const string classNames[] = {"case1_norm", "case1_PCA", "case1_FLD",
		"case2_norm", "case2_PCA", "case2_FLD",
		"case3_norm", "case3_PCA", "case3_FLD",
		"kNN_norm", "kNN_PCA", "kNN_FLD" };

int normalize(Matrix &, Matrix &, const int, const int);
int pca(Matrix &, Matrix &, const int, const float, const int);
Matrix Identity(const uint);

struct Sample
{
		vector<double> sample;
		Sample(const uint& _features, const double samp[])
		{
				for(int i = 0; i < _features + 1; i++)
						sample.push_back(samp[i]);
		}
		double & operator[](const int& i) { return sample[i]; }
		friend ostream & operator<<(ostream &, const Sample &);
};

class Mat
{
		protected:
				uint classes, features;
				Matrix X, nX, Xte, nXte;
				vector<Matrix> mu, sig;

				//index 0-2: case 1_norm, case 1_PCA, case 1_FLD
				//index 3-5: case 2_norm, case 2_PCA, case 2_FLD
				//index 6-8: case 3_norm, case 3_PCA, case 3_FLD
				//index 9-11: kNN_norm, kNN_PCA, kNN_FLD
				vector<Matrix> predictions;

				Mat(){ classes=features=0; compFunc = nullptr; }
				double (*compFunc)(const string &);
				virtual bool getSamp(ifstream &, double []);
				virtual void buildMatrix(vector<Sample> &, Matrix&);
				virtual void readFile(const char*, Matrix &);
				void addLabels(Matrix &, const Matrix &);
				void setParams(vector<Matrix> &, vector<Matrix> &, Matrix &, Matrix &);
				Matrix & MPP(Matrix&, const double [], const vector<Matrix> &,
								const vector<Matrix> &, Matrix&);
				Matrix kNN(const Matrix &, const Matrix &, const uint, const uint) const;
				Matrix Parallel_kNN(const Matrix &, const Matrix &, const uint, const uint) const;
				void kNN_Classify(const Matrix &, const Matrix &, const uint, const uint,
								const int, Matrix &) const;
				short neighborVoting(const Matrix &) const;
				static double Minkowski(const Matrix &, const Matrix &, const uint dist = 2);
				static void generateEvals(const Matrix &, const void*, FILE* = stdout, const uint = 0);
				static FILE* openFile(const char*);
				static Matrix cropMatrix(const Matrix &, const uint, const uint,
								const uint, const uint);
				static void writeHeader(const uint, FILE* = stdout, const uint = 0);
				Matrix getProbMatrix(const Matrix &) const;
				Matrix fuseProbMatrix(const Matrix &, const Matrix &) const;
				virtual void varyNorm1();
				virtual void varyPCA1();
				virtual void varyFLD1();
				virtual void varyNorm2();
				virtual void varyPCA2();
				virtual void varyFLD2();
				virtual void varyNorm3();
				virtual void varyPCA3();
				virtual void varyFLD3();
		public:
				virtual double prior0() { return ((double) getType(X, 0).getRow() ) / X.getRow(); }
				virtual double prior1() { return ((double) getType(X, 1).getRow() ) / X.getRow(); }
				Mat(const char*, const char*, const uint&, const uint&, 
								double (*_compFunc)(const string &));
				virtual void fuseNB_All();
				virtual void varyCase1();
				virtual void varyCase2();
				virtual void varyCase3();
				virtual void varykNN(const uint transType, const uint dist = 2);
				virtual void varyAllkNN(const uint dist = 2);
				virtual void varyAllCases();
				virtual void runCase1(const double []);
				virtual void runCase2(const double []);
				virtual void runCase3(const double []);
				virtual void runkNN(const uint = 0, const uint = 3, const uint = 2, FILE* = stdout);
				virtual void PCA(float maxErr = 0.1);
				virtual void FLD();
};
