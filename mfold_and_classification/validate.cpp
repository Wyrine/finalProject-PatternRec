#include "validate.hpp"
#include <cctype>

using namespace std;

Validation::Validation(const char* data, const char* grp, const uint _m, 
				const uint _f, const uint _c)
{
		features = _f;
		//read the two files and store them in parallel
		thread t1(&Validation::readAndBuildMatrix, this, data);
		thread t2(&Validation::readGroupingFile, this, grp);
		m = _m;
		classes = _c;
		t1.join(); t2.join();
}
		void
Validation::validate(const uint m)
{
		vector<Sample> test, train;
		for(int i=0; i< groups.size(); i++)
		{
				//i is the group that is the test set
				for(int j = 0; j< groups.size(); j++)
						for(int k = 0; k < groups[j].size(); k++)
						{
								if(j == i) //current test set
										test.push_back(dataSet[ groups[j][k] ]);
								else //current training set
										train.push_back(dataSet[ groups[j][k] ]);
						}
				buildMatrix(test, Xte); test.clear();
				buildMatrix(train, X); train.clear();
				nXte = cropMatrix(Xte, 0, Xte.getRow(), 0, Xte.getCol()-1);
				nX = cropMatrix(X, 0, X.getRow(), 0, X.getCol()-1);
				normalize(nX, nXte, features, 1);
				addLabels(nX, X);
				addLabels(nXte, Xte);
				varykNN(i+1, 2);
		}
}
		void
Validation::varykNN(const uint valStep, const uint dist)
{
		FILE* out;
		string fName = string("kNN_validation") + to_string(valStep) + string(".csv");
		out = openFile(fName.c_str());
		writeHeader(classes, out, 1);

		for(int k = 1; k < MAX_K_NEIGHBORS; k+=2)
				for(int d = 1; d < MAX_DIST; d++)
						runkNN(0, k, d, out);
		fclose(out);
}
		void
Validation::readGroupingFile(const char* grp)
{
		char junk;
		vector<short> curLine;
		short curVal;
		ifstream input(grp);
		if(! input.is_open() )
		{
				cerr << "Failed to open " << grp << ". Exiting." << endl;
				exit(1);
		}

		do{
				do{
						input >> curVal;
						//decrement to use it as index
						curLine.push_back(curVal - 1);
						//check if next character is a space, if it is then data is good
						//if it's any other character then it's not
				}while( isspace(input.peek()) );
				groups.push_back(curLine);
				curLine.clear();
				//if the character is a delimeter then start with the new group
				//if it's eof then the loop is terminated
		}while( input >> junk );
		input.close();
}
		void
Validation::readAndBuildMatrix(const char* data)
{
		ifstream input(data);
		string line;
		double tmp[features + 1];
		if(! input.is_open() )
		{
				cerr << "Failed to open " << data << ". Exiting." << endl;
				exit(1);
		}
		getline(input, line);
		while( getSamp(input, tmp) )
		{
				Sample s(features, tmp);
				dataSet.push_back(s);
		}
		input.close();	
}
		double
Validation::mComp(const int _cur)
{
		//scale the classes to be from 0 through (class - 1)
		return (_cur > 3) ? _cur - 2 : _cur - 1;
}
		bool
Validation::getSamp(ifstream & input, double samp[])
{
		int i; string tmp;

		for(i = 0; i < features && (input >> samp[i]); i++);
		if(i < features) return false;

		if(compFunc == nullptr)
		{
				input >> samp[i];
				samp[i] = mComp(samp[i]);
		}
		else
		{
				input >> tmp;
				samp[i] = compFunc(tmp);
		}
		return true;
}
