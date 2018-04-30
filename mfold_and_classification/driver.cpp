#include <iostream>
#include "validate.hpp"

using namespace std;

		double
compareFunc(const string &curStr)
{
		return (curStr == "No") ? 0 : 1;
}

		int
main(int argc, char** argv)
{
		if(argc < 5)
		{
				cerr << "Usage: ./driver trainFile testFile numFeatures numClasses\n";
				return(1);
		}
		Mat mat(argv[1], argv[2], atoi(argv[3]), atoi(argv[4]));
		double prior[atoi(argv[4])]; 
		prior[0] = mat.prior0();
		prior[1] = mat.prior1();
		cout << "Prior from data set: \n\n";
		mat.runCase1(prior);
		mat.runCase2(prior);
		mat.runCase3(prior);
		mat.runkNN(0, 3, 2);
		mat.runkNN(1, 3, 2);
		mat.runkNN(2, 3, 2);
		mat.fuseNB_All();

		cout << "Varying Prior probailities, k-values, and writing to files\n"; 
		mat.varyAllCases();
		cout << "Done\n"; 

//		Validation v("./data/fglass.dat", "./data/fglass.grp");
//		v.validate();
		return 0;
}
