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
				cerr << "Usage: ./driver dataFile grpFile m numFeatures numClasses\n";
				return(1);
		}

		Validation v(argv[1], argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
		v.validate();
		return 0;
}
