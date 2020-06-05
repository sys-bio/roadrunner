#include <vector>
#include <fstream>
#include "rrUtils.h"
#include "TestUtils.h"

using namespace std;
using namespace rr;
using namespace ls;

vector< complex<double> > getComplexVectorFromString(const string& text)
{
	vector< complex<double> > vec;

    //Parse the vector
    vector<string> rows = splitString(text, "\n");
    for(int row = 0; row < rows.size(); row++)
    {
		vec.push_back(toComplex(rows[row]));
    }
	return vec;
}

ls::DoubleMatrix getDoubleMatrixFromString(const string& textMatrix)
{
	DoubleMatrix mat;

    //Parse the matrix
    vector<string> rows = splitString(textMatrix, "\n");
    for(int row = 0; row < rows.size(); row++)
    {
        vector<string> values = splitString(rows[row], " \t");
        for(int col = 0; col < values.size(); col++)
        {
        	if(!mat.size())
            {
                mat.resize(rows.size(), values.size());
            }

            mat(row, col) = toDouble(values[col]);
        }
    }
	return mat;
}

ls::ComplexMatrix getComplexMatrixFromString(const string& textMatrix)
{
	ComplexMatrix mat;

    //Parse the matrix
    vector<string> rows = splitString(textMatrix, "\n");
    for(int row = 0; row < rows.size(); row++)
    {
        vector<string> values = splitString(rows[row], " \t");
        for(int col = 0; col < values.size(); col++)
        {
        	if(!mat.size())
            {
                mat.resize(rows.size(), values.size());
            }

            std::complex<double> val = toComplex(values[col]);
            mat(row, col) = val;
        }
    }
	return mat;
}

bool filesAreEqual(const string& expected, const string& actual) {
    ifstream efile(expected);
    ifstream afile(actual);

    string line1;
    string line2;
    while (efile && afile) {
        getline(efile, line1);
        getline(afile, line2);
        if (line1 != line2) {
            return false;
        }
    }

    // check for remaining lines
    return !(efile || afile);
}

