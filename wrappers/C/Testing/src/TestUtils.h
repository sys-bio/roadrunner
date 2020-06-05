#ifndef TestUtilsH
#define TestUtilsH
#include <string>
#include "rr-libstruct/lsMatrix.h"


std::vector< std::complex<double> > getComplexVectorFromString(const std::string& text);
ls::DoubleMatrix getDoubleMatrixFromString(const std::string& textMatrix);
ls::ComplexMatrix getComplexMatrixFromString(const std::string& textMatrix);
// simply returns if file contents are identical
bool filesAreEqual(const std::string& expected_fname, const std::string& actual_fname);


#endif
