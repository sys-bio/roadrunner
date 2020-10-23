#ifndef rrTestUtilsH
#define rrTestUtilsH
#include <string>
#include "rr-libstruct/lsMatrix.h"

std::string getModelDir(std::string base, std::string suite, std::string name);
bool testValidateSBML(std::string sbml);
ls::DoubleMatrix ParseMatrixFromText(const std::string& textMatrix);
void expectArrayClose(const ls::DoubleMatrix& jRef, const ls::DoubleMatrix& fullJacobian, double epsilon);

std::vector< std::complex<double> > getComplexVectorFromString(const std::string& text);
ls::DoubleMatrix getDoubleMatrixFromString(const std::string& textMatrix);
ls::ComplexMatrix getComplexMatrixFromString(const std::string& textMatrix);
// simply returns if file contents are identical
bool filesAreEqual(const std::string& expected_fname, const std::string& actual_fname);

#endif

