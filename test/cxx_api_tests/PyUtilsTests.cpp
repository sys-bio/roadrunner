//
// Created by Ciaran on 21/07/2021.
//
#include "gtest/gtest.h"
// Always need to define this before including numpy array
#define PY_ARRAY_UNIQUE_SYMBOL RoadRunner_ARRAY_API
#include <numpy/arrayobject.h>
#include "PyUtils.h"
#include <Python.h>



using namespace rr;

class PyUtilsTests : public ::testing::Test {
public:
    PyUtilsTests() = default;
};

/**
 * These tests should be written, but no time now.
 * doublematrix_to_py segfaults for some reason.
 */

//TEST_F(PyUtilsTests, CheckNonStructuredLSMatToPyObject ){
//    ls::Matrix<double> mat({{0.1, 0.2}});
//    PyObject* npArray = doublematrix_to_py(&mat, false, false);
//    std::cout << npArray << std::endl;
//
//}


