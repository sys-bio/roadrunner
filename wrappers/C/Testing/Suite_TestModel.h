/*
 * Suite_TestModel.h
 *
 *  Created on: Aug 28, 2014
 *      Author: andy
 */

#ifndef SUITE_TESTMODEL_H_
#define SUITE_TESTMODEL_H_

#include <string>
#include <set>

namespace SuiteTEST_MODEL {

    /**
     * enumerate the test files in the testing dir.
     *
     * If dir is empty, then the default location for test files
     * is used which is $ROADRUNNER_DIR/testing.
     */
    std::set<std::string> getTestFiles(const std::string& dir = "");

    /**
     * set the current test data file that the Suite_TESTMODEL
     * test will uses. This usually obtained from getTestFiles();
     */
    void setTestFile(const std::string& filePath);

} /* namespace rr */

#endif /* SUITETESTMODEL_H_ */
