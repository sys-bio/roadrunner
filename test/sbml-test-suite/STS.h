//
// Created by Ciaran on 13/12/2021.
//

#ifndef ROADRUNNER_STS_H
#define ROADRUNNER_STS_H

#include <iostream>
#include <vector>
#include "rrLogger.h"
#include <filesystem>
#include "fstream"
#include <mutex>
#include "parallel-hashmap-1.33/parallel_hashmap/phmap.h"
#include "thread_pool.hpp"
#include <mutex>

using namespace rr;

static std::mutex STSMutex;

class STSModel {
public:
    STSModel() = default;

    explicit STSModel(unsigned caseID);

    unsigned int getCaseId() const;


    /**
     * @brief return the root to the STS
     */
    std::string getRootDirectory();

    /**
     * @brief returns the path to the model description file (*.m)
     */
    std::string getModelDescriptionFile();

    /**
     * @brief returns the text inside the model description file.
     * REading is locked via a mutex to enable multithreading.
     */
    std::string readModelDescriptionFile();

    /**
     * @brief returns path to expected results file
     */
    std::string getResultsFile();

    /**
     * @brief returns path to settings file
     */
    std::string getSettingsFile();

    /**
     * @brief returns path to sbml level and version file,
     * if it exists. Throws otherwise.
     */
    std::string getLevelAndVersion(int level, int version);

    /**
     * @brief get the path to the newest sbml file available
     */
    std::string getNewestLevelAndVersion();

    /**
     * @brief get the component tags from model description file
     */
    std::vector<std::string> getComponentTags();

    /**
     * @brief get the test tags from model description file
     */
    std::vector<std::string> getTestTags();

    /**
     * @brief get the test type from model description file
     */
    std::vector<std::string> getTestType();

    unsigned int caseId = 1;

    /**
     * @brief string variable for caching the model description file stirng.
     * @details multiple methods need access to this string. So instead of expensive
     * reads, read only if this string is empty.
     */
    std::string modelDescriptionFileString = "";

protected:

    std::string buildCaseDir();

    /**
     * @brief splits a string on commas. Returns values as vector of strings.
     */
    std::vector<std::string> splitString(const std::string& s);

    std::string caseIdStr_;

    /**
     * Path to the root of the test suite.
     */
    std::filesystem::path root_;

    /**
     * @brief path to the directory containing models
     * for caseID STS::caseId
     */
    std::filesystem::path caseDir_;

};

class SemanticSTSModel : public STSModel {
public:
    SemanticSTSModel() = default;

    explicit SemanticSTSModel(unsigned int caseId);

    /**
     * @brief returns the root directory of the semantic
     * sbml test suite.
     */
    using STSModel::getRootDirectory;

    static unsigned int maxNum();

};

class StochasticSTSModel : public STSModel {
public:
    explicit StochasticSTSModel(unsigned int caseId);

    static unsigned int maxNum();

    /**
     * @brief returns the root directory of the stochastic
     * sbml test suite.
     */
    using STSModel::getRootDirectory;

    std::filesystem::path getMeanFile();

    std::filesystem::path getSDFile();

    std::filesystem::path getModFile();

};

static std::mutex STSMtx;


/**
 * @brief Little interface into sbml-test-suite.
 *
 * @details implementation details: we have to choose between two annoying options:
 *  1) Use template. Semantic or Stochastic.
 *  2) No template but use a factory method which returns dynamic memory.
 */
template<typename SemanticOrStochastic = SemanticSTSModel>
class STS {
public:

    STS() {
        // increase by 1 so that we can use caseId as vector index
        data_.resize(SemanticOrStochastic::maxNum() + 1);
        for (int i = 1; i < SemanticOrStochastic::maxNum(); i++) {
            data_[i] = SemanticOrStochastic(i);
        }
    };

    /**
     * @brief return the absolute file path to model number @param caseId
     * from the sbml-test-suite (STS).
     * @param level sbml level
     * @param version sbml version
     * @details Throws invalid_argument if the caseId, level and version does not
     * exist.
     */
    std::string getModelNFromSTS(int caseId, int level, int version) {
        return SemanticOrStochastic(caseId).getLevelAndVersion(level, version);
    }

    /**
     * @brief Return a data container representing the sbml test suite
     * model @param caseId.
     * @return Either a SemanticSTSModel or StochasticSTSModel
     * depending on the template parameter SemanticOrStochastic.
     */
    SemanticOrStochastic getModelNFromSTS(int caseId) {
        return SemanticOrStochastic(caseId);
    }

    /**
     * @brief similar to STS::getModelNFromSTS but always try to get the most recent
     * version of the @param caseId requested. Throws invalid_argument if there are no
     * levels or version from the requested @param caseId
     */
    std::string getNewestModelNFromSTS(int caseId) {
        return SemanticOrStochastic(caseId).getNewestLevelAndVersion();
    }

    /**
     * @brief Get absolute paths to models in the STS corresponding to @param caseIDs.
     * @details prefer the newest level and version available, but if it does not exist look for
     * an earlier version.
     */
    std::vector<std::string> getModelsFromSTS(const std::vector<int> &caseIDs) {
        std::vector<std::string> v(caseIDs.size());
        int i = 0;
        for (auto id: caseIDs) {
            v[i++] = data_[id].getNewestLevelAndVersion();
        }
        return v;
    }

    /**
     * @brief get models from between @param first and @param last
     * of the sbml test suite.
     * @returns vector of absolute directories to STS models. Newest
     * is preferred.
     */
    std::vector<std::string> getModelsFromSTS(int first, int last) {
        std::vector<std::string> v(last - first);
        int i = 0;
        for (int modelId = first; modelId < last; modelId++) {
            v[i++] = data_[modelId].getNewestLevelAndVersion();
        }
        return v;
    }


    std::vector<std::string> getModelsFromSTSAsStrings(int first, int last) {
        std::vector<std::string> v = getModelsFromSTS(first, last);
        for (int i = 0; i < v.size(); i++) {
            std::lock_guard<std::mutex> lock(STSMtx);
            std::ifstream f(v[i]);
            std::string fileContents;
            std::copy(
                    std::istreambuf_iterator<char>(f),
                    std::istreambuf_iterator<char>(),
                    std::back_inserter(fileContents)
            );
            v[i] = fileContents;
        }
        return v;
    }


    std::vector<SemanticSTSModel>::iterator begin() {
        return data_.begin();
    }

    std::vector<SemanticSTSModel>::iterator end() {
        return data_.end();
    }

    phmap::parallel_node_hash_map<std::string, unsigned int> searchTestTypes();

private:
    std::vector<SemanticSTSModel> data_;

};


#endif //ROADRUNNER_STS_H
