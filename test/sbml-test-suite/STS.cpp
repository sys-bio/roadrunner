//
// Created by Ciaran on 13/12/2021.
//

#include "STS.h"
#include <filesystem>
#include <regex>

class FileNotFound : public std::logic_error {
public:
    explicit FileNotFound(const std::string &msg);
};

FileNotFound::FileNotFound(const std::string &msg) : logic_error(msg) {}

std::string errorMsg(const std::filesystem::path &varName) {
    std::ostringstream os;
    os << "The \"" << varName << "\" variable is not defined. Make sure you have added it to your library or "
                                 "executable using cmake's target_compile_options command";
    return os.str();
}

STSModel::STSModel(unsigned int caseID) : caseId(caseID) {
#ifndef SBMLTestSuiteRoot
    throw std::logic_error(errorMsg("SBMLTestSuiteRoot"));
#endif

    if (!std::filesystem::exists(SBMLTestSuiteRoot)) {
        throw std::invalid_argument("Path to SBMLTestSuiteRoot does not exist");
    }
    root_ = SBMLTestSuiteRoot;
    std::ostringstream caseIdStream;
    caseIdStream << std::setw(5) << std::setfill('0') << caseID;
    caseIdStr_ = caseIdStream.str();
}

std::string STSModel::buildCaseDir() {
    return (root_ / caseIdStr_).string();
};


std::string STSModel::getRootDirectory() {
    return root_.string();
}

std::string STSModel::getLevelAndVersion(int level, int version) {
    std::ostringstream levelAndVersion;
    levelAndVersion << caseIdStr_ << "-sbml-l" << level << "v" << version << ".xml";
    std::filesystem::path p = caseDir_ / levelAndVersion.str();
    if (!std::filesystem::exists(p)) {
        throw FileNotFound(p.string());
    }
    return p.string();
}

std::string STSModel::getNewestLevelAndVersion() {
    try {
        return getLevelAndVersion(3, 2);
    } catch (std::exception &e) {
        try {
            return getLevelAndVersion(3, 1);
        } catch (std::exception &e) {
            try {
                return getLevelAndVersion(2, 4);
            } catch (std::exception &e) {
                try {
                    return getLevelAndVersion(2, 3);
                } catch (std::exception &e) {
                    try {
                        return getLevelAndVersion(2, 2);
                    } catch (std::exception &e) {
                        try {
                            return getLevelAndVersion(2, 1);
                        } catch (std::exception &e) {
                            rrLogErr << "Failed to find case l" << 2 << "v" << 1;
                            throw std::logic_error("can't locate sbml file");
                        }
                    }
                }
            }
        }
    }
    return {};
}

std::string STSModel::getModelDescriptionFile() {
    std::ostringstream desc;
    desc << caseIdStr_ << "-model.m";
    std::filesystem::path p = caseDir_ / desc.str();
    if (!std::filesystem::exists(p)) {
        throw FileNotFound(p.string());
    }
    return p.string();
}

std::string STSModel::readModelDescriptionFile() {
    std::lock_guard<std::mutex> lock(STSMtx);
    std::ifstream ifstream(getModelDescriptionFile());
    std::ostringstream os;
    for (std::string line; std::getline(ifstream, line);) {
        os << line << std::endl;
    }
    return os.str();
}


std::string STSModel::getResultsFile() {
    std::ostringstream res;
    res << caseIdStr_ << "-results.csv";
    std::filesystem::path p = caseDir_ / res.str();
    if (!std::filesystem::exists(p)) {
        throw FileNotFound(p.string());
    }
    return p.string();
}

std::string STSModel::getSettingsFile() {
    std::ostringstream settings;
    settings << caseIdStr_ << "-settings.txt";
    std::filesystem::path p = caseDir_ / settings.str();
    if (!std::filesystem::exists(p)) {
        throw FileNotFound(p.string());
    }
    return p.string();
}

unsigned int STSModel::getCaseId() const {
    return caseId;
}

std::vector<std::string> STSModel::getComponentTags() {
    if (modelDescriptionFileString.empty()) {
        modelDescriptionFileString = readModelDescriptionFile();
    }

    std::regex component_tags_regex("componentTags:\\s*(.*)");
    std::smatch regexResults;
    if (!std::regex_search(modelDescriptionFileString, regexResults, component_tags_regex)) {
        throw std::logic_error("Unable to locate component tags");
    };

    std::string componentTagsStr = regexResults.str(1);
    std::vector<std::string> componentTags = splitString(componentTagsStr);

    return componentTags;
}

std::vector<std::string> STSModel::getTestTags() {
    if (modelDescriptionFileString.empty()) {
        modelDescriptionFileString = readModelDescriptionFile();
    }

    std::regex testTagsRegex("testTags:\\s*(.*)");
    std::smatch regexResults;
    if (!std::regex_search(modelDescriptionFileString, regexResults, testTagsRegex)) {
        throw std::logic_error("Unable to locate test tags");
    };

    std::string testTagsStr = regexResults.str(1);
    std::vector<std::string> testTags = splitString(testTagsStr);

    return testTags;
}


std::vector<std::string> STSModel::getTestType() {
    if (modelDescriptionFileString.empty()) {
        modelDescriptionFileString = readModelDescriptionFile();
    }

    std::regex testTypeRegex("testType:\\s*(.*)");
    std::smatch regexResults;
    if (!std::regex_search(modelDescriptionFileString, regexResults, testTypeRegex)) {
        throw std::logic_error("Unable to locate test type");
    };

    std::string testTypesStr = regexResults.str(1);
    std::vector<std::string> testTypes = splitString(testTypesStr);

    return testTypes;


}


//std::vector<std::string> STSModel::splitString2(std::string stringToSplit){
//    std::vector<std::string> out;
//    auto it=stringToSplit.begin();
//    while (it!= stringToSplit.end()){
//        std::string s;
//        while (it != stringToSplit.end() && *it != ',' && *it != ' '){
//                s += *it;
//                ++it;
//        }
//        if (!s.empty())
//            out.push_back(s);
//        ++it;
//    }
//    return out;
//}
std::vector<std::string> STSModel::splitString(const std::string& stringToSplit) {
    std::stringstream ss;
    ss << stringToSplit;
    std::vector <std::string> result;

    while (ss.good()) {
        std::string substr;
        getline(ss, substr, ',');
        // remove begin white space
        while (*(substr.begin()) == ' '){
            substr.erase(substr.begin());
        }
        // remove end white space
        while (*(substr.end()-1) == ' '){
            substr.erase(substr.end()-1);
        }
        result.push_back(substr);
    }
    return result;
}

SemanticSTSModel::SemanticSTSModel(unsigned int caseId)
        : STSModel(caseId) {
#ifndef SBMLTestSuiteSemantic
    throw std::logic_error(errorMsg("SBMLTestSuiteSemantic"));
#endif
    if (!std::filesystem::exists(SBMLTestSuiteSemantic)) {
        throw std::invalid_argument("Path to SBMLTestSuiteSemantic does not exist");
    }

    root_ = SBMLTestSuiteSemantic;
    if (caseId < 1 || caseId > 1809) {
        throw std::invalid_argument("Models in the stochastic STS are numbered 1 through 1809 inclusive");
    }
    caseDir_ = SemanticSTSModel::buildCaseDir();
}

unsigned int SemanticSTSModel::maxNum() {
    return 1809;
}

StochasticSTSModel::StochasticSTSModel(unsigned int caseId)
        : STSModel(caseId) {
#ifndef SBMLTestSuiteStochastic
    throw std::logic_error(errorMsg("SBMLTestSuiteStochastic"));
#endif
    if (!std::filesystem::exists(SBMLTestSuiteStochastic)) {
        throw std::invalid_argument("Path to SBMLTestSuiteStochastic does not exist");
    }
    root_ = SBMLTestSuiteStochastic;
    if (caseId < 1 || caseId > 100) {
        throw std::invalid_argument("Models in the stochastic STS are numbered 1 through 100 inclusive");
    }
    caseDir_ = StochasticSTSModel::buildCaseDir();

}


std::filesystem::path StochasticSTSModel::getMeanFile() {
    // solution relies on fact that there is only 1 file with .mean.csv in the name
    for (auto &curr: std::filesystem::directory_iterator(caseDir_)) {
        if (curr.path().string().find("-mean.csv") != std::string::npos) {
            return curr.path();
        }
    }
    // if we get here then file not exist
    throw FileNotFound("dsmts-xxx-xx-mean.csv");
    return {};
}

std::filesystem::path StochasticSTSModel::getSDFile() {
    // solution relies on fact that there is only 1 file with .mean.csv in the name
    for (auto &curr: std::filesystem::directory_iterator(caseDir_)) {
        if (curr.path().string().find("-sd.csv") != std::string::npos) {
            return curr.path();
        }
    }
    // if we get here then file not exist
    throw FileNotFound("dsmts-xxx-xx-sd.csv");
    return {};
}


std::filesystem::path StochasticSTSModel::getModFile() {
    // solution relies on fact that there is only 1 file with .mean.csv in the name
    for (auto &curr: std::filesystem::directory_iterator(caseDir_)) {
        if (curr.path().string().find(".mod") != std::string::npos) {
            return curr.path();
        }
    }
    // if we get here then file not exist
    throw FileNotFound("dsmts-xxx-xx.mod");
    return {};
}

unsigned int StochasticSTSModel::maxNum() {
    return 100;
}



//std::filesystem::path STS::getModelNFromSTS(int caseId, int level, int version) {
//    std::ostringstream caseOs;
//    std::ostringstream fname;
//
//    caseOs << std::setfill('0') << std::setw(5) << caseId;
//    fname << caseOs.str() << "-sbml-l" << level << "v" << version << ".xml";
//
//    std::filesystem::path root = SBMLTestSuiteSemantic;
//    std::filesystem::path caseDir = root / caseOs.str();
//    std::filesystem::path xmlFile = caseDir / fname.str();
//    if (!std::filesystem::exists(xmlFile)) {
//        std::ostringstream os;
//        os << "file not found : " << xmlFile;
//        throw std::invalid_argument(os.str());
//    }
//    return xmlFile.string();
//}
//
//std::filesystem::path STS::getNewestModelNFromSTS(int caseId) {
//    if (caseId < 1 || caseId > 1809) {
//        throw std::invalid_argument("caseId must be between 1 and 1809");
//    }
//    try {
//        return getModelNFromSTS(caseId, 3, 2);
//    } catch (std::exception &e) {
//        rrLogInfo << "Failed to find case " << caseId << ", l" << 3 << "v" << 2 << ". Looking for another version...";
//        try {
//            return getModelNFromSTS(caseId, 3, 1);
//        } catch (std::exception &e) {
//            rrLogInfo << "Failed to find case " << caseId << ", l" << 3 << "v" << 1
//                      << ". Looking for another version...";
//            try {
//                return getModelNFromSTS(caseId, 2, 4);
//            } catch (std::exception &e) {
//                rrLogInfo << "Failed to find case " << caseId << ", l" << 2 << "v" << 4
//                          << ". Looking for another version...";
//                try {
//                    return getModelNFromSTS(caseId, 2, 3);
//                } catch (std::exception &e) {
//                    rrLogInfo << "Failed to find case " << caseId << ", l" << 2 << "v" << 2
//                              << ". Looking for another version...";
//                    try {
//                        return getModelNFromSTS(caseId, 2, 2);
//                    } catch (std::exception &e) {
//                        rrLogInfo << "Failed to find case " << caseId << ", l" << 2 << "v" << 2
//                                  << ". Looking for another version...";
//                        try {
//                            return getModelNFromSTS(caseId, 2, 1);
//                        } catch (std::exception &e) {
//                            rrLogErr << "Failed to find case " << caseId << ", l" << 2 << "v" << 1;
//                            throw std::logic_error("can't locate sbml file");
//                        }
//                    }
//                }
//            }
//        }
//    }
//    return {};
//}
//
//
//std::vector<std::filesystem::path> STS::getFirstNModelsFromSTS(int N, int start) {
//    if (start < 1) {
//        throw std::invalid_argument("start must be 1 or more");
//    }
//    int end = start + N;
//    int maxSize = 1809;
//    if (end > maxSize) {
//        throw std::invalid_argument("start + N must be less than or equal to 1809: " + std::to_string(end));
//    }
//    std::vector<std::filesystem::path> vec(N);
//    for (int i = start; i < end; i++) {
//        int idx = i - start;
//        vec[idx] = getNewestModelNFromSTS(i);
//    }
//    return vec;
//}
//
//std::vector<std::filesystem::path> STS::getModelsFromSTS(const std::vector<int> &caseIDs) {
//    std::vector<std::filesystem::path> vec(caseIDs.size());
//    int i = 0;
//    for (auto caseId: caseIDs) {
//        vec[i++] = getNewestModelNFromSTS(caseId);
//    }
//    return vec;
//}
