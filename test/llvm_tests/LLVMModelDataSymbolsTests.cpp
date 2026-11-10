//
// Created by Ciaran on 15/10/2021.
//

#include "gtest/gtest.h"
#include "TestModelFactory.h"
#include "sbml/SBMLTypes.h"
#include "llvm/LLVMModelDataSymbols.h"

// todo finish or delete

using namespace rr;
using namespace libsbml;

class LLVMModelDataSymbolTests : public :: testing::Test {
public:
    std::unique_ptr<SBMLDocument> doc = nullptr;

    // borrowed reference
    Model* model = nullptr;
    LLVMModelDataSymbolTests() {
        Logger::setFormattingPattern("%U:%u:%l: %t");
        //Logger::setLevel(Logger::LOG_DEBUG);
        doc = std::make_unique<SBMLDocument>(3, 2);
        model = doc->createModel("ID");
    };
};

//
TEST_F(LLVMModelDataSymbolTests, ModelName){
    model->setName("Mmmmodel");
    rrllvm::LLVMModelDataSymbols sym(model, 0);
    ASSERT_STREQ("Mmmmodel", sym.getModelName().c_str());
}
















