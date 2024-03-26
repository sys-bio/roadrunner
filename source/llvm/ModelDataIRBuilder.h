/*
 * ModelDataIRBuilder.h
 *
 * Created on: Jun 28, 2013
 *
 * Author: Andy Somogyi,
 *     email decode: V1 = "."; V2 = "@"; V3 = V1;
 *     andy V1 somogyi V2 gmail V3 com
 */

#ifndef ModelDataIRBuilderH
#define ModelDataIRBuilderH

#include "LLVMIncludes.h"
#include "LLVMModelDataSymbols.h"

#include <map>
#include <llvm/IR/DataLayout.h>

namespace libsbml
{
class Model;
}

namespace rrllvm
{
    /** @class ModelDataIRBuilder
    * The model data IR builder.  (Sorry, this is very old code.)
    */

    class ModelDataIRBuilder
    {
    public:

        /**
         * these only make sense attached to a modelData pointer.
         */
        ModelDataIRBuilder(llvm::Value* modelData, LLVMModelDataSymbols const&, llvm::IRBuilder<>&);

        llvm::Value* createFloatSpeciesAmtGEP(const std::string& id,
            const llvm::Twine& name = "");

        llvm::Value* createBoundSpeciesAmtGEP(const std::string& id,
            const llvm::Twine& name = "");

        llvm::Value* createGlobalParamGEP(const std::string& id);

        /**
         * create a GEP for a ModelData structure field.
         *
         * @param field value for a ModelData pointer type.
         */
        llvm::Value* createGEP(ModelDataFields field);

        /**
         * create a GEP for an array which belongs to the ModelData struct,
         * only valid for arrays, will crash on non-array fields.
         */
        llvm::Value* createGEP(ModelDataFields field, unsigned index,
            const llvm::Twine& name = "");

        llvm::Value* createLoad(ModelDataFields field, unsigned index,
            const llvm::Twine& name = "");

        llvm::Value* createStore(ModelDataFields field, unsigned index,
            llvm::Value* value, const llvm::Twine& name = "");

        /**
         * the GEP for a compartment
         */
        llvm::Value* createCompGEP(const std::string& id);

        /**
         * load the compartment value
         */
        llvm::Value* createCompLoad(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * store the compartment value
         */
        llvm::Value* createCompStore(const std::string& id,
            llvm::Value* value);

        /**
         * the GEP for a floating species initial amount
         */
        llvm::Value* createInitFloatSpeciesAmtGEP(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * the GEP for a boundary species initial amount
         */
        llvm::Value* createInitBoundarySpeciesAmtGEP(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * load the floating species initial amount value
         */
        llvm::Value* createInitFloatSpeciesAmtLoad(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * load the boundary species initial amount value
         */
        llvm::Value* createInitBoundarySpeciesAmtLoad(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * store the floating species initial value
         */
        llvm::Value* createInitFloatSpeciesAmtStore(const std::string& id,
            llvm::Value* value);

        /**
         * store the boundary species initial value
         */
        llvm::Value* createInitBoundarySpeciesAmtStore(const std::string& id,
            llvm::Value* value);

        /**
         * the GEP for a compartment
         */
        llvm::Value* createInitCompGEP(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * load the compartment value
         */
        llvm::Value* createInitCompLoad(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * store the compartment value
         */
        llvm::Value* createInitCompStore(const std::string& id,
            llvm::Value* value);


        /**
         * the GEP for a init global params
         */
        llvm::Value* createInitGlobalParamGEP(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * load the init global param value
         */
        llvm::Value* createInitGlobalParamLoad(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * store the init global param value
         */
        llvm::Value* createInitGlobalParamStore(const std::string& id,
            llvm::Value* value);

        /**
         * load the global param value
         */
        llvm::Value* createGlobalParamLoad(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * store the global param value
         */
        llvm::Value* createGlobalParamStore(const std::string& id,
            llvm::Value* value);

        /**
         * load the floating species amount value
         */
        llvm::Value* createFloatSpeciesAmtLoad(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * store the floating species amount
         */
        llvm::Value* createFloatSpeciesAmtStore(const std::string& id,
            llvm::Value* value);

        /**
         * the GEP for the amount rates
         */
        llvm::Value* createFloatSpeciesAmtRateGEP(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * load the floating species amount value
         */
        llvm::Value* createFloatSpeciesAmtRateLoad(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * store the floating species amount
         */
        llvm::Value* createFloatSpeciesAmtRateStore(const std::string& id,
            llvm::Value* value);

        /**
         * load the floating species amount value
         */
        llvm::Value* createBoundSpeciesAmtLoad(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * store the floating species amount
         */
        llvm::Value* createBoundSpeciesAmtStore(const std::string& id,
            llvm::Value* value);

        /**
         * load the floating species amount value
         */
        llvm::Value* createReactionRateLoad(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * store the floating species amount
         */
        llvm::Value* createReactionRateStore(const std::string& id,
            llvm::Value* value);

        /**
         * rate rule GEP
         */
        llvm::Value* createRateRuleValueGEP(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * load the rate rule value
         */
        llvm::Value* createRateRuleValueLoad(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * store the rate rule value
         */
        llvm::Value* createRateRuleValueStore(const std::string& id,
            llvm::Value* value);

        /**
         * rate rule GEP
         */
        llvm::Value* createRateRuleRateGEP(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * load the rate rule rate
         */
        llvm::Value* createRateRuleRateLoad(const std::string& id,
            const llvm::Twine& name = "");

        /**
         * store the rate rule rate
         */
        llvm::Value* createRateRuleRateStore(const std::string& id,
            llvm::Value* value);

        llvm::Value* createStoichiometryStore(uint row, uint col,
            llvm::Value* value, const llvm::Twine& name = "");

        llvm::Value* createStoichiometryLoad(uint row, uint col,
            const llvm::Twine& name = "");

        /**
         * pointer to ModelData::random field.
         */
        llvm::Value* createRandomLoad();

        /**
         * create a call to the csr_matrix_set_nz function.
         *
         * only valid whilst a basic block is being filled.
         */
        static llvm::CallInst* createCSRMatrixSetNZ(llvm::IRBuilder<>&,
            llvm::Value* csrPtr, llvm::Value* row, llvm::Value* col,
            llvm::Value* value, const llvm::Twine& name = "");

        /**
         * create a call to the csr_matrix_get_nz function.
         *
         * only valid whilst a basic block is being filled.
         */
        static llvm::CallInst* createCSRMatrixGetNZ(llvm::IRBuilder<>&,
            llvm::Value* csrPtr, llvm::Value* row, llvm::Value* col,
            const llvm::Twine& name = "");

        /**
         * get the function declaration (prototype) for the csr_matrix_set_nz function.
         *
         * create if not found.
         */
        static llvm::Function* getCSRMatrixSetNZDecl(llvm::Module* module);

        /**
         * get the function declaration (prototype) for the csr_matrix_set_nz function.
         *
         * create if not found
         */
        static llvm::Function* getCSRMatrixGetNZDecl(llvm::Module* module);

        /**
         * get the struct type for the CSR sparse storage struct.
         *
         * create if not found
         */
        static llvm::StructType* getCSRSparseStructType(llvm::Module* module,
            llvm::ExecutionEngine* engine = 0);

        /**
         * Creates a ModelData struct for the given sbml symbols in the
         * llvm Module.
         *
         * This must be called before any model accessing code is generated.
         */
        static llvm::StructType* createModelDataStructType(llvm::Module* module,
            llvm::ExecutionEngine* engine, LLVMModelDataSymbols const& symbols);

        /**
         * get the ModelData struct type.
         *
         * createModelDataStructType must have been called for this to work.
         *
         * Throws an exception if the ModelData struct could not be obtained.
         */
        static llvm::StructType* getStructType(llvm::Module* module);

        static unsigned getModelDataSize(llvm::Module* module, const llvm::DataLayout& dl);

        static const char* LLVMModelDataName;
        static const char* csr_matrixName;
        static const char* csr_matrix_set_nzName;
        static const char* csr_matrix_get_nzName;

    private:
        /**
         * the model data struct we generate the loads / stores relative to.
         */
        llvm::Value* modelData;

        llvm::IRBuilder<>& builder;

        LLVMModelDataSymbols const& symbols;

        /**
         * get the module, only valid whilst a BasicBlock is begin filled.
         *
         * @param builder the  name of the calling function, used for generating exception
         * @param func name of the calling function, used for generating exception
         * on failure.
         *
         * @returns a module on success, throws exeption on failure.
         */
        static llvm::Module* getModule(llvm::IRBuilder<>& builder, const char* func);

        /**
         * check to see that s is a ModelData struct pointer,
         * throw an exception if not.
         */
        void validateStruct(llvm::Value* s, const char* funcName);

        friend class LLVMModelDataIRBuilderTesting;
    };

    /** @class LLVMModelDataIRBuilderTesting
     * testing class
     */
    class LLVMModelDataIRBuilderTesting
    {
    public:

        LLVMModelDataIRBuilderTesting(LLVMModelDataSymbols const&,
            llvm::IRBuilder<>&);

        /*********************** TESTING STUFF WILL GO AWAY EVENTUALLY ***********************/

        void createAccessors(llvm::Module* module);

        std::pair<llvm::Function*, llvm::Function*> createFloatingSpeciesAccessors(
            llvm::Module* module, const std::string id);


        static llvm::Function* getDispIntDecl(llvm::Module* module);
        llvm::CallInst* createDispInt(llvm::Value* intVal);

        static llvm::Function* getDispDoubleDecl(llvm::Module* module);
        llvm::CallInst* createDispDouble(llvm::Value* doubleVal);

        static llvm::Function* getDispCharDecl(llvm::Module* module);
        llvm::CallInst* createDispChar(llvm::Value* doubleVal);

        /*************************************************************************************/


        llvm::IRBuilder<>& builder;

        LLVMModelDataSymbols const& symbols;
    };

} /* namespace rr */
#endif /* ModelDataIRBuilderH */
