/*
 * LLVMCSRMatrixTest.cpp
 *
 *  Created on: Jul 11, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "LLVMCSRMatrixTest.h"
#include "llvm/ModelDataIRBuilder.h"
#include "cpplapack.h"
#include <iostream>

using namespace llvm;
using namespace std;

namespace rr
{

LLVMCSRMatrixTest::LLVMCSRMatrixTest() :
        getFunc(0), setFunc(0)
{
}

LLVMCSRMatrixTest::~LLVMCSRMatrixTest()
{
    // TODO Auto-generated destructor stub
}

bool LLVMCSRMatrixTest::callCSRMatrixSetNZ(csr_matrix* mat, int row, int col,
        double value)
{
    ExecutionEngine &engine = context.getExecutionEngine();


    CsrMatrixSetNZPtr pfunc = (CsrMatrixSetNZPtr)engine.getPointerToFunction(setFunc);


    return pfunc(mat, row, col, value);
}

double LLVMCSRMatrixTest::callCSRMatrixGetNZ(csr_matrix* mat, int row, int col)
{
    ExecutionEngine &engine = context.getExecutionEngine();


    CsrMatrixGetNZPtr pfunc = (CsrMatrixGetNZPtr)engine.getPointerToFunction(getFunc);


    return pfunc(mat, row, col);
}

llvm::Function* LLVMCSRMatrixTest::getCSRMatrixSetNZTestFunc()
{
    LLVMContext &ctx = context.getContext();
    IRBuilder<> &builder = context.getBuilder();
    Module *module = context.getModule();

    setFunc = module->getFunction("test_csr_matrix_set_nz");

    if (setFunc == 0)
    {

        // bool csr_matrix_set_nz(csr_matrix *mat, int row, int col, double val);
        Type *argTypes[] = {
                ModelDataIRBuilder::getCSRSparseStructType(module)->getPointerTo(),
                Type::getInt32Ty(module->getContext()),
                Type::getInt32Ty(module->getContext()),
                Type::getDoubleTy(module->getContext())
        };

        FunctionType *funcType = FunctionType::get(
                IntegerType::get(module->getContext(), sizeof(bool) * 8),
                argTypes, false);
        setFunc = Function::Create(funcType, Function::InternalLinkage,
                "test_csr_matrix_set_nz", module);

        // Create a new basic block to start insertion into.
        BasicBlock *BB = BasicBlock::Create(context.getContext(), "entry",
                setFunc);
        builder.SetInsertPoint(BB);

        std::vector<Value*> args;

        // Set names for all arguments.
        unsigned idx = 0;
        for (Function::arg_iterator ai = setFunc->arg_begin();
                ai != setFunc->arg_end(); ++ai)
        {

            args.push_back(ai);
        }

        LLVMModelDataIRBuilderTesting mdirbuilder(LLVMModelDataSymbols(), builder);

        mdirbuilder.createDispInt(args[1]);
        mdirbuilder.createDispInt(args[2]);
        mdirbuilder.createDispDouble(args[3]);



        CallInst *call = ModelDataIRBuilder::createCSRMatrixSetNZ(builder, args[0], args[1],
                args[2], args[3], "nz_success");

        mdirbuilder.createDispChar(call);

        builder.CreateRet(call);

        // Validate the generated code, checking for consistency.
        verifyFunction(*setFunc);

#if !defined(NDEBUG)
		setFunc->dump();
#endif // !defined(NDEBUG)


    }

    return setFunc;
}

bool runLLVMCSRMatrixTest(const std::string& version, int caseNumber)
{
    LLVMCSRMatrixTest tester;

    Function *func = tester.getCSRMatrixSetNZTestFunc();

#if !defined(NDEBUG)
    func->dump();
#endif

    return true;

}

bool runLLVMCSRMatrixTest(const int m, const int n, const int nnz)
{
    LLVMCSRMatrixTest tester;

    tester.getCSRMatrixSetNZTestFunc();

    tester.getCSRMatrixGetNZTestFunc();


    double *A = (double*) calloc(m * n, sizeof(double));
    double *x = (double*) calloc(n, sizeof(double));
    double *y = (double*) calloc(m, sizeof(double));
    double *ys = (double*) calloc(m, sizeof(double));

    int true_nnz = 0;

    // row major:    offset = row*NUMCOLS + column
    // column major: offset = row + column*NUMROWS

    // set x and first row of A to ones.
    //for (int i = 0; i < n; i++) {
    //    A[0 + i*m] = 1;
    //    x[i] = 1;
    //}

    for (int k = 0; k < nnz; k++)
    {
        int i = rand() % m;
        int j = rand() % n;

        if (A[i + j * m] == 0)
        {
            A[i + j * m] = 1;
            true_nnz++;
        }
    }

    for (int k = 0; k < n; k++)
    {
        x[k] = rand() % 10;
    }

    //cout << "A: ";
    //for(int i = 0; i < m*n; i++) {
    //    cout << A[i] << ",";
    //}
    //cout << "\n";

    cout << "x: ";
    for (int i = 0; i < n; i++)
    {
        cout << x[i] << ",";
    }
    cout << "\n";

    cpp_dgemv('n', m, n, 1, A, m, x, 1, 0, y, 1);

    cout << "dense y:  ";
    for (int i = 0; i < m; i++)
    {
        cout << y[i] << ",";
    }
    cout << "\n";

    vector<uint> rowidx;
    vector<uint> colidx;
    vector<double> values;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double value = A[i + j * m];

            if (value != 0)
            {
                rowidx.push_back(i);
                colidx.push_back(j);
                values.push_back(value);
            }
        }
    }

    assert(rowidx.size() == true_nnz);
    assert(colidx.size() == true_nnz);
    assert(values.size() == true_nnz);

    csr_matrix *mat = csr_matrix_new(m, n, rowidx, colidx, values);

    csr_matrix_dgemv(1, mat, x, 0, ys);

    cout << "sparse y: ";
    for (int i = 0; i < m; i++)
    {
        cout << ys[i] << ",";
    }
    cout << "\n";

    // change some values
    for (int k = 0; k < true_nnz; k++)
    {
        int i = rowidx[k];
        int j = colidx[k];
        double value = rand() % 10;

        tester.callCSRMatrixSetNZ(mat, i, j, value);
        A[i + j * m] = value;

        assert(csr_matrix_get_nz(mat, i, j) == value);

        assert(tester.callCSRMatrixGetNZ(mat, i, j) == value);
    }

    memset(y, 0, m * sizeof(double));
    cpp_dgemv('n', m, n, 1, A, m, x, 1, 0, y, 1);

    cout << "dense y:  ";
    for (int i = 0; i < m; i++)
    {
        cout << y[i] << ",";
    }
    cout << "\n";

    memset(ys, 0, m * sizeof(double));
    csr_matrix_dgemv(1, mat, x, 0, ys);

    cout << "sparse y: ";
    for (int i = 0; i < m; i++)
    {
        cout << ys[i] << ",";
    }
    cout << "\n";

    csr_matrix_delete(mat);

    free(A);
    free(x);
    free(y);
    free(ys);

    return true;
}

llvm::Function* LLVMCSRMatrixTest::getCSRMatrixGetNZTestFunc()
{
    LLVMContext &ctx = context.getContext();
    IRBuilder<> &builder = context.getBuilder();
    Module *module = context.getModule();

    getFunc = module->getFunction("test_csr_matrix_get_nz");

    if (getFunc == 0)
    {
        // double csr_matrix_get_nz(const csr_matrix *mat, int row, int col);
        Type *argTypes[] = {
                ModelDataIRBuilder::getCSRSparseStructType(module)->getPointerTo(),
                Type::getInt32Ty(module->getContext()),
                Type::getInt32Ty(module->getContext())
        };

        FunctionType *funcType = FunctionType::get(
                Type::getDoubleTy(module->getContext()),
                argTypes, false);
        getFunc = Function::Create(funcType, Function::InternalLinkage,
                "test_csr_matrix_get_nz", module);

        // Create a new basic block to start insertion into.
        BasicBlock *BB = BasicBlock::Create(context.getContext(), "entry",
                getFunc);
        builder.SetInsertPoint(BB);

        std::vector<Value*> args;

        // Set names for all arguments.
        unsigned idx = 0;
        for (Function::arg_iterator ai = getFunc->arg_begin();
                ai != getFunc->arg_end(); ++ai)
        {

            args.push_back(ai);
        }

        LLVMModelDataIRBuilderTesting mdirbuilder(LLVMModelDataSymbols(), builder);

        mdirbuilder.createDispInt(args[1]);
        mdirbuilder.createDispInt(args[2]);

        CallInst *call = ModelDataIRBuilder::createCSRMatrixGetNZ(builder, args[0], args[1],
                args[2], "nz_val");

        mdirbuilder.createDispDouble(call);

        builder.CreateRet(call);

        // Validate the generated code, checking for consistency.
        verifyFunction(*getFunc);

#if !defined(NDEBUG)
        getFunc->dump();
#endif
    }

    return setFunc;
}


} /* namespace rr */

