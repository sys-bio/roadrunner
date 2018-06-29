/*
 * rrLLVMCodeGenerator.cpp
 *
 * Created on: Jun 16, 2013
 *
 * Author: Andy Somogyi,
 *     email decode: V1 = "."; V2 = "@"; V3 = V1;
 *     andy V1 somogyi V2 gmail V3 com
 */
#define _USE_MATH_DEFINES

#pragma hdrstop
#include "ASTNodeCodeGen.h"
#include "LLVMException.h"
#include "rrOSSpecifics.h"
#include "LLVMIncludes.h"
#include "rrLogger.h"
#include "rrStringUtils.h"
#include "rrConfig.h"

#include <sbml/math/ASTNode.h>
#include <sbml/math/FormulaFormatter.h>
#include <sbml/SBase.h>
#include <Poco/Logger.h>
#include <cmath>

using namespace libsbml;
using namespace llvm;
using namespace std;
using namespace rr;

namespace rrllvm
{

/**
 * idiotic test to test for an idiotic condition. Checks if there is a minus
 * in front of the expression so we can pass tests with arccot(-0)...
 */
static bool isNegative(const libsbml::ASTNode *node);

std::string to_string(const libsbml::ASTNode *ast)
{
    char* formula = SBML_formulaToString(ast);
    string str = formula;
    free(formula);
    return str;
}

ASTNodeCodeGen::ASTNodeCodeGen(llvm::IRBuilder<> &builder,
        LoadSymbolResolver &resolver) :
        builder(builder), resolver(resolver),
        scalar_mode_(false)
{
}

ASTNodeCodeGen::~ASTNodeCodeGen()
{
}

Value *ASTNodeCodeGen::binaryExprCodeGen(const ASTNode *ast)
{
    Value *result = 0;

    Value *lhs = toDouble(codeGen(ast->getLeftChild()));
    Value *rhs = toDouble(codeGen(ast->getRightChild()));
    if (lhs == 0 || rhs == 0)
    {
        // TODO: report this error???
        return 0;
    }

    switch (ast->getType())
    {
    case AST_PLUS:
        result = builder.CreateFAdd(lhs, rhs, "addtmp");
        break;
    case AST_MINUS:
        result = builder.CreateFSub(lhs, rhs, "subtmp");
        break;
    case AST_TIMES:
        result = builder.CreateFMul(lhs, rhs, "multmp");
        break;
    case AST_DIVIDE:
        result = builder.CreateFDiv(lhs, rhs, "divtmp");
        break;
        /*
         case '<':
         L = Builder.CreateFCmpULT(L, R, "cmptmp");
         // Convert bool 0/1 to double 0.0 or 1.0
         return Builder.CreateUIToFP(L, Type::getDoubleTy(getGlobalContext()),
         "booltmp");
         */
    default:
        break;
    }

    return result;
}

llvm::Value* ASTNodeCodeGen::codeGen(const libsbml::ASTNode* ast)
{
    Value *result = 0;

    if (ast == 0)
    {
        throw_llvm_exception("ASTNode is NULL");
    }

    switch (ast->getType())
    {
    case AST_PLUS:                  /* '+' */
    case AST_MINUS:                 /* '-' */
    case AST_TIMES:                 /* '*' */
    case AST_DIVIDE: {              /* '/' */
        ASTNodeCodeGenScalarTicket t(*this, true);
        result = applyArithmeticCodeGen(ast);
        break;
    }
    case AST_INTEGER: {
        result = integerCodeGen(ast);
        break;
    }
    case AST_REAL:
    case AST_REAL_E:
    case AST_RATIONAL: {
        ASTNodeCodeGenScalarTicket t(*this, true);
        result = realExprCodeGen(ast);
        break;
    }
    case AST_NAME:
    case AST_NAME_AVOGADRO:
    case AST_NAME_TIME: {
        result = nameExprCodeGen(ast);
        break;
    }
    case  AST_RELATIONAL_EQ:
    case  AST_RELATIONAL_GEQ:
    case  AST_RELATIONAL_GT:
    case  AST_RELATIONAL_LEQ:
    case  AST_RELATIONAL_LT:
    case  AST_RELATIONAL_NEQ:
		// AHu: For now I am unsure of what it's supposed to do when not in scalar mode,
		// so I am changing the non-scalar relation into a generic relational code gen,
		// and calling scalar relation code gen all the time, since it checks the number
		// of arguments to the relational operator (which caused problems with MathML
		//result = applyScalarRelationalCodeGen(ast);
		/*{
			Value* left = toDouble(codeGen(ast->getLeftChild()));
			Value* right = toDouble(codeGen(ast->getRightChild()));
			result = applyBinaryRelationalCodeGen(ast, left, right);
		}*/
		if (scalar_mode_) {
			result = applyScalarRelationalCodeGen(ast);
		} else {
			result = applyRelationalCodeGen(ast);
		}
        break;

    case  AST_LOGICAL_AND:
    case  AST_LOGICAL_NOT:
    case  AST_LOGICAL_OR:
    case  AST_LOGICAL_XOR: {
        result = applyLogicalCodeGen(ast);
        break;
    }
    case  AST_LOGICAL_IMPLIES: {
        result = toDouble(applyLogicalCodeGen(ast));
        break;
    }
    case AST_FUNCTION: {
        ASTNodeCodeGenScalarTicket t(*this, true);
        result = functionCallCodeGen(ast);
        break;
    }
    case AST_POWER:                 // '^' sbml considers this an operator,
                                    // left and right child nodes are the first
                                    // 2 child nodes for args.
    case AST_FUNCTION_ABS:
    case AST_FUNCTION_ARCCOS:
    case AST_FUNCTION_ARCCOSH:
    case AST_FUNCTION_ARCCOT:
    case AST_FUNCTION_ARCCOTH:
    case AST_FUNCTION_ARCCSC:
    case AST_FUNCTION_ARCCSCH:
    case AST_FUNCTION_ARCSEC:
    case AST_FUNCTION_ARCSECH:
    case AST_FUNCTION_ARCSIN:
    case AST_FUNCTION_ARCSINH:
    case AST_FUNCTION_ARCTAN:
    case AST_FUNCTION_ARCTANH:
    case AST_FUNCTION_CEILING:
    case AST_FUNCTION_COS:
    case AST_FUNCTION_COSH:
    case AST_FUNCTION_COT:
    case AST_FUNCTION_COTH:
    case AST_FUNCTION_CSC:
    case AST_FUNCTION_CSCH:
    case AST_FUNCTION_EXP:
    case AST_FUNCTION_FACTORIAL:
    case AST_FUNCTION_FLOOR:
    case AST_FUNCTION_LN:
    case AST_FUNCTION_LOG:
    case AST_FUNCTION_POWER:
    case AST_FUNCTION_ROOT:
    case AST_FUNCTION_SEC:
    case AST_FUNCTION_SECH:
    case AST_FUNCTION_SIN:
    case AST_FUNCTION_SINH:
    case AST_FUNCTION_TAN:
    case AST_FUNCTION_TANH:
    case AST_FUNCTION_QUOTIENT:
    case AST_FUNCTION_REM:{
        ASTNodeCodeGenScalarTicket t(*this, true);
        result = intrinsicCallCodeGen(ast);
        break;
    }
    case AST_FUNCTION_MAX:
    case AST_FUNCTION_MIN: {
        result = minmaxCodeGen(ast);
        break;
    }
    case AST_FUNCTION_PIECEWISE:
        result = piecewiseCodeGen(ast);
        break;
    case AST_CONSTANT_E:
        result = ConstantFP::get(builder.getContext(), APFloat(M_E));
        break;
    case AST_CONSTANT_FALSE:
        result = ConstantInt::getFalse(builder.getContext());
        break;
    case AST_CONSTANT_PI:
        result = ConstantFP::get(builder.getContext(), APFloat(M_PI));
        break;
    case AST_CONSTANT_TRUE:
        result = ConstantInt::getTrue(builder.getContext());
        break;
    case AST_FUNCTION_DELAY:
        result = delayExprCodeGen(ast);
        break;

    case AST_LAMBDA:
        result = notImplemented(ast);
        break;

    default:
        {
            stringstream msg;
            msg << "Unknown ASTNode type of " << ast->getType() << ", from " <<
                    ast->getParentSBMLObject()->toSBML();
            throw_llvm_exception(msg.str());
        }
        break;
    }
    return result;
}


llvm::Value* ASTNodeCodeGen::notImplemented(const libsbml::ASTNode* ast)
{
    char* formula = SBML_formulaToString(ast);
    string str = formula;
    free(formula);

    throw_llvm_exception("AST type not implemented yet: " + str);

    return 0;
}

llvm::Value* ASTNodeCodeGen::delayExprCodeGen(const libsbml::ASTNode* ast)
{
    if (ast->getNumChildren() == 0) {
        throw_llvm_exception("AST type 'delay' requires two children.");
    }

    char* formula = SBML_formulaToString(ast);
    string str = formula;
    free(formula);

    Log(Logger::LOG_WARNING)
      << "Unable to handle SBML csymbol 'delay'. Delay ignored in expression '"
      << str << "'.";

    return codeGen(ast->getChild(0));
}

llvm::Value* ASTNodeCodeGen::nameExprCodeGen(const libsbml::ASTNode* ast)
{
    switch(ast->getType())
    {
    case AST_NAME:
        return resolver.loadSymbolValue(ast->getName());
    case AST_NAME_AVOGADRO:
        // TODO: correct this for different units
        return ConstantFP::get(builder.getContext(), APFloat(6.02214179e23));
    case AST_NAME_TIME:
        return resolver.loadSymbolValue(SBML_TIME_SYMBOL);
    default:
        throw_llvm_exception(string(ast->getName()) +
                " is not a valid name name");
        break;
    }
    return 0;
}

llvm::Value* ASTNodeCodeGen::realExprCodeGen(const libsbml::ASTNode* ast)
{
    return ConstantFP::get(builder.getContext(), APFloat(ast->getReal()));
}

llvm::Value* ASTNodeCodeGen::integerCodeGen(const libsbml::ASTNode* ast)
{
    return ConstantFP::get(builder.getContext(), APFloat((double)ast->getInteger()));
}

llvm::Value* ASTNodeCodeGen::applyArithmeticCodeGen(
        const libsbml::ASTNode* ast)
{
    const int numChildren = ast->getNumChildren();
    const ASTNodeType_t type = ast->getType();
    Value *acc = 0;
    int start = 0;

    if (numChildren < 1)
    {
        //'plus' and 'times' both might have zero children.  This is legal MathML!
        if (type == AST_PLUS) {
            ASTNode zero(AST_INTEGER);
            zero.setValue(0);
            acc = integerCodeGen(&zero);
            return acc;
        }
        if (type==AST_TIMES) {
            ASTNode one(AST_INTEGER);
            one.setValue(1);
            acc = integerCodeGen(&one);
            return acc;
        }

        stringstream err;

        libsbml::SBase *parent = ast->getParentSBMLObject();
        char *sbml = parent ? parent->toSBML() : 0;
        err << "MathML apply node from " << (sbml ? sbml : "no parent sbml")
                << " must have at least one child node.";
        delete sbml;
        throw_llvm_exception(err.str());
    }
    else if (numChildren == 1 && type == AST_MINUS)
    {
        // treat it as a unary operator
        acc = ConstantFP::get(builder.getContext(), APFloat(0.0));
    }
    else
    {
        // start at the head of the list and evaluate each subsequent term.
        // accumulator
        start = 1;
        acc = toDouble(codeGen(ast->getChild(0)));
    }

    assert(acc);

    for (int i = start; i < numChildren; ++i)
    {
        Value *rhs = toDouble(codeGen(ast->getChild(i)));
        switch (type)
        {
        case AST_PLUS:
            acc = builder.CreateFAdd(acc, rhs, "addtmp");
            break;
        case AST_MINUS:
            acc = builder.CreateFSub(acc, rhs, "subtmp");
            break;
        case AST_TIMES:
            acc = builder.CreateFMul(acc, rhs, "multmp");
            break;
        case AST_DIVIDE:
            acc = builder.CreateFDiv(acc, rhs, "divtmp");
            break;
        default:
            break;
        }
    }
    return acc;
}

llvm::Value* ASTNodeCodeGen::applyBinaryRelationalCodeGen(const libsbml::ASTNode* ast,
		Value* left, Value* right)
{
	Value *result = 0;
	switch (ast->getType())
	{
	case AST_RELATIONAL_EQ:
		result = builder.CreateFCmpUEQ(left, right);
		break;
	case AST_RELATIONAL_GEQ:
		result = builder.CreateFCmpUGE(left, right);
		break;
	case AST_RELATIONAL_GT:
		result = builder.CreateFCmpUGT(left, right);
		break;
	case AST_RELATIONAL_LEQ:
		result = builder.CreateFCmpULE(left, right);
		break;
	case AST_RELATIONAL_LT:
		result = builder.CreateFCmpULT(left, right);
		break;
	case AST_RELATIONAL_NEQ:
		result = builder.CreateFCmpUNE(left, right);
		break;
	default:
		result = 0;
		break;
	}

    assert(result);

    return result;
}

llvm::Value* ASTNodeCodeGen::applyRelationalCodeGen(const libsbml::ASTNode* ast) {
	return applyScalarRelationalCodeGen(ast);
}

llvm::Value* ASTNodeCodeGen::applyScalarRelationalCodeGen(const libsbml::ASTNode* ast)
{
    if (!rr::Config::getBool(rr::Config::LOADSBMLOPTIONS_PERMISSIVE)) {
		Value* left = toDouble(codeGen(ast->getLeftChild()));
		Value* right = toDouble(codeGen(ast->getRightChild()));
        return applyBinaryRelationalCodeGen(ast, left, right);
    }
	cout << "Doing scalar relation" << endl;

	Value *result = 0;
	// Possible to have more than 2 children
	// In MathML, if there are >2 children, operation is applied between them
	// e.g. apply LEQ 1,2,1 -> 1<=2<=1
	unsigned int numKids = ast->getNumChildren();
	if (numKids == 2) {
		Value *left = toDouble(codeGen(ast->getLeftChild()));
		Value *right = toDouble(codeGen(ast->getRightChild()));
		
		cout << "left is a constant: " << isa<Constant>(left) << endl;
		cout << "left is an operator: " << isa<Operator>(left) << endl;
		cout << "left: " << endl;
		left->dump();

		cout << "right is a constant: " <<  isa<Constant>(right) << endl;
		cout << "right is an operator: " << isa<Operator>(right) << endl;
		cout << "right: " << endl;
		right->dump();

		result = applyBinaryRelationalCodeGen(ast, left, right);

		cout << "res is a constant: " << isa<Constant>(result) << endl;
		cout << "res is an operator: " << isa<Operator>(result) << endl;
		cout << "res: " << endl;
		result->dump();
	} else {
		/* There are multiple children, we can break the relations up by ANDing them together
		 * e.g. 1 <= 2 <= 1 === (1 <= 2) && (2 <= 1). The AST tree for *LLVM* will end up like
		 *		  /&&\
		 *    /&&\  1<3
		 * 1<2   2<1
		 */

		// Get the base two relations (minimum that must be present if children > 2)
		Value* baseLeftVal = toDouble(codeGen(ast->getChild(0)));
		Value* baseMidVal = toDouble(codeGen(ast->getChild(1)));
		Value* baseRightVal = toDouble(codeGen(ast->getChild(2)));
		cout << "baseLeft: " << endl;
		baseLeftVal->dump();
		cout << "baseMid: " << endl;
		baseMidVal->dump();
		cout << "baseRight: " << endl;
		baseRightVal->dump();
		
		Value* relationLeft = applyBinaryRelationalCodeGen(ast, baseLeftVal, baseMidVal);
		Value* relationRight = applyBinaryRelationalCodeGen(ast, baseMidVal, baseRightVal);
		
		cout << "relLeft is a constant: " << isa<Constant>(relationLeft);
		cout << "relLeft is an operator: " << isa<Operator>(relationLeft);
		cout << "relLeft: " << endl;
		relationLeft->dump();
		cout << "relRight is a constant: " << isa<Constant>(relationRight);
		cout << "relRight is an operator: " << isa<Operator>(relationRight);
		cout << "relRight: " << endl;
		relationRight->dump();
		
		result = builder.CreateAnd(relationLeft, relationRight);
		
		cout << "res: " << endl;
		result->dump();

		//loop over the children to create the AND'ing tree shown above
		for (int i = 3; i < numKids; i++) {
			// In the example above, tempPrevVal = 1, tempCurrVal = 3, tempRelation = 1<3
			Value* tempPrevVal = toDouble(codeGen(ast->getChild(i - 1)));
			Value* tempCurrVal = toDouble(codeGen(ast->getChild(i)));
			Value* tempRelation = applyBinaryRelationalCodeGen(ast, tempPrevVal, tempCurrVal);
			// Create new root of the tree
			result = builder.CreateAnd(result, tempRelation);
			
			cout << "res: " << endl;
			result->dump();
		}
	}

    assert(result);
	cout << "Passed assert" << endl;

    return result;
}

llvm::Value* ASTNodeCodeGen::applyLogicalCodeGen(const libsbml::ASTNode* ast)
{
    const ASTNodeType_t type = ast->getType();
    Value* acc = NULL;
    if (type == AST_LOGICAL_NOT)
    {
        if (ast->getNumChildren() != 1)
        {
            string msg = "logic not can only have a single argument, recieved ";
            msg += toString(ast->getNumChildren());
            msg += ", MathML node: ";
            msg += to_string(ast);
            throw_llvm_exception(msg);
        }
        Value *bval = toBoolean(codeGen(ast->getChild(0)));
        return builder.CreateNot(bval);
    }

    const int numChildren = ast->getNumChildren();

    if (type == AST_LOGICAL_IMPLIES)
    {
        if (numChildren != 2)
        {
            string msg = "logic implication can only have two arguments, recieved ";
            msg += toString(ast->getNumChildren());
            msg += ", MathML node: ";
            msg += to_string(ast);
            throw_llvm_exception(msg);
        }

        Value *atd = toBoolean(codeGen(ast->getChild(0)));
        Value *csq = toBoolean(codeGen(ast->getChild(1)));
        Value *natd = NULL;

        natd = builder.CreateNot(atd, "neg_tmp");

        return builder.CreateOr(natd, csq, "or_tmp");
    }

    if (numChildren == 0)
    {
        if (type == AST_LOGICAL_AND)
        {
            ASTNode t(AST_CONSTANT_TRUE);
            acc = toBoolean(codeGen(&t));
            return acc;
        }
        if (type == AST_LOGICAL_OR ||
            type == AST_LOGICAL_XOR)
        {
            ASTNode f(AST_CONSTANT_FALSE);
            acc = toBoolean(codeGen(&f));
            return acc;
        }
    }

    // start at the head of the list and evaluate each subsequent term.
    // accumulator
    acc = toBoolean(codeGen(ast->getChild(0)));

    for (int i = 1; i < numChildren; ++i)
    {
        Value *rhs = toBoolean(codeGen(ast->getChild(i)));
        switch (type)
        {
        case AST_LOGICAL_AND:
            acc = builder.CreateAnd(acc, rhs, "and_tmp");
            break;
        case AST_LOGICAL_OR:
            acc = builder.CreateOr(acc, rhs, "or_tmp");
            break;
        case AST_LOGICAL_XOR:
            acc = builder.CreateXor(acc, rhs, "xor_tmp");
            break;
        default:
            assert(0 && "invalid node type for logical");
            break;
        }
    }
    return acc;
}

llvm::Value* ASTNodeCodeGen::functionCallCodeGen(const libsbml::ASTNode* ast)
{
    const uint nargs = ast->getNumChildren();

    Value** args = (Value**) alloca(nargs*sizeof(Value*));

    for (uint i = 0; i < nargs; ++i)
    {
        const ASTNode *c = ast->getChild(i);
        args[i] = toDouble(codeGen(c));
    }

    Log(Logger::LOG_TRACE) << "ASTNodeCodeGen::functionCallCodeGen, name: "
            << ast->getName() << ", numChild: " << nargs;

    return resolver.loadSymbolValue(ast->getName(), ArrayRef<Value*>(args, nargs));
}

llvm::Value* ASTNodeCodeGen::intrinsicCallCodeGen(const libsbml::ASTNode *ast)
{
    LibFunc::Func funcId;
    TargetLibraryInfo targetLib;
    Function* func;
    Module *module = getModule();

    switch (ast->getType())
    {
    case AST_FUNCTION_POWER:
    case AST_POWER:                 // '^' sbml considers this an operator,
                                    // left and right child nodes are the first
                                    // 2 child nodes for args.
        funcId = LibFunc::pow;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_ABS:
        funcId = LibFunc::fabs;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_ARCCOS:
        funcId = LibFunc::acos;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_ARCCOSH:
        func = module->getFunction("arccosh");
        break;
    case AST_FUNCTION_ARCCOT:
        // lame hack, need to check for negative zero to pass test...
        if (isNegative(ast))
        {
            func = module->getFunction("rr_arccot_negzero");
        }
        else
        {
            func = module->getFunction("arccot");
        }
        break;
    case AST_FUNCTION_ARCCOTH:
        func = module->getFunction("arccoth");
        break;
    case AST_FUNCTION_ARCCSC:
        func = module->getFunction("arccsc");
        break;
    case AST_FUNCTION_ARCCSCH:
        func = module->getFunction("arccsch");
        break;
    case AST_FUNCTION_ARCSEC:
        func = module->getFunction("arcsec");
        break;
    case AST_FUNCTION_ARCSECH:
        func = module->getFunction("arcsech");
        break;
    case AST_FUNCTION_ARCSIN:
        funcId = LibFunc::asin;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_ARCSINH:
        func = module->getFunction("arcsinh");
        break;
    case AST_FUNCTION_ARCTAN:
        funcId = LibFunc::atan;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_ARCTANH:
        func = module->getFunction("arctanh");
        break;
    case AST_FUNCTION_CEILING:
        funcId = LibFunc::ceil;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_COS:
        funcId = LibFunc::cos;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_COSH:
        funcId = LibFunc::cosh;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_COT:
        func = module->getFunction("cot");
        break;
    case AST_FUNCTION_COTH:
        func = module->getFunction("coth");
        break;
    case AST_FUNCTION_CSC:
        func = module->getFunction("csc");
        break;
    case AST_FUNCTION_CSCH:
        func = module->getFunction("csch");
        break;
    case AST_FUNCTION_EXP:
        funcId = LibFunc::exp;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_FACTORIAL:
        func = module->getFunction("rr_factoriald");
        break;
    case AST_FUNCTION_FLOOR:
        funcId = LibFunc::floor;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_LN:
        funcId = LibFunc::log;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_LOG:
        func = module->getFunction("rr_logd");
        break;
    case AST_FUNCTION_ROOT:
        func = module->getFunction("rr_rootd");
        break;
    case AST_FUNCTION_SEC:
        func = module->getFunction("sec");
        break;
    case AST_FUNCTION_SECH:
        func = module->getFunction("sech");
        break;
    case AST_FUNCTION_SIN:
        funcId = LibFunc::sin;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_SINH:
        funcId = LibFunc::sinh;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_TAN:
        funcId = LibFunc::tan;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_TANH:
        funcId = LibFunc::tanh;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    case AST_FUNCTION_QUOTIENT:
        func = module->getFunction("quotient");
        break;
    case AST_FUNCTION_REM:
        funcId = LibFunc::fmod;
        func = module->getFunction(targetLib.getName(funcId));
        break;
    default:
    {
        string msg = "unknown intrinsic function ";
        msg += ast->getName();
        throw_llvm_exception(msg);
    }
    break;
    }

    // get the function
    if (func == 0)
    {
        string msg = "could not obtain a function for intrinsic " +
                string(ast->getName());
        msg += ", your operating system might not supoort it.";
        throw_llvm_exception(msg);
    }

    // check if the arg counts match
    if (func->arg_size() != ast->getNumChildren())
    {
        stringstream err;
        err << "function call argument count in "
                << ast->getParentSBMLObject()->toSBML()
                << " does not match the specfied number of arguments, "
                << (string) func->getName() << " requires " << func->arg_size()
                << " args, but was given " << ast->getNumChildren();
        throw_llvm_exception(err.str());
    }

    std::vector<Value*> args;
    for (unsigned i = 0; i < ast->getNumChildren(); ++i)
    {
        args.push_back(toDouble(codeGen(ast->getChild(i))));
    }

    return builder.CreateCall(func, args, "calltmp");

    assert(0 && "should not get here");
    return 0;

    /*
     Intrinsic::ID id = Intrinsic::not_intrinsic;

     switch (ast->getType())
     {
     case AST_FUNCTION_ABS:
     id = Intrinsic::fabs;
     break;
     case AST_FUNCTION_CEILING:
     id = Intrinsic::ceil;
     break;
     case AST_FUNCTION_COS:
     id = Intrinsic::cos;
     break;
     case AST_FUNCTION_COSH:
     id = Intrinsic::cos
     case AST_FUNCTION_COT:
     case AST_FUNCTION_COTH:
     case AST_FUNCTION_CSC:
     case AST_FUNCTION_CSCH:
     case AST_FUNCTION_EXP:
     case AST_FUNCTION_FACTORIAL:
     case AST_FUNCTION_FLOOR:
     case AST_FUNCTION_LN:
     case AST_FUNCTION_LOG:
     case AST_FUNCTION_POWER:
     case AST_FUNCTION_ROOT:
     case AST_FUNCTION_SEC:
     case AST_FUNCTION_SECH:
     case AST_FUNCTION_SIN:
     case AST_FUNCTION_SINH:
     case AST_FUNCTION_TAN:
     case AST_FUNCTION_TANH:

     case AST_FUNCTION_ARCCOS:

     case AST_FUNCTION_ARCCOSH:
     case AST_FUNCTION_ARCCOT:
     case AST_FUNCTION_ARCCOTH:
     case AST_FUNCTION_ARCCSC:
     case AST_FUNCTION_ARCCSCH:
     case AST_FUNCTION_ARCSEC:
     case AST_FUNCTION_ARCSECH:
     case AST_FUNCTION_ARCSIN:
     case AST_FUNCTION_ARCSINH:
     case AST_FUNCTION_ARCTAN:
     case AST_FUNCTION_ARCTANH:

     }
     */

}

llvm::Value* ASTNodeCodeGen::toBoolean(llvm::Value* value)
{
    Type *type = value->getType();

    if (type->isIntegerTy(1))
    {
        return value;
    }
    else if (type->isIntegerTy())
    {
        return builder.CreateICmpNE(value,
                ConstantInt::get(builder.getContext(),
                        APInt(type->getIntegerBitWidth(), 0)), "ne_zero");
    }
    else if (type->isDoubleTy())
    {
        return builder.CreateFCmpONE(value,
                ConstantFP::get(builder.getContext(), APFloat(0.0)), "ne_zero");
    }

    throw_llvm_exception("unsupported type conversion to boolean");
    return 0;
}

llvm::Value* ASTNodeCodeGen::toDouble(llvm::Value* value)
{
    Type *type = value->getType();

    if (type->isDoubleTy()) {
        return value;
    }

    if (type->isIntegerTy()) {
        return builder.CreateUIToFP(value,
                Type::getDoubleTy(builder.getContext()), "double_tmp");
    }

    throw_llvm_exception("unsupported type convertion to double");
    return 0;
}

llvm::Module* ASTNodeCodeGen::getModule()
{
    BasicBlock *bb = 0;
    if((bb = builder.GetInsertBlock()) != 0)
    {
        Function *function = bb->getParent();
        if(function)
        {
            Module *module = function->getParent();
            assert(module && "could not get module from function");
            return module;
        }
    }
    throw_llvm_exception("could not get module, a BasicBlock is not currently being populated.");
    return 0;
}

llvm::Value* ASTNodeCodeGen::piecewiseCodeGen(const libsbml::ASTNode* ast)
{
    // the 'otherwise' is an optional MathML element.
    // the child nodes contain the piece elements in pairs, so
    // [value0, condition0, value1, condition1, ... {otherwise value}]
    // even number of child nodes means we have no otherwise node.
    //
    // each piece coresponds to the 'then' part of a conditions, and
    // the next piece is the 'else' part.  This is terminated with the
    // otherwise block.

    LLVMContext &context = builder.getContext();

    Function *func = builder.GetInsertBlock()->getParent();

    BasicBlock *mergeBB = BasicBlock::Create(context, "merge");

    vector<Value*> values;
    vector<BasicBlock*> blocks;

    const uint nchild = ast->getNumChildren();
    uint i = 0;

    // cond comes after val, but we need to get cond
    // before evaluating val.
    while ((i + 1) < nchild)
    {
        // added to end of function
        BasicBlock *thenBB = BasicBlock::Create(context,
                "then_" + toString(i), func);

        BasicBlock *elseBB = BasicBlock::Create(context, "else_" + toString(i));

        // the value
        const ASTNode *thenNode = ast->getChild(i++);
        // the conditional
        const ASTNode *condNode = ast->getChild(i++);

        resolver.pushCacheBlock();
        Value *cond = toBoolean(codeGen(condNode));
        resolver.popCacheBlock();

        builder.CreateCondBr(cond, thenBB, elseBB);

        // the then value
        builder.SetInsertPoint(thenBB);

        // turn on scalar mode since this is a scalar expression
        ASTNodeCodeGenScalarTicket t(*this, true);

        resolver.pushCacheBlock();
        Value *thenVal = toDouble(codeGen(thenNode));
        resolver.popCacheBlock();

        values.push_back(thenVal);

        builder.CreateBr(mergeBB);

        // codegen of the 'then' block changes current block, update
        thenBB = builder.GetInsertBlock();
        blocks.push_back(thenBB);

        // the else block
        func->getBasicBlockList().push_back(elseBB);
        builder.SetInsertPoint(elseBB);
    }

    // final else value
    Value *owVal = 0;

    if (i < nchild)
    {
        // we have an otherwise block
        assert((i+1) == nchild);

        // turn on scalar mode since this is a scalar expression
        ASTNodeCodeGenScalarTicket t(*this, true);

        const ASTNode *ow = ast->getChild(i);
        resolver.pushCacheBlock();
        owVal = toDouble(codeGen(ow));
        resolver.popCacheBlock();
    }
    else
    {
        Log(Logger::LOG_WARNING) << "No \"otherwise\" element in MathML "
                "piecewise, returning NaN as \"otherwise\" value";

        owVal = ConstantFP::get(builder.getContext(),
                APFloat::getQNaN(APFloat::IEEEdouble));
    }

    builder.CreateBr(mergeBB);

    // otherwise codegen changes current block, update for PHI
    BasicBlock *owBlock = builder.GetInsertBlock();

    values.push_back(owVal);
    blocks.push_back(owBlock);

    // emit the merge block
    func->getBasicBlockList().push_back(mergeBB);
    builder.SetInsertPoint(mergeBB);

    assert(values.size() == blocks.size());

    PHINode *pn = builder.CreatePHI(Type::getDoubleTy(context), values.size(),
            "iftmp");

    for (uint i = 0; i < values.size(); ++i)
    {
        pn->addIncoming(values[i], blocks[i]);
    }

    return pn;
}

llvm::Value* ASTNodeCodeGen::minmaxCodeGen(const libsbml::ASTNode* ast) {

    const ASTNodeType_t type = ast->getType();

    LLVMContext &context = builder.getContext();

    Module *module = getModule();

    const uint nchild = ast->getNumChildren();
    
    if (type == AST_FUNCTION_MAX) {
        llvm::Function* func = module->getFunction("rr_max");

        // If there's no child
        if (nchild == 0) {
            return ConstantFP::get(builder.getContext(), APFloat(-std::numeric_limits<double>::infinity()));
        }

        // if nchild is at least 1
        Value* value = toDouble(codeGen(ast->getChild(0)));

        uint i = 1;
        while (i < nchild) {
            std::vector<Value*> args;
            args.push_back(value);
            args.push_back(toDouble(codeGen(ast->getChild(i))));
            value = builder.CreateCall(func, args, "rr_max");
            i++;
        }
        assert(value);

        return value;
    }
    else {
        llvm::Function* func = module->getFunction("rr_min");

        // If there's no child
        if (nchild == 0) {
            return ConstantFP::get(builder.getContext(), APFloat(std::numeric_limits<double>::infinity()));
        }

        // if nchild is at least 1
        Value* value = toDouble(codeGen(ast->getChild(0)));

        uint i = 1;
        while (i < nchild) {
            std::vector<Value*> args;
            args.push_back(value);
            args.push_back(toDouble(codeGen(ast->getChild(i))));
            value = builder.CreateCall(func, args, "rr_min");
            i++;
        }
        assert(value);

        return value;
    }
}

static bool isNegative(const libsbml::ASTNode *ast)
{
    if (ast->getNumChildren() > 0)
    {
        const ASTNode* m = ast->getChild(0);
        if (m->getType() == AST_MINUS && m->getNumChildren() > 0)
        {
            return true;
        }
    }
    return false;
}

ASTNodeCodeGenScalarTicket::ASTNodeCodeGenScalarTicket(ASTNodeCodeGen& gen, bool val, std::string n)
    : z_(gen), v_(gen.scalar_mode_), n_(n) {
    z_.scalar_mode_ = val;
}

ASTNodeCodeGenScalarTicket::~ASTNodeCodeGenScalarTicket() {
    z_.scalar_mode_ = v_;
}


} /* namespace rr */


