#pragma hdrstop
#include "ASTPreProcessor.h"
#include "rrOSSpecifics.h"
#include "rrLogger.h"
#include "rrStringUtils.h"
#include "rrConfig.h"

#include <sbml/math/ASTNode.h>
#include <sbml/math/FormulaFormatter.h>
#include <sbml/SBase.h>
#ifdef LIBSBML_HAS_PACKAGE_ARRAYS
#include <sbml/packages/arrays/common/ArraysExtensionTypes.h>
#endif
#include <Poco/Logger.h>
#include <cmath>

using namespace libsbml;
using namespace std;
using namespace rr;

namespace rr
{
	ASTPreProcessor::ASTPreProcessor()
	{
	}

	ASTPreProcessor::~ASTPreProcessor()
	{
	}

	ASTNode* ASTPreProcessor::preProcess(libsbml::ASTNode* ast, valMap *values)
	{
		ASTNode* result;
		if (ast == 0)
		{
			throw("ASTNode is NULL");
		}

		switch (ast->getType())
		{
		case AST_PLUS:
		case AST_MINUS:
		case AST_TIMES:
		case AST_DIVIDE:
		{
			result = applyArithmetic(ast, values);
			break;
		}
		// No need to do anything here
		case AST_INTEGER:
		case AST_REAL:
		case AST_REAL_E:
		case AST_RATIONAL:
		{
			result = ast;
			break;
		}
		case AST_NAME:
		{
			if (values->find(ast->getName()) != values->end())
			{
				result = new ASTNode(AST_INTEGER);
				result->setValue((int)((*values)[ast->getName()]));
			}
			break;
		}
		/*case AST_RELATIONAL_EQ:
		case AST_RELATIONAL_GEQ:
		case AST_RELATIONAL_GT:
		case AST_RELATIONAL_LEQ:
		case AST_RELATIONAL_LT:
		case AST_RELATIONAL_NEQ:
		{
			result = applyRelational(ast);
			break;
		}
		case AST_LOGICAL_AND:
		case AST_LOGICAL_NOT:
		case AST_LOGICAL_OR:
		case AST_LOGICAL_XOR:
		{
			result = applyLogical(ast);
			break;
		}
		case AST_FUNCTION:
		{
			result = functionCall(ast);
			break;
		}
		case AST_POWER:
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
		{
			result = intrinsicCall(ast);
			break;
		}
		case AST_FUNCTION_PIECEWISE:
			result = piecewise(ast);
			break;
		case AST_CONSTANT_E:
		{
			result->setType(AST_REAL);
			result->setValue(M_E);
			break;
		}
		case AST_CONSTANT_FALSE:
		{
			result = ConstantInt::getFalse(builder.getContext());
			break;
		}
		case AST_CONSTANT_PI:
		{
			result->setType(AST_REAL);
			result->setValue(M_PI);
			break;
		}
		case AST_CONSTANT_TRUE:
		{
			result = ConstantInt::getTrue(builder.getContext());
			break;
		}
		case AST_FUNCTION_DELAY:
			result = delayExpr(ast);
			break;
		case AST_LAMBDA:
			result = notImplemented(ast);
			break;*/
		case AST_ORIGINATES_IN_PACKAGE:
		{
			if (ast->getExtendedType() == AST_LINEAR_ALGEBRA_SELECTOR)
				result = selector(NULL, ast, values);
			break;
		}
		default:
		{
			stringstream msg;
			msg << "Unknown ASTNode type of " << ast->getType() << ", from " << ast->getParentSBMLObject()->toSBML();
			throw(msg.str());
		}
		break;
		}
		return result;
	}

	ASTNode *ASTPreProcessor::applyArithmetic(ASTNode *ast, valMap *values)
	{
		int numChildren = ast->getNumChildren();
		ASTNodeType_t type = ast->getType();
		ASTNode* result;

		if (numChildren == 0)
		{
			//'plus' and 'times' both might have zero children.  This is legal MathML!
			if (type == AST_PLUS)
			{
				ASTNode zero(AST_INTEGER);
				zero.setValue(0);
				free(ast);
				return &zero;
			}
			if (type == AST_TIMES)
			{
				ASTNode one(AST_INTEGER);
				one.setValue(1);
				free(ast);
				return &one;
			}

			stringstream err;

			libsbml::SBase *parent = ast->getParentSBMLObject();
			char *sbml = parent ? parent->toSBML() : 0;
			err << "MathML apply node from " << (sbml ? sbml : "no parent sbml")
				<< " must have at least one child node.";
			delete sbml;
			throw(err.str());
		}
		else
		{
			result = preProcess(ast->getChild(0), values);
			for (int i = 1; i < numChildren; ++i)
			{
				ASTNode *rhs = preProcess(ast->getChild(i), values);
				if (rhs->getNumChildren() == 0 && result->getNumChildren() == 0)
				{
					switch (type)
					{
					case AST_PLUS:
						result->setValue(result->getValue() + rhs->getValue());
						break;
					case AST_MINUS:
						result->setValue(result->getValue() - rhs->getValue());
						break;
					case AST_TIMES:
						result->setValue(result->getValue() * rhs->getValue());
						break;
					case AST_DIVIDE:
						result->setValue(result->getValue() / rhs->getValue());
						break;
					default:
						break;
					}
				}
				else
				{
					throw runtime_error("Please use simplified expressions for calculating dimensions\n");
				}
			}
		}
		return result;
	}

	ASTNode* ASTPreProcessor::selector(ASTNode* parent, ASTNode *ast, valMap *values)
	{
		/**
		 * There are 2 ways to represent a selector a[x][y] and a[x, y]
		 * They have the following AST representations
		 *
		 *       Selector                    Selector
		 *        /    \                     /   |   \
		 *    Selector  y                   a    x    y
		 *     /    \
		 *    a      x
		 *
		*/
		// If the current AST is another selector not related to the parent
		// AST_LINEAR_ALGEBRA_SELECTOR, we have to call preProcess() on it and get its value
		if (parent && parent->getExtendedType() != AST_LINEAR_ALGEBRA_SELECTOR)
		{
			ASTNode* result = preProcess(ast, values);
			return result;
		}
		else
		{
			uint numChildren = ast->getNumChildren();
			string res = "";
			for (uint i = 0; i < numChildren; i++)
			{
				libsbml::ASTNode *child = ast->getChild(i);
				if (child->getType() == AST_ORIGINATES_IN_PACKAGE && child->getExtendedType() == AST_LINEAR_ALGEBRA_SELECTOR)
				{
					// Should return a string
					res += "-";
					res += selector(ast, child, values)->getName();
				}
				else if (child->getType() == AST_NAME && i == 0 && (!parent || (parent && parent->getExtendedType() == AST_LINEAR_ALGEBRA_SELECTOR)))
				{
					// This is our actual variable from which we have to select
					res += child->getName();
				}
				else
				{
					res += "-";
					double val = preProcess(child, values)->getValue();
					double iptr;
					if (modf(val, &iptr) != 0.0 || iptr < 0.0)
					{
						throw invalid_argument("Dimension of parameter is not a positive integer");
					}
					res += to_string((uint)iptr);
				}
			}
			ast->~ASTNode();
			ASTNode* result = new ASTNode(AST_NAME);
			result->setName(res.c_str());
			return result;
		}
	}
}