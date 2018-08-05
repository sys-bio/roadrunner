/*
 * LLVMModelSymbols.cpp
 *
 *  Created on: Jun 29, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "LLVMModelSymbols.h"
#include "LLVMException.h"
#include "ASTNodeCodeGen.h"
#include "rrLogger.h"
#include "rrStringUtils.h"
#include "rrException.h"

#include "conservation/ConservationExtension.h"

#include <sbml/math/ASTNode.h>
#include <sbml/math/FormulaFormatter.h>
#include <sbml/SBMLDocument.h>
#ifdef LIBSBML_HAS_PACKAGE_ARRAYS
#include <sbml/packages/arrays/common/ArraysExtensionTypes.h>
#include <ASTPreProcessor.h>
#endif

using namespace libsbml;
using namespace llvm;
using namespace std;

using rr::Logger;
using rr::getLogger;


namespace rrllvm
{

LLVMModelSymbols::LLVMModelSymbols(const libsbml::Model *m, LLVMModelDataSymbols const &sym) :
        model(m),
        symbols(sym),
        reactions(sym.getReactionSize(), ReactionSymbols())
{
    // make sure we have no events
    model->getListOfEvents()->accept(*this);

    model->getListOfSpecies()->accept(*this);
    model->getListOfCompartments()->accept(*this);
    model->getListOfReactions()->accept(*this);

    // only want to process global parameters, however calling
    // getListOfParameters()->acept(..) ends up calling us
    // with all the local paramters, and older sbml uses
    // the class Parameter for locals, so if we manually
    // itterate the list, we only get global params.
    const libsbml::ListOfParameters *params = model->getListOfParameters();
    for (uint i = 0; i < params->size(); ++i)
    {
        const Parameter* param = params->get(i);

        Log(Logger::LOG_TRACE) << "global parameter " << param->getId() <<
                " initial value: " << param->getValue();

        ASTNode *value = nodes.create(AST_REAL);

        if (param->isSetValue())
        {
            value->setValue(param->getValue());
        }
        else
        {
            // check if we have an initial assignment for this param.
            const string& id = param->getId();
            if (model->getInitialAssignment(id) == NULL &&
                    model->getAssignmentRule(id) == NULL)
            {
                Log(Logger::LOG_WARNING) << "Global parameter, \'"
                        << param->getId() << "\' missing value and missing init "
                        "assignment and assignment rule!, defaulting value to 0.0.";
                Log(Logger::LOG_WARNING) << "This probably is NOT what you want "
                        "with global parameter \'" << param->getId() << "\'.";
                value->setValue(0.0);
                std::stringstream ss;
                ss << "Global parameter '" << param->getId() << "' missing value and missing init assignment and assignment rule!";
                rr::UninitializedValue(ss.str());
            }
            else
            {
                Log(Logger::LOG_INFORMATION) << "parameter " << param->getId()
                        << " missing value, but has init rule or rule, setting "
                        " value to 0.0";
                value->setValue(0.0);
            }
        }

		// Don't have to check for errors because we would've called LLVMModelDataSymbols
		// which would have thrown an error
		if (param->getPlugin("arrays") != NULL)
		{
			const set<string> arrayedParameters = sym.getArrayedGlobalParameters(param->getId());
			for (set<string>::const_iterator it = arrayedParameters.begin(), jt = arrayedParameters.end(); it!=jt; it++)
			{
				initialValues.globalParameters[*it] = value;
			}
		}
		else
			initialValues.globalParameters[param->getId()] = value;
    }



    // initial assignments override initial values.
    model->getListOfInitialAssignments()->accept(*this);

    // assignment override initial assignment rules at run time.
    model->getListOfRules()->accept(*this);
}

LLVMModelSymbols::~LLVMModelSymbols()
{
}

bool LLVMModelSymbols::visit(const libsbml::Compartment& x)
{
	// Need to include an array of compartments Vin
    ASTNode *node = nodes.create(AST_REAL);
    if (x.isSetVolume())
    {
        node->setValue(x.getVolume());
    } else
    {
        string compid = x.getId();
        const Model* model = x.getSBMLDocument()->getModel();
        if (model->getInitialAssignment(compid) == NULL &&
            model->getAssignmentRule(compid) == NULL &&
            x.getSpatialDimensions() != 0) {
            Log(Logger::LOG_WARNING) << "volume not set for compartment '"
              << compid << "'.  Defaulting to 1.0";
        }
        //We still need to set up a fake initial value if we need one, even though it'll be overwritten or is unneeded.
        node->setValue(1.0);
    }
    initialValues.compartments[x.getId()] = node;
    return true;
}

bool LLVMModelSymbols::visit(const libsbml::Species& x)
{
    processSpecies(initialValues, &x, 0);
    return true;
}

bool LLVMModelSymbols::visit(const libsbml::AssignmentRule& x)
{
    Log(Logger::LOG_TRACE) << "processing AssignmentRule, id: " << x.getId();
    SBase *element = const_cast<Model*>(model)->getElementBySId(x.getVariable());
    
	if (element)
    {
        processElement(assigmentRules, element, x.getMath());
    }
    else
    {
        Log(Logger::LOG_ERROR) << "Could not get elment for assignment rule \""
                << const_cast<libsbml::AssignmentRule&>(x).toSBML()
                << "\", it will be ignored";
    }

    return true;
}

bool LLVMModelSymbols::visit(const libsbml::InitialAssignment& x)
{
    Log(Logger::LOG_TRACE) << "processing InitialAssignment, id: " +  x.getId();
    SBase *element = const_cast<Model*>(model)->getElementBySId(x.getSymbol());
    processElement(initialValues, element, x.getMath());
    processElement(initialAssignmentRules, element, x.getMath());
    return true;
}

bool LLVMModelSymbols::visit(const libsbml::RateRule& rule)
{
    Log(Logger::LOG_TRACE) << "processing RateRule, id: " +  rule.getId();
    SBase *element = const_cast<Model*>(model)->getElementBySId(rule.getVariable());
    processElement(rateRules, element, rule.getMath());
    return true;
}

bool LLVMModelSymbols::visit (const libsbml::Event &event)
{
    return true;
}

// Need to change this function to take in arrays Vin
// Should I process the AST of the index and the AST of LHS here by writing a 
// seperate function for getting the value of the AST
// instead of dealing with everything in LLVM? Vin
void LLVMModelSymbols::processElement(SymbolForest& currentSymbols,
        const libsbml::SBase *element, const ASTNode* math)
{
    const Compartment *comp = 0;
    const Parameter *param = 0;
    const Species *species = 0;
    const SpeciesReference *reference = 0;

    if (!element)
        throw LLVMException("LLVMModelSymbols: Unable to process element");

    if ((comp = dynamic_cast<const Compartment*>(element)))
    {
        currentSymbols.compartments[comp->getId()] = math;
    }
    else if ((param = dynamic_cast<const Parameter*>(element)))
    {
		processParameter(currentSymbols, param, math);
        // currentSymbols.globalParameters[param->getId()] = math;
    }
    else if ((species = dynamic_cast<const Species*>(element)))
    {
        processSpecies(currentSymbols, species, math);
    }
    else if ((reference = dynamic_cast<const SpeciesReference*>(element)))
    {
        currentSymbols.speciesReferences[reference->getId()] = math;
    }
    else
    {
        char* sbml = const_cast<SBase*>(element)->toSBML();
        Log(Logger::LOG_WARNING) << "Unknown element whilst processing symbols: "
                << sbml;
        free(sbml);
    }
}

bool LLVMModelSymbols::visit(const libsbml::Rule& x)
{
    Log(Logger::LOG_TRACE) << "Rule, id: " << x.getId();
    return true;
}

bool LLVMModelSymbols::visit(const libsbml::Reaction& r)
{
    const ListOfSpeciesReferences *reactants = r.getListOfReactants();
    const ListOfSpeciesReferences *products = r.getListOfProducts();

    ReactionSymbols &rs = reactions[symbols.getReactionIndex(r.getId())];

    for (int i = 0; i < reactants->size(); i++)
    {
        const SpeciesReference* reactant =
                dynamic_cast<const SpeciesReference*>(reactants->get(i));
        try
        {

            ASTNodeList &speciesNodes =
                    rs.reactants[symbols.getFloatingSpeciesIndex(reactant->getSpecies())];

            const ASTNode *stoich = getSpeciesReferenceStoichMath(reactant);

            if(reactant->isSetId() && reactant->getId().size())
            {
                initialValues.speciesReferences[reactant->getId()] = stoich;
                ASTNode *refName = nodes.create(AST_NAME);
                refName->setName(reactant->getId().c_str());
                speciesNodes.push_back(refName);
            }
            else
            {
                speciesNodes.push_back(stoich);
            }
        }
        catch (LLVMException&)
        {
            // we get here if the getFloatingSpeciesIndex throws an exception, that's OK
            // because the species is most likely a boundary species, which is OK
            // to be used as a reactant (it just won't get consumed like a floating species).
            // TODO this is normal, should not throw an exception!
        }
    }

    for (int i = 0; i < products->size(); i++)
    {
        const SpeciesReference* product =
                dynamic_cast<const SpeciesReference*>(products->get(i));
        try
        {
            ASTNodeList &speciesNodes =
                    rs.products[symbols.getFloatingSpeciesIndex(product->getSpecies())];

            const ASTNode *stoich = getSpeciesReferenceStoichMath(product);

            if(product->isSetId() && product->getId().size())
            {
                initialValues.speciesReferences[product->getId()] = stoich;
                ASTNode *refName = nodes.create(AST_NAME);
                refName->setName(product->getId().c_str());
                speciesNodes.push_back(refName);
            }
            else
            {
                speciesNodes.push_back(stoich);
            }
        }
        catch (LLVMException&)
        {
            // get here if product is not a floating species, its OK if its a
            // boundary or conserved moiety
        }
    }

    /*
     const ListOf *list = dynamic_cast<const ListOf *>(sr.getParentSBMLObject());
     const Reaction *r = dynamic_cast<const Reaction*>(list->getParentSBMLObject());

     string speciesId = sr.getSpecies();
     string reactionId = r->getId();

     cout << "species: " << sr.getSpecies() << "\n";
     cout << "reaction: " << r->getId() << "\n";
     cout << "sr.isSetStoichiometry(): " << sr.isSetStoichiometry() << "\n";
     cout << "sr.isSetStoichiometryMath(): " << sr.isSetStoichiometryMath() << "\n";
     cout << "stoichiometry: " << sr.getStoichiometry() << "\n";
     */

    return true;
}



void LLVMModelSymbols::processSpecies(SymbolForest &currentSymbols,
        const libsbml::Species *species, const ASTNode* math)
{
    // ASTNode takes ownership of children, so only allocate the ones that
    // are NOT given to an ASTNode addChild.
    Log(Logger::LOG_TRACE) << "processing species " << species->getId() << endl;

    if (!math)
    {
        if (species->getHasOnlySubstanceUnits())
        {
            // value should be an amount
            if (species->isSetInitialConcentration())
            {
                // need to convert conc to amount,
                // these two nodes are owned by the parent
                ASTNode *conc = new ASTNode(AST_REAL);
                conc->setValue(species->getInitialConcentration());
                ASTNode *comp = new ASTNode(AST_NAME);
                comp->setName(species->getCompartment().c_str());

                ASTNode *amt = nodes.create(AST_TIMES);
                amt->addChild(conc);
                amt->addChild(comp);
                math = amt;
            }
            else if (species->isSetInitialAmount())
            {
                // we got an amount, all good
                ASTNode *amt = nodes.create(AST_REAL);
                amt->setValue(species->getInitialAmount());
                math = amt;
            }
            else
            {
                string spid = species->getId();
                const Model* model = species->getSBMLDocument()->getModel();
                if (model->getInitialAssignment(spid) == NULL &&
                        model->getAssignmentRule(spid) == NULL)
                {
                    Log(Logger::LOG_WARNING) << "species '" << spid
                            << "' has neither initial amount nor concentration set. "
                            << " Setting initial amount to 0.0";
                }
                // We still need to set up a fake initial value if we need one,
                // even though it'll be overwritten:
                ASTNode *amt = nodes.create(AST_REAL);
                amt->setValue(0.0);
                math = amt;
            }
        }
        else
        {
            // value should be a concentration
            if (species->isSetInitialConcentration())
            {
                // we got an conc, all good
                ASTNode *conc = nodes.create(AST_REAL);
                conc->setValue(species->getInitialConcentration());
                math = conc;
            }
            else if (species->isSetInitialAmount())
            {
                // need to convert amt to concentraion,
                // these two nodes are owned by the parent
                ASTNode *amt = new ASTNode(AST_REAL);
                amt->setValue(species->getInitialAmount());
                ASTNode *comp = new ASTNode(AST_NAME);
                comp->setName(species->getCompartment().c_str());

                ASTNode *conc = nodes.create(AST_DIVIDE);
                conc->addChild(amt);
                conc->addChild(comp);
                math = conc;
            }
            else
            {
                string spid = species->getId();
                const Model* model = species->getSBMLDocument()->getModel();
                if (model->getInitialAssignment(spid) == NULL &&
                        model->getAssignmentRule(spid) == NULL)
                {
                    Log(Logger::LOG_WARNING) << "species '" << spid
                            << "' has neither initial amount nor concentration set. "
                            << " Setting initial concentration to 0.0";
                }
                //We still need to set up a fake initial value if we need one, even though it'll be overwritten:
                ASTNode *conc = nodes.create(AST_REAL);
                conc->setValue(0.0);
                math = conc;
            }
        }
    }

    assert(math);

    if (species->getBoundaryCondition())
    {
        currentSymbols.boundarySpecies[species->getId()] = math;
    }
    else
    {
        currentSymbols.floatingSpecies[species->getId()] = math;
    }
}

void LLVMModelSymbols::getUnknownValues(map<string, uint> *values, const ASTNode* ast)
{
	if (ast->getType() == AST_NAME)
	{
		int result;
		LLVMModelDataSymbols::SymbolIndexType type = this->symbols.getSymbolIndex(ast->getName(), result);
		switch (type)
		{
		/*case LLVMModelDataSymbols::FLOATING_SPECIES:
			break;
		case LLVMModelDataSymbols::BOUNDARY_SPECIES:
			break;
		case LLVMModelDataSymbols::COMPARTMENT:
			break;*/
		case LLVMModelDataSymbols::GLOBAL_PARAMETER:
		{
			double val = this->model->getParameter(ast->getName())->getValue();
			double iptr;
			if (modf(val, &iptr) != 0.0 || iptr < 0.0)
			{
				Log(Logger::LOG_ERROR) << "Parameter used in dimension calculation is not a positive integer";
				throw invalid_argument("Parameter used in dimension calculation is not a positive integer");
			}
			(*values)[ast->getName()] = (uint)iptr;
			break;
		}
		/*case LLVMModelDataSymbols::REACTION:
			break;
		case LLVMModelDataSymbols::EVENT:
			break;*/
		}
		return;
	}
	for (uint i = 0; i < ast->getNumChildren(); i++)
		getUnknownValues(values, ast->getChild(i));
}

void LLVMModelSymbols::processArrayAST(SymbolForest &currentSymbols, map<string, uint> *values, vector<uint> *sizeOfDimensions, uint ind, const ASTNode *lhsMath, const ASTNode *rhsMath)
{
	if (ind == sizeOfDimensions->size())
	{
		ASTNode* lhs = lhsMath->deepCopy();
		lhs = rr::ASTPreProcessor().preProcess(lhs, values);
		ASTNode* rhs = rhsMath->deepCopy();
		rhs = rr::ASTPreProcessor().preProcess(rhs, values);
		currentSymbols.globalParameters[lhs->getName()] = rhs;
		return;
	}
	static const ArraysSBasePlugin *arraysPlug = static_cast<const ArraysSBasePlugin*>(rhsMath->getParentSBMLObject()->getPlugin("arrays"));
	const string &id = arraysPlug->getDimension(ind)->getId();
	for (uint i = 0; i < sizeOfDimensions->at(ind); i++)
	{
		(*values)[id] = i;
		processArrayAST(currentSymbols, values, sizeOfDimensions, ind + 1, lhsMath, rhsMath);
	}
}

void LLVMModelSymbols::processParameter(SymbolForest &currentSymbols, 
	const Parameter *param, const ASTNode *rhs)
{
#ifndef LIBSBML_HAS_PACKAGE_ARRAYS
	if (param->isPackageEnabled("arrays"))
	{
		throw runtime_error("The parameter " + id + "has a ListOfDimensions or ListOfIndices but the libSBML arrays package has not been enabled. Please build roadrunner with libSBML arrays package enabled");
	}
	else
	{
		currentSymbols.globalParameters[param->getId()] = rhs;
		return;
	}
#endif
	const ArraysSBasePlugin *arraysParam = static_cast<const ArraysSBasePlugin*>(param->getPlugin("arrays"));
	ArraysSBasePlugin *arraysRule = static_cast<ArraysSBasePlugin*>(rhs->getParentSBMLObject()->getPlugin("arrays"));
	uint numDimensions = arraysParam->getNumDimensions();
	if (numDimensions==0)
	{
		currentSymbols.globalParameters[param->getId()] = rhs;
		return;
	}
	vector<uint> sizeOfDimensions = this->symbols.getSizeOfDimensions(param->getId());
	map<string, uint> values;
	uint sizeOfIndices = arraysRule->getNumIndices();
	ASTNode* lhs = new ASTNode(AST_LINEAR_ALGEBRA_SELECTOR);
	ASTNode* var = new ASTNode(AST_NAME);
	var->setName(param->getId().c_str());
	lhs->addChild(var);
	for (uint i = 0; i < sizeOfIndices; i++)
	{
		lhs->addChild(arraysRule->getIndex(i)->getMath());
	}
	getUnknownValues(&values, rhs);
	getUnknownValues(&values, lhs);
	processArrayAST(currentSymbols, &values, &sizeOfDimensions, 0, lhs, rhs);
}

const ASTNode* LLVMModelSymbols::getSpeciesReferenceStoichMath(
        const libsbml::SpeciesReference* reference)
{
    const ASTNode *stoich = 0;
    if (reference->isSetStoichiometryMath()
            && reference->getStoichiometryMath()->isSetMath())
    {
        stoich = reference->getStoichiometryMath()->getMath();
    }
    else
    {
        ASTNode *m = nodes.create(AST_REAL);
        m->setValue(reference->getStoichiometry());
        stoich = m;
    }
    return stoich;
}

/*
void LLVMModelSymbols::processSpeciesReference(
        const libsbml::SpeciesReference* ref, const libsbml::Reaction* reaction,
        SpeciesReferenceType type, const ASTNode* stoich)
{
    // we might not have a floating species for this reference,
    // that should probably be an error, but there are many
    // cases in the test suite that have this, so just ignore
    // the ref in such cases.
    try
    {
        if (stoich == 0)
        {
            if (ref->isSetStoichiometryMath()
                    && ref->getStoichiometryMath()->isSetMath())
            {
                stoich = ref->getStoichiometryMath()->getMath();
            }
            else
            {
                ASTNode *m = nodes.create(AST_REAL);
                m->setValue(ref->getStoichiometry());
                stoich = m;
            }
        }

        if (ref->isSetId() && ref->getId().length() > 0)
        {
            currentSymbols->speciesReferences[ref->getId()] = stoich;
        }

        int rowIdx = symbols.getFloatingSpeciesIndex(ref->getSpecies());
        int colIdx = symbols.getReactionIndex(reaction->getId());

        ReactionSymbols &reacSym = reactions[colIdx];

        // check if we already have a species ref with this id, if so
        // replace it, if not, add it

        if (type == Reactant)
        {
            StringIntMap::iterator i;
            if (ref->isSetId() &&
                    (i = reacSym.reactantRefIds.find(ref->getId())) !=
                            reacSym.reactantRefIds.end())
            {
                reacSym.nodes[i->second] = stoich;
            }
            else
            {
                reacSym.nodes.push_back(stoich);
                int stoichIdx = reacSym.nodes.size()-1; // index of new node
                reacSym.reactantIdx[rowIdx].push_back(stoichIdx);
                if (ref->isSetId())
                {
                    reacSym.reactantRefIds[ref->getId()] = stoichIdx;
                }
            }
        }
        else
        {
            StringIntMap::iterator i;
            if (ref->isSetId() &&
                    (i = reacSym.productRefIds.find(ref->getId())) !=
                            reacSym.productRefIds.end())
            {
                reacSym.nodes[i->second] = stoich;
            }
            else
            {
                reacSym.nodes.push_back(stoich);
                int stoichIdx = reacSym.nodes.size()-1; // index of new node
                reacSym.productIdx[rowIdx].push_back(stoichIdx);
                if (ref->isSetId())
                {
                    reacSym.productRefIds[ref->getId()] = stoichIdx;
                }
            }
        }
    }
    catch (LLVMException &)
    {
        string err = "could not find product ";
        err += ref->getSpecies();
        err += " in the list of floating species for reaction ";
        err += ref->getId();
        err += ", this species will be ignored in this reaction.";
        Log(lWarning) << err;
    }
}
*/



ASTNode* LLVMModelSymbols::createStoichiometryNode(int row, int col) const
{
    // col is species id, row is reaction.
    ReactionSymbols const &r = reactions[col];
    ASTNodeList productList;
    ASTNodeList reactantList;

    IntASTNodeListMap::const_iterator pi = r.products.find(row);
    if (pi != r.products.end())
    {
        productList = pi->second;
    }

    IntASTNodeListMap::const_iterator ri = r.reactants.find(row);
    if (ri != r.reactants.end())
    {
        reactantList = ri->second;
    }

    if (productList.size() == 0 && reactantList.size() == 0)
    {
        string err = "species " + symbols.getFloatingSpeciesIds()[row] +
                " has neither products nor reactants in reaction " +
                symbols.getReactionIds()[col];
        throw LLVMException(err, __FUNC__);
    }

    // we keep track of the top level node, it takes ownership
    // of all child nodes.
    // should be exception free here on.
    ASTNode *result = new ASTNode(AST_PLUS);
    ASTNode *reactants = 0;
    ASTNode *products = 0;

    Log(Logger::LOG_TRACE) << "\t{" + rr::toString((int)row) + ", " + rr::toString((int)col) +
            "}, #reactants: " + rr::toString((int)reactantList.size()) + " #products: " +
            rr::toString((int)productList.size());

    if (reactantList.size())
    {
        // list is nearly always length 1, so don't waste
        // time making a plus out of it...
        if (reactantList.size() == 1)
        {
            reactants = new ASTNode(*reactantList.front());
        }
        else
        {
            reactants = new ASTNode(AST_PLUS);
            for (ASTNodeList::const_iterator i = reactantList.begin();
                    i != reactantList.end(); i++)
            {
                const ASTNode *reactant = *i;
                reactants->addChild(new ASTNode(*reactant));
            }
        }

        ASTNode *negOne = new ASTNode(AST_REAL);
        negOne->setValue(-1.);

        ASTNode *times = new ASTNode(AST_TIMES);
        times->addChild(negOne);
        times->addChild(reactants);

        reactants = times;
    }
    else
    {
        reactants = new ASTNode(AST_REAL);
        reactants->setValue(0.);
    }

    if (productList.size())
    {
        if (productList.size() == 1)
        {
            products = new ASTNode(*productList.front());
        }
        else
        {
            products = new ASTNode(AST_PLUS);
            for (ASTNodeList::const_iterator i = productList.begin();
                    i != productList.end(); i++)
            {
                const ASTNode *product = *i;
                products->addChild(new ASTNode(*product));
            }
        }
    }
    else
    {
        products = new ASTNode(AST_REAL);
        products->setValue(0.);
    }

    result->addChild(reactants);
    result->addChild(products);

    return result;
}

const SymbolForest& LLVMModelSymbols::getAssigmentRules() const
{
    return assigmentRules;
}

const SymbolForest& LLVMModelSymbols::getInitialValues() const
{
    return initialValues;
}

const SymbolForest& LLVMModelSymbols::getRateRules() const
{
    return rateRules;
}

const SymbolForest& LLVMModelSymbols::getInitialAssignmentRules() const
{
    return initialAssignmentRules;
}



} /* namespace rr */


