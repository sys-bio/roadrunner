/*
 * ConservedMoietyConverter.cpp
 *
 *  Created on: Oct 19, 2013
 *      Author: andy
 */

#include "ConservedMoietyConverter.h"
#include "ConservedMoietyPlugin.h"
#include "ConservationDocumentPlugin.h"
#include "rr-libstruct/lsLibStructural.h"

#include <sbml/conversion/SBMLConverterRegistry.h>
#include <sbml/conversion/SBMLLevelVersionConverter.h>
#include "ConservationExtension.h"
#include "rrSBMLReader.h"
#include "rrLogger.h"
#include "rrConfig.h"

#include <sbml/math/ASTNode.h>
#include <sbml/AlgebraicRule.h>
#include <sbml/AssignmentRule.h>
#include <sbml/RateRule.h>
#include <sbml/InitialAssignment.h>
#include <sbml/SBMLDocument.h>
#include <sbml/Model.h>

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <stdexcept>

#include "rrStringUtils.h"

#include <Poco/UUIDGenerator.h>

using namespace std;
using namespace libsbml;

static const int loggingLevel = rr::Logger::LOG_DEBUG;

namespace rr { namespace conservation {

/**
 * A little hack so we can copy species and get the ConservedMoiety plugin.
 *
 * Otherwise, we would have had to essentially re-write the Species
 * copy ctor, which is problematic should the contents of Species
 * ever change.
 */
class ConservedMoietySpecies : public libsbml::Species
{
public:
    ConservedMoietySpecies(libsbml::Species& orig, bool conservedMoiety, const std::string& quantity = "") :
        libsbml::Species(orig)
    {
        ConservationPkgNamespaces ns(3,2,1);
        this->loadPlugins(&ns);

        ConservedMoietyPlugin *plugin = (ConservedMoietyPlugin*)getPlugin("conservation");

        assert(plugin && "could not get conservation plugin from new species");

        plugin->setConservedMoiety(conservedMoiety);

        if (quantity.size()) {
            plugin->setConservedQuantity(quantity);
        }
    }

    void setConservedQuantity(const std::string& quantity) {
        ConservedMoietyPlugin *plugin = (ConservedMoietyPlugin*)getPlugin("conservation");

        assert(plugin && "could not get conservation plugin from species");

        plugin->setConservedQuantity(quantity);
    }

    void addConservedQuantity(const std::string& quantity) {
        ConservedMoietyPlugin *plugin = (ConservedMoietyPlugin*)getPlugin("conservation");

        assert(plugin && "could not get conservation plugin from species");

        plugin->addConservedQuantity(quantity);
    }
};


typedef std::vector<std::string> StringVector;

/**
 * check if a document is valid for conversion,
 * raises std::invalid_argument exception if not.
 */
static void conservedMoietyCheck(const SBMLDocument *doc);

/**
 * remove any species from new model and replace with species
 * in
 */
static void createReorderedSpecies(Model* newModel, Model* oldModel,
        const std::vector<std::string>& indSpecies,
        const std::vector<std::string>& depSpecies);

/**
 * creates a set of global parameters for the conserved moieties.
 *
 * A set of initialAssigments are also created to set thier values.
 */
static std::vector<std::string> createConservedMoietyParameters(
        Model* newModel,
        const ls::DoubleMatrix& L0, const std::vector<std::string>& indSpecies,
        const std::vector<std::string>& depSpecies);


static void createDependentSpeciesRules(Model* newModel,
        const ls::DoubleMatrix& L0,
        const std::vector<std::string>& conservedMoieties,
        const std::vector<std::string>& indSpecies,
        const std::vector<std::string>& depSpecies);

static void updateReactions(Model* newModel,
        const std::vector<std::string>& indSpecies,
        const std::vector<std::string>& depSpecies);

/**
 * creates an ast node for a species, and this value will alwasy be an
 * amount.
 */
static ASTNode *createSpeciesAmountNode(const Model* model, const std::string& name);

static void insertAsModifier(Reaction* reaction, SimpleSpeciesReference *s);

void ConservedMoietyConverter::init()
{
    ConservedMoietyConverter converter;
    SBMLConverterRegistry::getInstance().addConverter(&converter);
}

ConservedMoietyConverter::ConservedMoietyConverter() :
        SBMLConverter(),
        mModel(0),
        structural(0),
        resultDoc(0),
        resultModel(0)
{
}

ConservedMoietyConverter::ConservedMoietyConverter(
        const ConservedMoietyConverter& orig) :
        SBMLConverter(orig),
        mModel(0),
        structural(0),
        resultDoc(0),
        resultModel(0)
{
}

ConservedMoietyConverter::~ConservedMoietyConverter()
{
    delete structural;
    delete resultDoc;
}

SBMLConverter* ConservedMoietyConverter::clone() const
{
    return new ConservedMoietyConverter(*this);
}

ConversionProperties ConservedMoietyConverter::getDefaultProperties() const
{
    static ConversionProperties prop;
    prop.addOption("sortRules", true,
            "Sort AssignmentRules and InitialAssignments in the model");
    return prop;
}

bool ConservedMoietyConverter::matchesProperties(
        const ConversionProperties &props) const
{
    if (&props == NULL || !props.hasOption("sortRules"))
        return false;
    return true;
}

int ConservedMoietyConverter::convert()
{
    int err = 0;

    if (mDocument == NULL)
    {
        Log(Logger::LOG_ERROR) << "ConservedMoietyConverter document as not been set";
        return LIBSBML_INVALID_OBJECT;
    }

    if (mDocument->checkL3v2Compatibility() != 0)
    {
        Log(Logger::LOG_ERROR) << "ConservedMoietyConverter document not compatible with L3v2 "
                << std::endl;
        return 1;
    }

    if (!mDocument->setLevelAndVersion(3,2))
    {
        Log(Logger::LOG_ERROR) << "ConservedMoietyConverter mDocument->setLevelAndVersion failed "
                << std::endl;
        return 1;
    }


    Model* mModel = mDocument->getModel();
    if (mModel == NULL)
    {
        Log(Logger::LOG_ERROR) << "ConservedMoietyConverter document does not have a model";
        return LIBSBML_INVALID_OBJECT;
    }

    /* The document was checked for consistency in setDocument */
    ConservationPkgNamespaces ns(3,2,1);
    if (mDocument->isPackageURIEnabled("http://www.sbml.org/sbml/level3/version1/fbc/version2"))
        ns.addNamespace("http://www.sbml.org/sbml/level3/version1/fbc/version2", "fbc");
    if (mDocument->isPackageURIEnabled("http://www.sbml.org/sbml/level3/version1/fbc/version1"))
        ns.addNamespace("http://www.sbml.org/sbml/level3/version1/fbc/version1", "fbc");
    if (mDocument->isPackageURIEnabled("http://www.sbml.org/sbml/level3/version1/layout/version1"))
        ns.addNamespace("http://www.sbml.org/sbml/level3/version1/layout/version1", "layout");
    if (mDocument->isPackageURIEnabled("http://www.sbml.org/sbml/level3/version1/render/version1"))
        ns.addNamespace("http://www.sbml.org/sbml/level3/version1/render/version1", "render");
    resultDoc = new SBMLDocument(&ns);

    ConservationDocumentPlugin *docPlugin = dynamic_cast<ConservationDocumentPlugin*>
        (resultDoc->getPlugin("conservation"));

    assert(docPlugin && "ConservationDocumentPlugin is NULL");

    if ((err = docPlugin->setRequired(true)) != LIBSBML_OPERATION_SUCCESS)
    {
        Log(Logger::LOG_ERROR) << "ConservedMoietyConverter docPlugin->setRequired(true) failed: "
                << std::endl
                << OperationReturnValue_toString(err);
        return err;
    }

    // makes a clone of the model
    if((err = resultDoc->setModel(mModel)) != LIBSBML_OPERATION_SUCCESS)
    {
        Log(Logger::LOG_ERROR) << "ConservedMoietyConverter resultDoc->setModel(mModel) failed: "
                << std::endl
                << OperationReturnValue_toString(err);
        return err;
    }

    resultModel = resultDoc->getModel();

    assert(resultModel && "resultModel is NULL");

    vector<string> indSpecies = structural->getIndependentSpecies();

    vector<string> depSpecies = structural->getDependentSpecies();

    // creates a new matrix
    ls::DoubleMatrix *L0 = structural->getL0Matrix();

    if (rr::Logger::getLevel() >= loggingLevel)
    {
        Log(loggingLevel) << "performing conversion on " << mModel->getName();
        Log(loggingLevel) << "independent species: " << toString(indSpecies);
        Log(loggingLevel) << "dependent species: " << toString(depSpecies);
        Log(loggingLevel) << "L0 matrix: " << endl << *L0;
        Log(loggingLevel) << "Stoichiometry Matrix: " << endl
                << *(structural->getStoichiometryMatrix());
        Log(loggingLevel) << "Reordered Stoichiometry Matrix: "
                << endl << *(structural->getReorderedStoichiometryMatrix());
    }

    createReorderedSpecies(resultModel, mModel, indSpecies, depSpecies);

    vector<string> conservedMoieties = createConservedMoietyParameters(
            resultModel, *L0, indSpecies, depSpecies);

    createDependentSpeciesRules(resultModel, *L0, conservedMoieties,
            indSpecies, depSpecies);

    updateReactions(resultModel, indSpecies, depSpecies);

    delete L0;

    return LIBSBML_OPERATION_SUCCESS;
}

libsbml::SBMLDocument* ConservedMoietyConverter::getDocument()
{
    return resultDoc;
}

const libsbml::SBMLDocument* ConservedMoietyConverter::getDocument() const
{
    return resultDoc;
}

int ConservedMoietyConverter::setDocument(const libsbml::SBMLDocument* doc)
{
    delete structural; structural = 0;
    delete resultDoc; resultDoc = 0;

    int result = 0;

    if (doc == 0)
    {
        Log(Logger::LOG_ERROR) << "ConservedMoietyConverter::setDocument argument is NULL";
        return LIBSBML_INVALID_OBJECT;
    }

    // throws exception if invalid.
    conservedMoietyCheck(doc);

    // The version converter checks consistency, if we are already
    // at L3V1, need to check consistency ourselves.
    if (doc->getLevel() != ConservationExtension::getDefaultLevel()
        || doc->getVersion() != ConservationExtension::getDefaultVersion())
    {
        if ((rr::Config::getInt(rr::Config::ROADRUNNER_DISABLE_WARNINGS) &
                rr::Config::ROADRUNNER_DISABLE_WARNINGS_CONSERVED_MOIETY) == 0)
        {
            Log(rr::Logger::LOG_NOTICE) << "source document is level " << doc->getLevel()
                        << ", version " << doc->getVersion() << ", converting to "
                        << "level " << ConservationExtension::getDefaultLevel()
                        << ", version " << ConservationExtension::getDefaultVersion()
                        << " for Moiety Conservation Conversion";
        }

        // this does an in-place conversion, at least for the time being
        SBMLLevelVersionConverter versionConverter;

        ConversionProperties versionProps = versionConverter.getDefaultProperties();

        versionProps.addOption("strict", false);

        versionConverter.setProperties(&versionProps);

        // need to set what checks the converter does. The const doc arg is fixed in 5.10, but
        // we work with 5.9 for now.
        libsbml::SBMLDocument* pdoc = const_cast<libsbml::SBMLDocument*>(doc);
        pdoc->setApplicableValidators((unsigned char)Config::getInt(
                Config::SBML_APPLICABLEVALIDATORS));

        versionConverter.setDocument(doc);

        if ((result = versionConverter.convert()) != LIBSBML_OPERATION_SUCCESS)
        {
            Log(rr::Logger::LOG_ERROR) << "could not upgrade source sbml level or version";

            const SBMLErrorLog *log = doc->getErrorLog();
            string errors = log ? log->toString() : string(" NULL SBML Error Log");
            Log(rr::Logger::LOG_ERROR) << "Conversion Errors: " + errors;

            return result;
        }

        // TODO, in case behavior of version converter changes,
        // need to update ownership logic.
        doc = versionConverter.getDocument();
    }
    else
    {
        if (!rr::Config::getBool(rr::Config::LOADSBMLOPTIONS_PERMISSIVE)) {
            // need to set what checks the converter does. The const doc arg is fixed in 5.10, but
            // we work with 5.9 for now.

            libsbml::SBMLDocument* pdoc = const_cast<libsbml::SBMLDocument*>(doc);
            pdoc->setApplicableValidators((unsigned char)Config::getInt(
                    Config::SBML_APPLICABLEVALIDATORS));


            /* use validators that the user has selected
            */
            /* hack to catch errors caught at read time */
            char* docStr = writeSBMLToString(pdoc);
            SBMLDocument *d = readSBMLFromString(docStr);
            util_free(docStr);
            unsigned int errors = d->getNumErrors();

            for (unsigned int i = 0; i < errors; i++)
            {
                pdoc->getErrorLog()->add(*(d->getError(i)));
            }
            delete d;

            errors += pdoc->checkConsistency();
            errors = pdoc->getErrorLog()->getNumFailsWithSeverity(LIBSBML_SEV_ERROR);

            /* if the current model is not valid dont convert and dont set
            */
            if (errors > 0)
            {
                Log(rr::Logger::LOG_ERROR) << "Invalid document for moiety conversion:";

                const SBMLErrorLog *log = pdoc->getErrorLog();
                string errors = log ? log->toString() : string(" NULL SBML Error Log");
                Log(rr::Logger::LOG_ERROR) << "Conversion Errors: " + errors;

                return LIBSBML_CONV_INVALID_SRC_DOCUMENT;
            }

            pdoc->getErrorLog()->clearLog();
        }

    }

    // sets mDocument
    if ((result = SBMLConverter::setDocument(doc)) != LIBSBML_OPERATION_SUCCESS)
    {
        return result;
    }

    // owned by the doc
    mModel = mDocument->getModel();
    if (mModel == NULL)
    {
        return LIBSBML_INVALID_OBJECT;
    }

    structural = new ls::LibStructural(mModel);

    return LIBSBML_OPERATION_SUCCESS;
}

const libsbml::SBMLDocument *ConservedMoietyConverter::getLevelConvertedDocument() const
{
    return mDocument;
}

static void createReorderedSpecies(Model* newModel, Model* oldModel,
        const std::vector<std::string>& indSpecies,
        const std::vector<std::string>& depSpecies)
{
    // remove all the existing independent species
    ListOfSpecies *species = newModel->getListOfSpecies();

    unsigned index = 0;

    while(index < species->size())
    {
        Species *s = species->get(index);
        if (!s->getBoundaryCondition())
        {
            species->remove(index);
            delete s;
        }
        else
        {
            index++;
        }
    }

#if DEBUG
    // TODO make sure that remaining species are in the given list.
#endif


    species = oldModel->getListOfSpecies();
    ListOfSpecies *newSpecies = newModel->getListOfSpecies();

    // insert independent first, then dependent
    index = 0;
    for (int i = 0; i < indSpecies.size(); ++i)
    {
        Species *s = species->get(indSpecies[i]);

        assert(s && "could not get independent species from original model");

        newSpecies->insertAndOwn(index++, new ConservedMoietySpecies(*s, false));
    }

    for (int i = 0; i < depSpecies.size(); ++i)
    {
        Species *s = species->get(depSpecies[i]);

        assert(s && "could not get dependent species from original model");

        newSpecies->insertAndOwn(index++, new ConservedMoietySpecies(*s, true));
    }
}

/**
 * create the conservied moiteies by
 * T = S_d (0) - L_0 S_i (0)
 */
static std::vector<std::string> createConservedMoietyParameters(
        Model* newModel,
        const ls::DoubleMatrix& L0, const std::vector<std::string>& indSpecies,
        const std::vector<std::string>& depSpecies)
{
    StringVector conservedMoieties(depSpecies.size());

    Poco::UUIDGenerator uuidGen;

    for (int i = 0; i < depSpecies.size(); ++i)
    {
        Poco::UUID uuid = uuidGen.create();
        string id = "_CSUM" + rr::toString(i);
        std::replace( id.begin(), id.end(), '-', '_');


        ConservedMoietySpecies* cmDepSpecies = dynamic_cast<ConservedMoietySpecies*>(newModel->getSpecies(indSpecies.size()+i));
        if (cmDepSpecies)
            cmDepSpecies->setConservedQuantity(id);

        Parameter *cm = newModel->createParameter();
        cm->setId(id);
        cm->setConstant(true);

        ConservedMoietyPlugin *plugin = dynamic_cast<ConservedMoietyPlugin*>(
                cm->getPlugin("conservation"));
        plugin->setConservedMoiety(true);

        conservedMoieties[i] = id;

        InitialAssignment *ia = newModel->createInitialAssignment();
        ia->setSymbol(id);

        ASTNode sum(AST_PLUS);

        ASTNode *Sd0 = createSpeciesAmountNode(newModel, depSpecies[i]);
        sum.addChild(Sd0);

        ASTNode *mult = new ASTNode(AST_TIMES);
        sum.addChild(mult);

        ASTNode *neg = new ASTNode(AST_REAL);
        neg->setValue(-1.0);
        mult->addChild(neg);

        ASTNode *sum2 = new ASTNode(AST_PLUS);
        mult->addChild(sum2);

        for (int j = 0; j < indSpecies.size(); ++j)
        {
            double stoich = L0(i, j);

            if (stoich != 0)
            {
                ASTNode *times = new ASTNode(AST_TIMES);
                ASTNode *value = new ASTNode(AST_REAL);
                ASTNode *species = createSpeciesAmountNode(newModel, indSpecies[j]);

                value->setValue(stoich);
                times->addChild(value);
                times->addChild(species);

                sum2->addChild(times);

                ConservedMoietySpecies* cmIndSpecies = dynamic_cast<ConservedMoietySpecies*>(newModel->getSpecies(j));
                if (cmIndSpecies) {
                    std::cerr << "cmIndSpecies " << cmIndSpecies->getId() << " conserved quantity " << id << "\n";
                    cmIndSpecies->addConservedQuantity(id);
                }
            }
        }

        ia->setMath(&sum);
    }

    return conservedMoieties;
}

static void createDependentSpeciesRules(Model* newModel,
        const ls::DoubleMatrix& L0,
        const std::vector<std::string>& conservedMoieties,
        const std::vector<std::string>& indSpecies,
        const std::vector<std::string>& depSpecies)
{
    for (unsigned i = 0; i < depSpecies.size(); ++i)
    {
        const string& id = depSpecies[i];

        const Species *dspecies = newModel->getSpecies(id);
        if (dspecies == 0)
        {
            throw std::invalid_argument("model does not contain dependent species " + id);
        }

        bool isAmt = dspecies->getHasOnlySubstanceUnits();

        AssignmentRule *rule = newModel->createAssignmentRule();
        rule->setVariable(id);

        ASTNode *amt = new ASTNode(AST_PLUS);

        // conserved moiety node
        ASTNode *t = new ASTNode(AST_NAME);
        t->setName(conservedMoieties[i].c_str());

        amt->addChild(t);

        for (int j = 0; j < indSpecies.size(); ++j)
        {
            double stoich = L0(i, j);

            if (stoich != 0)
            {
                ASTNode *times = new ASTNode(AST_TIMES);
                ASTNode *value = new ASTNode(AST_REAL);
                ASTNode *ispecies = createSpeciesAmountNode(newModel, indSpecies[j]);

                value->setValue(stoich);
                times->addChild(value);
                times->addChild(ispecies);

                amt->addChild(times);
            }
        }

        if (isAmt)
        {
            rule->setMath(amt);
            delete amt;
            amt = 0;
        }
        else
        {
            ASTNode conc(AST_DIVIDE);
            ASTNode *volume = new ASTNode(AST_NAME);
            volume->setName(dspecies->getCompartment().c_str());
            conc.addChild(amt);
            conc.addChild(volume);
            rule->setMath(&conc);
        }
    }
}

static void updateReactions(Model* newModel,
        const std::vector<std::string>& indSpecies,
        const std::vector<std::string>& depSpecies)
{
    set<string> depSet(depSpecies.begin(), depSpecies.end());

    ListOfReactions *reactions = newModel->getListOfReactions();

    for (unsigned i = 0; i < reactions->size(); ++i)
    {
        Reaction *r = reactions->get(i);
        ListOfSpeciesReferences *products = r->getListOfProducts();
        ListOfSpeciesReferences *reactants = r->getListOfReactants();

        unsigned j = 0;
        while(j < products->size())
        {
            SimpleSpeciesReference *product = products->get(j);

            if (depSet.find(product->getSpecies()) != depSet.end())
            {
                products->remove(j);
                insertAsModifier(r, product);
                delete product;
            }
            else
            {
                j++;
            }
        }

        j = 0;
        while(j < reactants->size())
        {
            SimpleSpeciesReference *reactant = reactants->get(j);

            if (depSet.find(reactant->getSpecies()) != depSet.end())
            {
                reactants->remove(j);
                insertAsModifier(r, reactant);
                delete reactant;
            }
            else
            {
                j++;
            }
        }
    }
}

static void insertAsModifier(Reaction* reaction, SimpleSpeciesReference *s)
{
    ModifierSpeciesReference *m = reaction->createModifier();
    m->setId(s->getId());
    m->setSpecies(s->getSpecies());
    m->setName(s->getName());
}


static ASTNode *createSpeciesAmountNode(const Model* model, const std::string& name)
{
    const Species* species = model->getSpecies(name);

    if(species == 0)
    {
        throw std::invalid_argument("model does not have species with name" + name);
    }

    ASTNode *speciesNode = new ASTNode(AST_NAME);
    speciesNode->setName(name.c_str());

    if(species->getHasOnlySubstanceUnits())
    {
        // its an amount, we're good
        return speciesNode;
    }
    else
    {
        ASTNode *mul = new ASTNode(AST_TIMES);
        ASTNode *volume = new ASTNode(AST_NAME);
        volume->setName(species->getCompartment().c_str());
        mul->addChild(volume);
        mul->addChild(speciesNode);
        return mul;
    }
}

static inline void conservedMoietyException(const std::string& what)
{
    Log(rr::Logger::LOG_INFORMATION) << what;

    static const char* help = "\n To disable conserved moiety conversion, either \n"
            "\t a: set [Your roadrunner variable].conservedMoietyAnalysis = False, \n"
            "\t before calling the load(\'myfile.xml\') method, or\n"
            "\t b: create a LoadSBMLOptions object, set the conservedMoieties property \n"
            "\t to False and use this as the second argument to the RoadRunner \n"
            "\t constructor or load() method, i.e. \n"
            "\t o = roadrunner.LoadSBMLOptions()\n"
            "\t o.conservedMoieties = False\n"
            "\t r = roadrunner.RoadRunner(\'myfile.xml\', o)\n";
    throw std::invalid_argument(what + help);
}

static void conservedMoietyCheck(const SBMLDocument *doc)
{

    const Model *model = doc->getModel();

    // check if any species are defined by assignment rules
    const ListOfRules *rules = model->getListOfRules();

    for(int i = 0; i < rules->size(); ++i)
    {
        const Rule *rule = rules->get(i);

        const SBase *element = const_cast<Model*>(model)->getElementBySId(
                rule->getVariable());

        const Species *species = dynamic_cast<const Species*>(element);
        if(species && !species->getBoundaryCondition() && model->getNumReactions() > 0)
        {
            string msg = "Cannot perform moiety conversion when floating "
                    "species are defined by rules. The floating species, "
                    + species->getId() + " is defined by rule " + rule->getId()
                    + ".";
            conservedMoietyException(msg);
        }

        const SpeciesReference *ref =
                dynamic_cast<const SpeciesReference*>(element);
        if(ref)
        {
            string msg = "Cannot perform moiety conversion with non-constant "
                    "stoichiometry. The species reference " + ref->getId() +
                    " which refers to species " + ref->getSpecies() + " has "
                    "stoichiometry defined by rule " + rule->getId() + ".";
            conservedMoietyException(msg);
        }
    }

    const ListOfReactions *reactions = model->getListOfReactions();
    for(int i = 0; i < reactions->size(); ++i)
    {
        const Reaction *reaction = reactions->get(i);

        const ListOfSpeciesReferences *references = reaction->getListOfProducts();

        for (int i = 0; i < references->size(); ++i)
        {
            const SpeciesReference *ref =
                    dynamic_cast<const SpeciesReference*>(references->get(i));

            // has the constant attribute
            if (doc->getLevel() >= 3 &&  !ref->getConstant())
            {
                string msg = "Cannot perform moiety conversion with non-constant "
                        "stoichiometry. The species reference " + ref->getId() +
                        " which refers to species " + ref->getSpecies() +
                        " does not have the constant attribute set.";
                conservedMoietyException(msg);
            }

            else if(ref->isSetStoichiometryMath())
            {
                string msg = "Cannot perform moiety conversion with non-constant "
                        "stoichiometry. The species reference " + ref->getId() +
                        " which refers to species " + ref->getSpecies() +
                        " has stochiometryMath set.";
                conservedMoietyException(msg);
            }
        }
    }

    const ListOfEvents *events = model->getListOfEvents();
    for(int i = 0; i < events->size(); ++i)
    {
        const Event *event = events->get(i);
        const ListOfEventAssignments *assignments =
                event->getListOfEventAssignments();

        for(int j = 0; j < assignments->size(); ++j)
        {
            const EventAssignment *ass = assignments->get(j);
            const SBase *element = const_cast<Model*>(model)->getElementBySId(
                    ass->getVariable());

            const Species *species = dynamic_cast<const Species*>(element);
            if(species && !species->getBoundaryCondition())
            {
                string msg = "Cannot perform moiety conversion when floating "
                        "species have events. The floating species, "
                        + species->getId() + " has event " + event->getId() + ".";
                conservedMoietyException(msg);
            }

            const SpeciesReference *ref =
                    dynamic_cast<const SpeciesReference*>(element);
            if(ref)
            {
                string msg = "Cannot perform moiety conversion with non-constant "
                        "stoichiometry. The species reference " + ref->getId() +
                        " which refers to species " + ref->getSpecies() + " has "
                        "event " + event->getId() + ".";
                conservedMoietyException(msg);
            }
        }
    }
}


} // namespace conservation }



PyConservedMoietyConverter::PyConservedMoietyConverter()
    : doc(0)
{
}

PyConservedMoietyConverter::~PyConservedMoietyConverter()
{
    delete doc;
}

int PyConservedMoietyConverter::setDocument(const std::string& fileOrPath)
{
    delete doc;
    doc = 0;

    std::string sbml = rr::SBMLReader::read(fileOrPath);

    libsbml::SBMLReader reader;

    doc = reader.readSBMLFromString(sbml);

    return conv.setDocument(doc);
}

int PyConservedMoietyConverter::convert()
{
    return conv.convert();
}

std::string PyConservedMoietyConverter::getDocument()
{
    libsbml::SBMLWriter writer;

    libsbml::SBMLDocument *resultDoc = conv.getDocument();

    if (!resultDoc)
    {
        return "";
    }

    char* buffer = writer.writeToString(resultDoc);

    std::string result = buffer;

    delete[] buffer;

    return result;
}

} // namespace rr }
