#ifndef rrNOMSupportH
#define rrNOMSupportH
//---------------------------------------------------------------------------
#include <list>
#include <vector>
#include <deque>
#include <string>
#include <stack>
#include <memory>
#include "sbml/math/ASTNode.h"
#include "rrExporter.h"
#include "rrStringListContainer.h"
#include "rrHashTable.h"
//---------------------------------------------------------------------------

namespace rr
{
using std::vector;
using std::string;
using std::pair;
using std::list;
using std::stack;
using std::deque;

// You HAVE to define LIBSBML_USE_CPP_NAMESPACE for your project to compile this!
using libsbml::Model;
using libsbml::SBMLDocument;
using libsbml::ASTNode;
using libsbml::SBase;
using libsbml::Rule;
using libsbml::UnitDefinition;
using libsbml::KineticLaw;

/**
 * Methods to query various information from an SBML document.
 *
 * This class provides a set of static method to query various items
 * from sbml objects.
 */
class RR_DECLSPEC NOMSupport
{
public:
    NOMSupport();
    ~NOMSupport();
    Model*                  getModel();
    SBMLDocument*           getSBMLDocument();
    static string           getlibSBMLVersion();

    StringListContainer     getDerivedUnitDefinition(const string& sId);
    StringListContainer     getListOfBoundarySpeciesIds();
    StringListContainer     getListOfErrors();
    StringListContainer     getListOfFloatingSpecies();
    StringListContainer     getListOfFloatingSpeciesIds();
    StringListContainer     getListOfParameters();
    StringListContainer     getNthError(const int& nIndex);
    StringListContainer     getNthEvent(const int& arg);
    StringListContainer     getNthFunctionDefinition(const int& arg);
    StringListContainer     getNthListOfModifiers(const int& nIndex);
    StringListContainer     getNthListOfProducts(const int& nIndex);
    StringListContainer     getNthListOfReactants(const int& nIndex);
    ASTNode                 replaceSymbol(ASTNode* node, const string& oldId, const string& newId);
    bool                    exists(const string& sId);
    bool                    getNthParameterHasValue(const int& nReactionIndex, const int& nParameterIndex);
    bool                    getNthUseValuesFromTriggerTime(const int& arg);
    bool                    hasInitialAmount(const string& sId);
    bool                    hasInitialConcentration(const string& sId);
    bool                    hasSBOTerm(const string& sId);
    bool                    hasValue(const string& sId);

    /**
     * Checks whether the element for the given SBML id is a boundary species
     *
     * @ param name the id to check
     */
    bool                    isBoundary(const string& sId);

    /**
     * Checks whether the element for the given SBML id is a compartment
     *
     * @param sId the id to check
     * @returns true if element is a compartment, false otherwise
     */
    bool                    isCompartment(const string& sId);
    bool                    isConstantImpl(const string& sId);
    bool                    isFloating(const string& sId);
    bool                    isReactionReversible(const int& nIndex);
    bool                    isSpecies (const string& sId);
    bool                    multiplyCompartment(const string& sbmlId, string& compartmentId);
    bool                    needEmptySetNode(Model model);
    bool                    needSinkNode(Model model);
    bool                    needSourceNode(Model model);
    double                  getNthParameterValue(const int& nReactionIndex, const int& nParameterIndex);
    double                  getNthProductStoichiometryDouble(const int& nIndex, const int& nProduct);
    double                  getNthReactantStoichiometryDouble(const int& nIndex, const int& nReactant);
    double                  getValue(const string& sId);
    int                     checkConsistency();
    int                     getNthProductStoichiometry(const int& nIndex, const int& nProduct);
    int                     getNthReactantStoichiometry(const int& nIndex, const int& nReactant);
    int                     getNumBoundarySpecies();
    int                     getNumCompartments();
    int                     getNumConstraints();
    int                     getNumErrors();
    int                     getNumEvents();
    int                     getNumFloatingSpecies();
    int                     getNumFunctionDefinitions();
    int                     getNumGlobalParameters();
    int                     getNumInitialAssignments();

    /**
     * @return the number of parameters for the i'th reaction, if reaction is
     * not a kinetic law, 0 otherwise.
     */
    int                     getNumParameters(int reaction);

    int                     getNumProducts(const int& var0);
    int                     getNumReactants(const int& var0);

    /**
     * @return the number of Reactions in this Model.
     */
    int                     getNumReactions();

    int                     getNumRules();
    int                     getSBOTerm(const string& sId);
    pair<string, string>    getNthInitialAssignmentPair(const int& nIndex);
    SBase*                  getElement(const string& sId);

    string                  addEmptySetNode(const string& sbml);
    string                  addEmptySetNodes(const string& sbml);
    string                  addMissingModifiers(const string& sModel);
    string                  addSourceSinkNodes(const string& sbml);
    string                  convertLevel1ToLevel2Impl(const string& sSBML);
    string                  convertLevel2ToLevel1Impl(const string& sSBML);

    string                  convertPowImpl(const string& sSBML);
    string                  convertSBML(const string& sModel, const int& nLevel, const int& nVersion);
    string                  convertSBML(const string& sModel, const int& nLevel, const int& nVersion, bool throwError);
    string                  convertTimeToCSymbol(const string& sArg, const string& sTimeSymbol);
    string                  fixCommonIssues(const string& sbml);
    string                  fixCommonIssues(const string& sbml, const string& programName, const string& programVersion);
    string                  fetAnnotatedModel(const string& targetSBML, const string& sourceSBML, bool checkModelId);

    string                  getAnnotation(const string& sId);
    string                  getCompartmentIdBySpeciesId(const string& sId);
    /**
     * gets the kinetic law math formula for the i'th reaction.
     * This will be changed to return a KineticLaw object soon.
     */
    string                  getKineticLaw(const int& index);
    string                  getMetaId(const string& sId);
    string                  getModelId();
    string                  getModelName() const;
    string                  getNotes(const string& sId);
    string                  getNthBoundarySpeciesId(const int& nIndex);
    string                  getNthBoundarySpeciesName(const int& nIndex);
    string                  getNthCompartmentId(const int& nIndex);
    string                  getNthCompartmentName(const int& nIndex);
    string                  getNthConstraint(const int& nIndex, string& sMessage);
    string                  getNthFloatingSpeciesId(const int& nIndex);
    string                  getNthFloatingSpeciesName(const int& nIndex);
    string                  getNthGlobalParameterId(const int& nIndex);
    string                  getNthGlobalParameterName(const int& nIndex);
    string                  getNthInitialAssignment(const int& nIndex);
    string                  getNthParameterId(const int& nReactionIndex, const int& nParameterIndex);
    string                  getNthParameterName(const int& nReactionIndex, const int& nParameterIndex);
    string                  getNthProductName(const int& nIndex, const int& nProduct);
    string                  getNthReactantName(const int& nIndex, const int& nReactant);
    string                  getNthReactionId(const int& nIndex);
    string                  getNthReactionName(const int& nIndex);
    string                  getNthRule(const int& nIndex);
    string                  getNthRuleType(const int& arg);
    string                  getOutsideCompartment(const string& var0);

    string                  getSBML();
    string                  getSBOCapableSBML(const string& sModel);
    string                  removeJD1Layout(const string& sSBML);
    string                  removeJD2Layout(const string& sSBML);
    string                  removeLayoutInformation(const string& sSBML);
    string                  validateSBML(const string& sModel);
    string                  validateWithConsistency(const string& sModel);
    vector<string>          getBuiltinFunctionInfo(const string& var0);
    vector<string>          getBuiltinFunctions();
    void                    changeConstantForRules(Model* model);

    void                    loadFromFile(const string& fileName);
    void                    loadParameterPromotedSBML(const string& var0, const string& sTimeSymbol);
    void                    loadSBML(const string& var0, const std::string& filename);
    void                    loadSBML(const string& var0, const string& sTimeSymbol, const std::string& filename);

    void                    setAnnotation(const string& sId, const string& sAnnotation);
    void                    setModelId(const string& sId);
    void                    setNotes(const string& sId, const string& sNotes);
    void                    setSBOTerm(const string& sId, const int& nSBOTerm);
    void                    setValue(const string& sId, const double& dValue);
    void                    setValue(Model* model, const string& id, const double& value, const bool& throwIfNotFound);
    void                    testASTTime();

    string                  getNthBoundarySpeciesCompartmentName(const int& nIndex);

    /**
     * gets the compartment name of the i'th species with boundary condition's
     * compartment name. So, say we have a model like:
     * model {
     *     species{boundary = false}
     *     species{boundary = true}
     *     species{boundary = false}
     *     species{boundary = true}
     *     species{boundary = true}
     *     ...
     * }
     * Using 0 based indexing, if index = 0, we'd get the compartment name for species 1,
     * index = 1 yields compartment name for species 3,
     * index = 2 yields compartment name for species 4, and so forth.
     */
    string                  getNthFloatingSpeciesCompartmentName(int index);
    StringListContainer     getListOfBoundarySpecies();

    /**
     * stateless utility functions, these obviously do not
     * change the state of this NOM object.
     */
    static string           convertTime(const string& sArg, const string& sTimeSymbol);

    /**
     * Returns the list of all names contained in the ASTNode
     *
     * @param math ASTNode
     * @returns List of all symbols
     */
    static StringList       getSymbols(const ASTNode* math);

    /**
     * This should return an initialization for the given sbmlId that is sideeffect free
     */
    stack<string>           getMatchForSymbol(const string& sbmlId);

    /**
     * Reorders assignment rules. In SBML assignment rules does not have to appear in the correct order.
     * That is you could have an assignment rule A = B, and a rule B = C. Now the result would differ,
     * if the first rule is evaluated before the second. Thus the rules will be reordered such that
     * this will be taken care of.
     *
     * @param assignmentRules assignment rules in original ordering
     * @return assignment rules in independent order
     */
    static deque<Rule*>     reorderAssignmentRules(deque<Rule*>& assignmentRules);

    static void             fillStack(stack<string>& stack, SBMLSymbol& symbol);

    /**
     * Reorders the Rules of the model in such a way, that AssignmentRules are calculated first,
     * followed by Rate Rules and Algebraic Rules.
     *
     * @param doc the document to use
     * @param name the model to use
     */
    static void             reorderRules(SBMLDocument& oc, Model& model);

    static string           getParamPromotedSBML(const string& sArg);

    static string           getId(SBase& element);
    static string           convertMathMLToString(const string& sMathML);

    static string           convertStringToMathML(const string& var0);

private:

    /**
     * the loaded sbml doc, we own this.
     */
    SBMLDocument            *mSBMLDoc;

    /**
     * reference to the model object owned by the sbml doc,
     * hence we do NOT delete this.
     */
    Model                   *mModel;

    StringSymbolHashTable   mSymbolTable;

    StringListContainer     returnUnitDefinition(UnitDefinition oDefinition);

    ASTNode                 changeTimeToCSymbol(ASTNode* node, const string& name, const int& type);
    bool                    addMissingModifiers(Model *oModel);
    StringList              getSymbols(const string& formula);

    string                  getInitialAssignmentFor(const string& sbmlId);

    string                  getRuleFor(const string& sbmlId);
    void                    addDependenciesToList(const ASTNode* node, StringList& sResult);
    void                    addMissingParameter(const string& parameterId, SBMLDocument *doc);
    void                    buildSymbolTable();
    void                    changeNameToCSymbol(Model* model, const string& name, const int& type);
    void                    changePow(ASTNode* node);
    void                    checkForMissingNames(ASTNode *node, StringListContainer results, StringListContainer symbols);
       void                    lookForDependencies();
    void                    removeSpatialSizeUnitsFromSpecies(SBMLDocument* doc);
    void                    removeSubstanceUnitsFromKineticLaws(SBMLDocument* doc);
    void                    removeTimeUnitsFromKineticLaws(SBMLDocument* doc);
    void                    updateDependencies(const string& sbmlId);
    void                    upgradeToL2V4IfNecessary(SBMLDocument* doc);

    /**
     * clear the state of this object, free any objects we own.
     */
    void                    reset();
    /**
     * returns empty string if element is null, otherwise gets the element's
     * name if its set, otherwise gets the element's id.
     */
    static string           getName(const SBase* element);
    static void             modifyKineticLaws(SBMLDocument& oSBMLDoc, Model &oModel);
    static void             modifyKineticLawsForLocalParameters(KineticLaw& oLaw, const string& reactionId, Model &oModel);
    static void             modifyKineticLawsForReaction(KineticLaw& oLaw, const string& reactionId, Model &oModel);
    static void             changeSymbol(Model& oModel, const string& sTimeSymbol, const int& targetType);
    static void             changeTimeSymbol(Model& model, const string& timeSymbol);

    static void             changeParameterName(ASTNode& node, const string& sParameterName, const string& sPrefix);
    static void             getSymbols(const ASTNode* node, StringList& list);

    /**
     * type of formating we should use for printing doubles.
     */
    static const string     STR_DoubleFormat;
};

/**
 * SBML_formulaToString is used all over the place here,
 * SBML_formulaToString returns a char* that MUST BE FREED!!!
 *
 * This function frees the string and returns a std::string with its contents.
 */
std::string SBML_formulaToStdString(const ASTNode *tree);

}//namespace rr
#endif
