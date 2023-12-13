//
// Created by Ciaran on 07/05/2021.
//

#include "RoadRunnerAPITests.h"
#include "rrConfig.h"

using namespace rr;

class RoadRunnerAPITestsWithLLJit : public RoadRunnerAPITests {
public:
    RoadRunnerAPITestsWithLLJit() 
        : RoadRunnerAPITests(){
        //Logger::setLevel(Logger::LOG_DEBUG);
        Config::setValue(Config::LLVM_BACKEND, Config::LLVM_BACKEND_VALUES::LLJIT);
    }
};



TEST_F(RoadRunnerAPITestsWithLLJit, DefaultJacobianMode){
    DefaultJacobianMode();
}

TEST_F(RoadRunnerAPITestsWithLLJit, SetJacobianModeToAmt){
    SetJacobianModeToAmt();
}

TEST_F(RoadRunnerAPITestsWithLLJit, GetFullJacobianDefaultConfigSettings){
    GetFullJacobianDefaultConfigSettings();
}

TEST_F(RoadRunnerAPITestsWithLLJit, GetFullJacobianUsingConcMode){
    GetFullJacobianUsingConcMode();
}

TEST_F(RoadRunnerAPITestsWithLLJit, GetFullJacobianUsingAmtMode){
    GetFullJacobianUsingAmtMode();
}

TEST_F(RoadRunnerAPITestsWithLLJit, GetFullJacobianUsingAmtModeAsLong){
    GetFullJacobianUsingAmtModeAsLong();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getIds){
    getIds();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getIdsAfterRegenerate){
    getIdsAfterRegenerate();
}

TEST_F(RoadRunnerAPITestsWithLLJit, DISABLED_getIdsAfterSetConservedMoiety){
    getIdsAfterSetConservedMoiety();
}

TEST_F(RoadRunnerAPITestsWithLLJit, LoadModelWithOnlyLayoutInformation){
    LoadModelWithOnlyLayoutInformation();
}

TEST_F(RoadRunnerAPITestsWithLLJit, loadSBMLWithLeadingNewlineCharacter){
    loadSBMLWithLeadingNewlineCharacter();
}

TEST_F(RoadRunnerAPITestsWithLLJit, RoadRunnerConstructorVersion){
    RoadRunnerConstructorVersion();
}

TEST_F(RoadRunnerAPITestsWithLLJit, DISABLED_RoadRunnerConstructorFromURL){
    DISABLED_RoadRunnerConstructorFromURL();
}

TEST_F(RoadRunnerAPITestsWithLLJit, DISABLED_CompilerTempDirSupportCodeDir){
    DISABLED_CompilerTempDirSupportCodeDir();
}

TEST_F(RoadRunnerAPITestsWithLLJit, CopyConstructor){
    CopyConstructor();
}

TEST_F(RoadRunnerAPITestsWithLLJit, DISABLED_EqualityOperators){
    DISABLED_EqualityOperators();
}

TEST_F(RoadRunnerAPITestsWithLLJit, DISABLED_InequalityOperators){
    DISABLED_InequalityOperators();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getInstanceID){
    getInstanceID();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getInstanceCount){
    getInstanceCount();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getParamPromotedSBML){
    getParamPromotedSBML();
}

TEST_F(RoadRunnerAPITestsWithLLJit, DISABLED_getCompiler){
    DISABLED_getCompiler();
}

TEST_F(RoadRunnerAPITestsWithLLJit, IsModelLoadedWhenFalse){
    IsModelLoadedWhenFalse();
}

TEST_F(RoadRunnerAPITestsWithLLJit, IsModelLoadedWhenTrue){
    IsModelLoadedWhenTrue();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getConservedMoietyAnalysisWhenFalse){
    getConservedMoietyAnalysisWhenFalse();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getConservedMoietyAnalysisWhenTrue){
    getConservedMoietyAnalysisWhenTrue();
}

TEST_F(RoadRunnerAPITestsWithLLJit, tmpDir){
    tmpDir();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getModelName){
    getModelName();
}

TEST_F(RoadRunnerAPITestsWithLLJit, clearModel){
    clearModel();
}

TEST_F(RoadRunnerAPITestsWithLLJit, oneStep){
    oneStep();
}

TEST_F(RoadRunnerAPITestsWithLLJit, internalOneStep){
    internalOneStep();
}

TEST_F(RoadRunnerAPITestsWithLLJit, saveState){
    saveState();
}

TEST_F(RoadRunnerAPITestsWithLLJit, loadState){
    loadState();
}

TEST_F(RoadRunnerAPITestsWithLLJit, loadStateFromCopy){
    loadStateFromCopy();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSimulationData){
    getSimulationData();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setSimulateOptions){
    setSimulateOptions();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSBML){
    getSBML();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getCurrentSBML){
    getCurrentSBML();
}

TEST_F(RoadRunnerAPITestsWithLLJit, reset){
    reset();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getCurrentTime){
    getCurrentTime();
}

TEST_F(RoadRunnerAPITestsWithLLJit, resetSelectionLists){
    resetSelectionLists();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSelections){
    getSelections();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getValue){
    getValue();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setSelections){
    setSelections();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSelectedValues){
    getSelectedValues();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getIndependentFloatingSpeciesIds){
    getIndependentFloatingSpeciesIds();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getDependentFloatingSpeciesIds){
    getDependentFloatingSpeciesIds();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getFloatingSpeciesInitialConcentrations){
    getFloatingSpeciesInitialConcentrations();
}

TEST_F(RoadRunnerAPITestsWithLLJit, DISABLED_getSupportedIdTypes){
    DISABLED_getSupportedIdTypes();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setValue){
    setValue();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getFloatingSpeciesAmountsNamedArray){
    getFloatingSpeciesAmountsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getFloatingSpeciesConcentrationsNamedArray){
    getFloatingSpeciesConcentrationsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getDependentFloatingSpeciesConcentrations) {
    getDependentFloatingSpeciesConcentrations();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getDependentFloatingSpeciesAmounts) {
    getDependentFloatingSpeciesAmounts();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getDependentFloatingSpeciesConcentrationsNamedArray) {
    getDependentFloatingSpeciesConcentrationsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getDependentFloatingSpeciesAmountsNamedArray) {
    getDependentFloatingSpeciesAmountsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getIndependentFloatingSpeciesConcentrations) {
    getIndependentFloatingSpeciesConcentrations();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getIndependentFloatingSpeciesAmounts) {
    getIndependentFloatingSpeciesAmounts();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getIndependentFloatingSpeciesConcentrationsNamedArray) {
    getIndependentFloatingSpeciesConcentrationsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getIndependentFloatingSpeciesAmountsNamedArray) {
    getIndependentFloatingSpeciesAmountsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getBoundarySpeciesAmountsNamedArray) {
    getBoundarySpeciesAmountsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getBoundarySpeciesConcentrationsNamedArray){
    getBoundarySpeciesConcentrationsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getRatesOfChange){
    getRatesOfChange();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getRatesOfChangeNamedArray){
    getRatesOfChangeNamedArray();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getIndependentRatesOfChange){
    getIndependentRatesOfChange();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getIndependentRatesOfChangeNamedArray){
    getIndependentRatesOfChangeNamedArray();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getDependentRatesOfChange){
    getDependentRatesOfChange();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getDependentRatesOfChangeNamedArray){
    getDependentRatesOfChangeNamedArray();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getFullJacobian){
    getFullJacobian();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getReducedJacobian){
    getReducedJacobian();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getFullEigenValues){
    getFullEigenValues();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getReducedEigenValues){
    getReducedEigenValues();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getLinkMatrix){
    getLinkMatrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getNrMatrix){
    getNrMatrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getKMatrix){
    getKMatrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getReducedStoichiometryMatrix){
    getReducedStoichiometryMatrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getFullStoichiometryMatrix){
    getFullStoichiometryMatrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getExtendedStoichiometryMatrix){
    getExtendedStoichiometryMatrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getL0Matrix){
    getL0Matrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getConservationMatrix){
    getConservationMatrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getUnscaledConcentrationControlCoefficientMatrix){
    getUnscaledConcentrationControlCoefficientMatrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getScaledConcentrationControlCoefficientMatrix){
    getScaledConcentrationControlCoefficientMatrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getUnscaledFluxControlCoefficientMatrix){
    getUnscaledFluxControlCoefficientMatrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getScaledFluxControlCoefficientMatrix){
    getScaledFluxControlCoefficientMatrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getEigenValueIds){
    getEigenValueIds();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getUnscaledParameterElasticity){
    getUnscaledParameterElasticity();
}

TEST_F(RoadRunnerAPITestsWithLLJit, DISABLED_getFrequencyResponse){
    DISABLED_getFrequencyResponse();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setConservedMoietyAnalysis){
    setConservedMoietyAnalysis();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getDiffStepSize){
    getDiffStepSize();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSteadyStateThreshold){
    getSteadyStateThreshold();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getuCC){
    getuCC();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getCC){
    getCC();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getuEE){
    getuEE();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getEE){
    getEE();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getUnscaledElasticityMatrix){
    getUnscaledElasticityMatrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getScaledElasticityMatrix){
    getScaledElasticityMatrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getScaledFloatingSpeciesElasticity){
    getScaledFloatingSpeciesElasticity();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getUnscaledSpeciesElasticity){
    getUnscaledSpeciesElasticity();
}

TEST_F(RoadRunnerAPITestsWithLLJit, addSpeciesConcentration){
    addSpeciesConcentration();
}

TEST_F(RoadRunnerAPITestsWithLLJit, addSpeciesAmount){
    addSpeciesAmount();
}

TEST_F(RoadRunnerAPITestsWithLLJit, removeSpecies){
    removeSpecies();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setBoundary){
    setBoundary();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setHasOnlySubstanceUnits){
    setHasOnlySubstanceUnits();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setInitAmount){
    setInitAmount();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setInitConcentration){
    setInitConcentration();
}

TEST_F(RoadRunnerAPITestsWithLLJit, DISABLED_setConstant){
    DISABLED_setConstant();
}

TEST_F(RoadRunnerAPITestsWithLLJit, addReactionFromSBML){
    addReactionFromSBML();
}

TEST_F(RoadRunnerAPITestsWithLLJit, addReactionDirect){
    addReactionDirect();
}

TEST_F(RoadRunnerAPITestsWithLLJit, removeReactionAndUnusedParameters){
    removeReactionAndUnusedParameters();
}

TEST_F(RoadRunnerAPITestsWithLLJit, removeReactionButNotUnusedParameters){
    removeReactionButNotUnusedParameters();
}

TEST_F(RoadRunnerAPITestsWithLLJit, DISABLED_setReversible){
    DISABLED_setReversible();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setKineticLaw){
    setKineticLaw();
}

TEST_F(RoadRunnerAPITestsWithLLJit, addParameter){
    addParameter();
}

TEST_F(RoadRunnerAPITestsWithLLJit, removeParameter){
    removeParameter();
}

TEST_F(RoadRunnerAPITestsWithLLJit, addCompartment){
    addCompartment();
}

TEST_F(RoadRunnerAPITestsWithLLJit, removeCompartment){
    removeCompartment();
}

TEST_F(RoadRunnerAPITestsWithLLJit, addAssignmentRule){
    addAssignmentRule();
}

TEST_F(RoadRunnerAPITestsWithLLJit, addRateRule){
    addRateRule();
}

TEST_F(RoadRunnerAPITestsWithLLJit, removeRules){
    removeRules();
}

TEST_F(RoadRunnerAPITestsWithLLJit, addInitialAssignment){
    addInitialAssignment();
}

TEST_F(RoadRunnerAPITestsWithLLJit, removeInitialAssignment){
    removeInitialAssignment();
}

TEST_F(RoadRunnerAPITestsWithLLJit, addEvent){
    addEvent();
}

TEST_F(RoadRunnerAPITestsWithLLJit, addTrigger){
    addTrigger();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setPersistent){
    setPersistent();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setTriggerInitialValue){
    setTriggerInitialValue();
}

TEST_F(RoadRunnerAPITestsWithLLJit, addPriority){
    addPriority();
}

TEST_F(RoadRunnerAPITestsWithLLJit, addDelay){
    addDelay();
}

TEST_F(RoadRunnerAPITestsWithLLJit, addEventAssignment){
    addEventAssignment();
}

TEST_F(RoadRunnerAPITestsWithLLJit, removeEventAssignments){
    removeEventAssignments();
}

TEST_F(RoadRunnerAPITestsWithLLJit, removeEvent){
    removeEvent();
}

TEST_F(RoadRunnerAPITestsWithLLJit, validateCurrentSBML){
    validateCurrentSBML();
}

TEST_F(RoadRunnerAPITestsWithLLJit, DISABLED_mcaSteadyState){
    DISABLED_mcaSteadyState();
}

TEST_F(RoadRunnerAPITestsWithLLJit, steadyState){
    steadyState();
}

TEST_F(RoadRunnerAPITestsWithLLJit, steadyStateNamedArray){
    steadyStateNamedArray();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSteadyStateSelections){
    getSteadyStateSelections();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setSteadyStateSelections){
    setSteadyStateSelections();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSteadyStateValues){
    getSteadyStateValues();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSteadyStateSelectionStrings){
    getSteadyStateSelectionStrings();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSteadyStateValuesNamedArray){
    getSteadyStateValuesNamedArray();
}

TEST_F(RoadRunnerAPITestsWithLLJit, regenerateModel){
    regenerateModel();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getIntegratorDefault){
    getIntegratorDefault();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSteadyStateSolver){
    getSteadyStateSolver();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSensitivitySolver){
    getSensitivitySolver();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getIntegratorByName){
    getIntegratorByName();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSteadyStateSolverByName){
    getSteadyStateSolverByName();
}

TEST_F(RoadRunnerAPITestsWithLLJit, DISABLED_getSensitivitySolverByName){
    DISABLED_getSensitivitySolverByName();
}

TEST_F(RoadRunnerAPITestsWithLLJit, makeIntegrator){
    makeIntegrator();
}

TEST_F(RoadRunnerAPITestsWithLLJit, makeSteadyStateSolver){
    makeSteadyStateSolver();
}

TEST_F(RoadRunnerAPITestsWithLLJit, DISABLED_makeSensitivitySolver){
    DISABLED_makeSensitivitySolver();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getExistingIntegratorNamesOneExists){
    getExistingIntegratorNamesOneExists();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getExistingIntegratorNamesTwoExists){
    getExistingIntegratorNamesTwoExists();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getExistingSteadyStateSolverNamesOneExists){
    getExistingSteadyStateSolverNamesOneExists();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getExistingSteadyStateSolverNamesTwoExists){
    getExistingSteadyStateSolverNamesTwoExists();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getExistingSensitivitySolverNames){
    getExistingSensitivitySolverNames();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getRegisteredIntegratorNames){
    getRegisteredIntegratorNames();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getRegisteredSteadyStateSolverNames){
    getRegisteredSteadyStateSolverNames();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getRegisteredSensitivitySolverNames){
    getRegisteredSensitivitySolverNames();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setIntegrator){
    setIntegrator();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setIntegratorToRk4){
    setIntegratorToRk4();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setIntegratorToRk45){
    setIntegratorToRk45();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setIntegratorErrorMessage){
    setIntegratorErrorMessage();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setSteadyStateSolverErrorMessage){
    setSteadyStateSolverErrorMessage();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setSensitivitySolverErrorMessage){
    setSensitivitySolverErrorMessage();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setSteadyStateSolver){
    setSteadyStateSolver();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setSensitivitySolver){
    setSensitivitySolver();
}

TEST_F(RoadRunnerAPITestsWithLLJit, integratorExistsWhenTrue){
    integratorExistsWhenTrue();
}

TEST_F(RoadRunnerAPITestsWithLLJit, steadyStateSolverExistsWhenTrue){
    steadyStateSolverExistsWhenTrue();
}

TEST_F(RoadRunnerAPITestsWithLLJit, sensitivitySolverExistsWhenTrue){
    sensitivitySolverExistsWhenTrue();
}

TEST_F(RoadRunnerAPITestsWithLLJit, integratorExistsWhenFalse){
    integratorExistsWhenFalse();
}

TEST_F(RoadRunnerAPITestsWithLLJit, steadyStateSolverExistsWhenFalse){
    steadyStateSolverExistsWhenFalse();
}

TEST_F(RoadRunnerAPITestsWithLLJit, DISABLED_sensitivitySolverExistsWhenFalse){
    DISABLED_sensitivitySolverExistsWhenFalse();
}

//TEST_F(RoadRunnerAPITestsWithLLJit, setBoundarySpeciesByIndex){
//    setBoundarySpeciesByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getNumberOfIndependentSpecies){
//    getNumberOfIndependentSpecies();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getGlobalParameterIds){
//    getGlobalParameterIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getBoundarySpeciesIds){
//    getBoundarySpeciesIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getBoundarySpeciesConcentrationIds){
//    getBoundarySpeciesConcentrationIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getBoundarySpeciesByIndex){
//    getBoundarySpeciesByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getGlobalParameterByIndex){
//    getGlobalParameterByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, setGlobalParameterByName){
//    setGlobalParameterByName();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getGlobalParameterByName){
//    getGlobalParameterByName();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getConservedMoietyValues){
//    getConservedMoietyValues();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getConservedMoietyIds){
//    getConservedMoietyIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getNumberOfReactions){
//    getNumberOfReactions();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getReactionRate){
//    getReactionRate();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getRateOfChange){
//    getRateOfChange();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getRateOfChangeIds){
//    getRateOfChangeIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getNumberOfCompartments){
//    getNumberOfCompartments();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, setCompartmentByIndex){
//    setCompartmentByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getCompartmentByIndex){
//    getCompartmentByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getCompartmentIds){
//    getCompartmentIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getNumberOfBoundarySpecies){
//    getNumberOfBoundarySpecies();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getBoundarySpeciesConcentrations){
//    getBoundarySpeciesConcentrations();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, setBoundarySpeciesConcentrations){
//    setBoundarySpeciesConcentrations();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getNumberOfFloatingSpecies){
//    getNumberOfFloatingSpecies();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getFloatingSpeciesByIndex){
//    getFloatingSpeciesByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, setFloatingSpeciesByIndex){
//    setFloatingSpeciesByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getFloatingSpeciesConcentrationsV){
//    getFloatingSpeciesConcentrationsV();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getFloatingSpeciesAmountsV){
//    getFloatingSpeciesAmountsV();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getBoundarySpeciesConcentrationsV){
//    getBoundarySpeciesConcentrationsV();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getBoundarySpeciesAmountsV){
//    getBoundarySpeciesAmountsV();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getFloatingSpeciesInitialConcentrations){
//    getFloatingSpeciesInitialConcentrations();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, setFloatingSpeciesConcentrations){
//    setFloatingSpeciesConcentrations();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, setFloatingSpeciesInitialConcentrationByIndex){
//    setFloatingSpeciesInitialConcentrationByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, setFloatingSpeciesInitialConcentrations){
//    setFloatingSpeciesInitialConcentrations();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getFloatingSpeciesIds){
//    getFloatingSpeciesIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getFloatingSpeciesInitialConditionIds){
//    getFloatingSpeciesInitialConditionIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getNumberOfGlobalParameters){
//    getNumberOfGlobalParameters();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, setGlobalParameterByIndex){
//    setGlobalParameterByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getGlobalParameterValues){
//    getGlobalParameterValues();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, evalModel){
//    evalModel();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getNumberOfDependentSpecies){
//    getNumberOfDependentSpecies();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getReactionRates){
//    getReactionRates();
//}
//
//TEST_F(RoadRunnerAPITestsWithLLJit, getReactionIds){
//    getReactionIds();
//}

