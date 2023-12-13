//
// Created by Ciaran on 07/05/2021.
//

#include "RoadRunnerAPITests.h"
#include "rrConfig.h"

using namespace rr;

class RoadRunnerAPITestsWithMCJit : public RoadRunnerAPITests {
public:
    RoadRunnerAPITestsWithMCJit() 
        : RoadRunnerAPITests(){
        //Logger::setLevel(Logger::LOG_WARNING);
        Config::setValue(Config::LLVM_BACKEND, Config::LLVM_BACKEND_VALUES::MCJIT);
    }
};



TEST_F(RoadRunnerAPITestsWithMCJit, DefaultJacobianMode){
    DefaultJacobianMode();
}

TEST_F(RoadRunnerAPITestsWithMCJit, SetJacobianModeToAmt){
    SetJacobianModeToAmt();
}

TEST_F(RoadRunnerAPITestsWithMCJit, GetFullJacobianDefaultConfigSettings){
    GetFullJacobianDefaultConfigSettings();
}

TEST_F(RoadRunnerAPITestsWithMCJit, GetFullJacobianUsingConcMode){
    GetFullJacobianUsingConcMode();
}

TEST_F(RoadRunnerAPITestsWithMCJit, GetFullJacobianUsingAmtMode){
    GetFullJacobianUsingAmtMode();
}

TEST_F(RoadRunnerAPITestsWithMCJit, GetFullJacobianUsingAmtModeAsLong){
        GetFullJacobianUsingAmtModeAsLong();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getIds){
    getIds();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getIdsAfterRegenerate){
    getIdsAfterRegenerate();
}

TEST_F(RoadRunnerAPITestsWithMCJit, DISABLED_getIdsAfterSetConservedMoiety){
    getIdsAfterSetConservedMoiety();
}

TEST_F(RoadRunnerAPITestsWithMCJit, LoadModelWithOnlyLayoutInformation){
    LoadModelWithOnlyLayoutInformation();
}

TEST_F(RoadRunnerAPITestsWithMCJit, loadSBMLWithLeadingNewlineCharacter){
    loadSBMLWithLeadingNewlineCharacter();
}

TEST_F(RoadRunnerAPITestsWithMCJit, RoadRunnerConstructorVersion){
    RoadRunnerConstructorVersion();
}

TEST_F(RoadRunnerAPITestsWithMCJit, DISABLED_RoadRunnerConstructorFromURL){
    DISABLED_RoadRunnerConstructorFromURL();
}

TEST_F(RoadRunnerAPITestsWithMCJit, DISABLED_CompilerTempDirSupportCodeDir){
    DISABLED_CompilerTempDirSupportCodeDir();
}

TEST_F(RoadRunnerAPITestsWithMCJit, CopyConstructor){
    CopyConstructor();
}

TEST_F(RoadRunnerAPITestsWithMCJit, DISABLED_EqualityOperators){
    DISABLED_EqualityOperators();
}

TEST_F(RoadRunnerAPITestsWithMCJit, DISABLED_InequalityOperators){
    DISABLED_InequalityOperators();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getInstanceID){
    getInstanceID();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getInstanceCount){
    getInstanceCount();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getParamPromotedSBML){
    getParamPromotedSBML();
}

TEST_F(RoadRunnerAPITestsWithMCJit, DISABLED_getCompiler){
    DISABLED_getCompiler();
}

TEST_F(RoadRunnerAPITestsWithMCJit, IsModelLoadedWhenFalse){
    IsModelLoadedWhenFalse();
}

TEST_F(RoadRunnerAPITestsWithMCJit, IsModelLoadedWhenTrue){
    IsModelLoadedWhenTrue();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getConservedMoietyAnalysisWhenFalse){
    getConservedMoietyAnalysisWhenFalse();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getConservedMoietyAnalysisWhenTrue){
    getConservedMoietyAnalysisWhenTrue();
}

TEST_F(RoadRunnerAPITestsWithMCJit, tmpDir){
    tmpDir();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getModelName){
    getModelName();
}

TEST_F(RoadRunnerAPITestsWithMCJit, clearModel){
    clearModel();
}

TEST_F(RoadRunnerAPITestsWithMCJit, oneStep){
    oneStep();
}

TEST_F(RoadRunnerAPITestsWithMCJit, internalOneStep){
    internalOneStep();
}

TEST_F(RoadRunnerAPITestsWithMCJit, saveState){
    saveState();
}

TEST_F(RoadRunnerAPITestsWithMCJit, loadState){
    loadState();
}

TEST_F(RoadRunnerAPITestsWithMCJit, loadStateFromCopy){
    loadStateFromCopy();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getSimulationData){
    getSimulationData();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setSimulateOptions){
    setSimulateOptions();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getSBML){
    getSBML();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getCurrentSBML){
    getCurrentSBML();
}

TEST_F(RoadRunnerAPITestsWithMCJit, reset){
    reset();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getCurrentTime){
    getCurrentTime();
}

TEST_F(RoadRunnerAPITestsWithMCJit, resetSelectionLists){
    resetSelectionLists();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getSelections){
    getSelections();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getValue){
    getValue();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setSelections){
    setSelections();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getSelectedValues){
    getSelectedValues();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getIndependentFloatingSpeciesIds){
    getIndependentFloatingSpeciesIds();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getDependentFloatingSpeciesIds){
    getDependentFloatingSpeciesIds();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getFloatingSpeciesInitialConcentrations){
    getFloatingSpeciesInitialConcentrations();
}

TEST_F(RoadRunnerAPITestsWithMCJit, DISABLED_getSupportedIdTypes){
    DISABLED_getSupportedIdTypes();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setValue){
    setValue();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getFloatingSpeciesAmountsNamedArray){
    getFloatingSpeciesAmountsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getFloatingSpeciesConcentrationsNamedArray){
    getFloatingSpeciesConcentrationsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getDependentFloatingSpeciesConcentrations) {
    getDependentFloatingSpeciesConcentrations();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getDependentFloatingSpeciesAmounts) {
    getDependentFloatingSpeciesAmounts();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getDependentFloatingSpeciesConcentrationsNamedArray) {
    getDependentFloatingSpeciesConcentrationsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getDependentFloatingSpeciesAmountsNamedArray) {
    getDependentFloatingSpeciesAmountsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getIndependentFloatingSpeciesConcentrations) {
    getIndependentFloatingSpeciesConcentrations();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getIndependentFloatingSpeciesAmounts) {
    getIndependentFloatingSpeciesAmounts();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getIndependentFloatingSpeciesConcentrationsNamedArray) {
    getIndependentFloatingSpeciesConcentrationsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getIndependentFloatingSpeciesAmountsNamedArray) {
    getIndependentFloatingSpeciesAmountsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getBoundarySpeciesAmountsNamedArray){
    getBoundarySpeciesAmountsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getBoundarySpeciesConcentrationsNamedArray){
    getBoundarySpeciesConcentrationsNamedArray();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getRatesOfChange){
    getRatesOfChange();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getRatesOfChangeNamedArray){
    getRatesOfChangeNamedArray();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getIndependentRatesOfChange){
    getIndependentRatesOfChange();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getIndependentRatesOfChangeNamedArray){
    getIndependentRatesOfChangeNamedArray();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getDependentRatesOfChange){
    getDependentRatesOfChange();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getDependentRatesOfChangeNamedArray){
    getDependentRatesOfChangeNamedArray();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getFullJacobian){
    getFullJacobian();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getReducedJacobian){
    getReducedJacobian();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getFullEigenValues){
    getFullEigenValues();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getReducedEigenValues){
    getReducedEigenValues();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getLinkMatrix){
    getLinkMatrix();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getNrMatrix){
    getNrMatrix();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getKMatrix){
    getKMatrix();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getReducedStoichiometryMatrix){
    getReducedStoichiometryMatrix();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getFullStoichiometryMatrix){
    getFullStoichiometryMatrix();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getExtendedStoichiometryMatrix){
    getExtendedStoichiometryMatrix();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getL0Matrix){
    getL0Matrix();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getConservationMatrix){
    getConservationMatrix();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getUnscaledConcentrationControlCoefficientMatrix){
    getUnscaledConcentrationControlCoefficientMatrix();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getScaledConcentrationControlCoefficientMatrix){
    getScaledConcentrationControlCoefficientMatrix();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getUnscaledFluxControlCoefficientMatrix){
    getUnscaledFluxControlCoefficientMatrix();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getScaledFluxControlCoefficientMatrix){
    getScaledFluxControlCoefficientMatrix();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getEigenValueIds){
    getEigenValueIds();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getUnscaledParameterElasticity){
    getUnscaledParameterElasticity();
}

TEST_F(RoadRunnerAPITestsWithMCJit, DISABLED_getFrequencyResponse){
    DISABLED_getFrequencyResponse();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setConservedMoietyAnalysis){
    setConservedMoietyAnalysis();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getDiffStepSize){
    getDiffStepSize();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getSteadyStateThreshold){
    getSteadyStateThreshold();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getuCC){
    getuCC();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getCC){
    getCC();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getuEE){
    getuEE();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getEE){
    getEE();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getUnscaledElasticityMatrix){
    getUnscaledElasticityMatrix();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getScaledElasticityMatrix){
    getScaledElasticityMatrix();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getScaledFloatingSpeciesElasticity){
    getScaledFloatingSpeciesElasticity();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getUnscaledSpeciesElasticity){
    getUnscaledSpeciesElasticity();
}

TEST_F(RoadRunnerAPITestsWithMCJit, addSpeciesConcentration){
    addSpeciesConcentration();
}

TEST_F(RoadRunnerAPITestsWithMCJit, addSpeciesAmount){
    addSpeciesAmount();
}

TEST_F(RoadRunnerAPITestsWithMCJit, removeSpecies){
    removeSpecies();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setBoundary){
    setBoundary();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setHasOnlySubstanceUnits){
    setHasOnlySubstanceUnits();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setInitAmount){
    setInitAmount();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setInitConcentration){
    setInitConcentration();
}

TEST_F(RoadRunnerAPITestsWithMCJit, DISABLED_setConstant){
    DISABLED_setConstant();
}

TEST_F(RoadRunnerAPITestsWithMCJit, addReactionFromSBML){
    addReactionFromSBML();
}

TEST_F(RoadRunnerAPITestsWithMCJit, addReactionDirect){
    addReactionDirect();
}

TEST_F(RoadRunnerAPITestsWithMCJit, removeReactionAndUnusedParameters){
    removeReactionAndUnusedParameters();
}

TEST_F(RoadRunnerAPITestsWithMCJit, removeReactionButNotUnusedParameters){
    removeReactionButNotUnusedParameters();
}

TEST_F(RoadRunnerAPITestsWithMCJit, DISABLED_setReversible){
    DISABLED_setReversible();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setKineticLaw){
    setKineticLaw();
}

TEST_F(RoadRunnerAPITestsWithMCJit, addParameter){
    addParameter();
}

TEST_F(RoadRunnerAPITestsWithMCJit, removeParameter){
    removeParameter();
}

TEST_F(RoadRunnerAPITestsWithMCJit, addCompartment){
    addCompartment();
}

TEST_F(RoadRunnerAPITestsWithMCJit, removeCompartment){
    removeCompartment();
}

TEST_F(RoadRunnerAPITestsWithMCJit, addAssignmentRule){
    addAssignmentRule();
}

TEST_F(RoadRunnerAPITestsWithMCJit, addRateRule){
    addRateRule();
}

TEST_F(RoadRunnerAPITestsWithMCJit, removeRules){
    removeRules();
}

TEST_F(RoadRunnerAPITestsWithMCJit, addInitialAssignment){
    addInitialAssignment();
}

TEST_F(RoadRunnerAPITestsWithMCJit, removeInitialAssignment){
    removeInitialAssignment();
}

TEST_F(RoadRunnerAPITestsWithMCJit, addEvent){
    addEvent();
}

TEST_F(RoadRunnerAPITestsWithMCJit, addTrigger){
    addTrigger();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setPersistent){
    setPersistent();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setTriggerInitialValue){
    setTriggerInitialValue();
}

TEST_F(RoadRunnerAPITestsWithMCJit, addPriority){
    addPriority();
}

TEST_F(RoadRunnerAPITestsWithMCJit, addDelay){
    addDelay();
}

TEST_F(RoadRunnerAPITestsWithMCJit, addEventAssignment){
    addEventAssignment();
}

TEST_F(RoadRunnerAPITestsWithMCJit, removeEventAssignments){
    removeEventAssignments();
}

TEST_F(RoadRunnerAPITestsWithMCJit, removeEvent){
    removeEvent();
}

TEST_F(RoadRunnerAPITestsWithMCJit, validateCurrentSBML){
    validateCurrentSBML();
}

TEST_F(RoadRunnerAPITestsWithMCJit, DISABLED_mcaSteadyState){
    DISABLED_mcaSteadyState();
}

TEST_F(RoadRunnerAPITestsWithMCJit, steadyState){
    steadyState();
}

TEST_F(RoadRunnerAPITestsWithMCJit, steadyStateNamedArray){
    steadyStateNamedArray();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getSteadyStateSelections){
    getSteadyStateSelections();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setSteadyStateSelections){
    setSteadyStateSelections();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getSteadyStateValues){
    getSteadyStateValues();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getSteadyStateSelectionStrings){
    getSteadyStateSelectionStrings();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getSteadyStateValuesNamedArray){
    getSteadyStateValuesNamedArray();
}

TEST_F(RoadRunnerAPITestsWithMCJit, regenerateModel){
    regenerateModel();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getIntegratorDefault){
    getIntegratorDefault();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getSteadyStateSolver){
    getSteadyStateSolver();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getSensitivitySolver){
    getSensitivitySolver();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getIntegratorByName){
    getIntegratorByName();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getSteadyStateSolverByName){
    getSteadyStateSolverByName();
}

TEST_F(RoadRunnerAPITestsWithMCJit, DISABLED_getSensitivitySolverByName){
    DISABLED_getSensitivitySolverByName();
}

TEST_F(RoadRunnerAPITestsWithMCJit, makeIntegrator){
    makeIntegrator();
}

TEST_F(RoadRunnerAPITestsWithMCJit, makeSteadyStateSolver){
    makeSteadyStateSolver();
}

TEST_F(RoadRunnerAPITestsWithMCJit, DISABLED_makeSensitivitySolver){
    DISABLED_makeSensitivitySolver();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getExistingIntegratorNamesOneExists){
    getExistingIntegratorNamesOneExists();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getExistingIntegratorNamesTwoExists){
    getExistingIntegratorNamesTwoExists();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getExistingSteadyStateSolverNamesOneExists){
    getExistingSteadyStateSolverNamesOneExists();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getExistingSteadyStateSolverNamesTwoExists){
    getExistingSteadyStateSolverNamesTwoExists();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getExistingSensitivitySolverNames){
    getExistingSensitivitySolverNames();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getRegisteredIntegratorNames){
    getRegisteredIntegratorNames();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getRegisteredSteadyStateSolverNames){
    getRegisteredSteadyStateSolverNames();
}

TEST_F(RoadRunnerAPITestsWithMCJit, getRegisteredSensitivitySolverNames){
    getRegisteredSensitivitySolverNames();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setIntegrator){
    setIntegrator();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setIntegratorToRk4){
    setIntegratorToRk4();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setIntegratorToRk45){
    setIntegratorToRk45();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setIntegratorErrorMessage){
    setIntegratorErrorMessage();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setSteadyStateSolverErrorMessage){
    setSteadyStateSolverErrorMessage();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setSensitivitySolverErrorMessage){
    setSensitivitySolverErrorMessage();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setSteadyStateSolver){
    setSteadyStateSolver();
}

TEST_F(RoadRunnerAPITestsWithMCJit, setSensitivitySolver){
    setSensitivitySolver();
}

TEST_F(RoadRunnerAPITestsWithMCJit, integratorExistsWhenTrue){
    integratorExistsWhenTrue();
}

TEST_F(RoadRunnerAPITestsWithMCJit, steadyStateSolverExistsWhenTrue){
    steadyStateSolverExistsWhenTrue();
}

TEST_F(RoadRunnerAPITestsWithMCJit, sensitivitySolverExistsWhenTrue){
    sensitivitySolverExistsWhenTrue();
}

TEST_F(RoadRunnerAPITestsWithMCJit, integratorExistsWhenFalse){
    integratorExistsWhenFalse();
}

TEST_F(RoadRunnerAPITestsWithMCJit, steadyStateSolverExistsWhenFalse){
    steadyStateSolverExistsWhenFalse();
}

TEST_F(RoadRunnerAPITestsWithMCJit, DISABLED_sensitivitySolverExistsWhenFalse){
    DISABLED_sensitivitySolverExistsWhenFalse();
}

//TEST_F(RoadRunnerAPITestsWithMCJit, setBoundarySpeciesByIndex){
//    setBoundarySpeciesByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getNumberOfIndependentSpecies){
//    getNumberOfIndependentSpecies();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getGlobalParameterIds){
//    getGlobalParameterIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getBoundarySpeciesIds){
//    getBoundarySpeciesIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getBoundarySpeciesConcentrationIds){
//    getBoundarySpeciesConcentrationIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getBoundarySpeciesByIndex){
//    getBoundarySpeciesByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getGlobalParameterByIndex){
//    getGlobalParameterByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, setGlobalParameterByName){
//    setGlobalParameterByName();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getGlobalParameterByName){
//    getGlobalParameterByName();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getConservedMoietyValues){
//    getConservedMoietyValues();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getConservedMoietyIds){
//    getConservedMoietyIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getNumberOfReactions){
//    getNumberOfReactions();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getReactionRate){
//    getReactionRate();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getRateOfChange){
//    getRateOfChange();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getRateOfChangeIds){
//    getRateOfChangeIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getNumberOfCompartments){
//    getNumberOfCompartments();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, setCompartmentByIndex){
//    setCompartmentByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getCompartmentByIndex){
//    getCompartmentByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getCompartmentIds){
//    getCompartmentIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getNumberOfBoundarySpecies){
//    getNumberOfBoundarySpecies();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getBoundarySpeciesConcentrations){
//    getBoundarySpeciesConcentrations();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, setBoundarySpeciesConcentrations){
//    setBoundarySpeciesConcentrations();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getNumberOfFloatingSpecies){
//    getNumberOfFloatingSpecies();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getFloatingSpeciesByIndex){
//    getFloatingSpeciesByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, setFloatingSpeciesByIndex){
//    setFloatingSpeciesByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getFloatingSpeciesConcentrationsV){
//    getFloatingSpeciesConcentrationsV();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getFloatingSpeciesAmountsV){
//    getFloatingSpeciesAmountsV();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getBoundarySpeciesConcentrationsV){
//    getBoundarySpeciesConcentrationsV();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getBoundarySpeciesAmountsV){
//    getBoundarySpeciesAmountsV();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getFloatingSpeciesInitialConcentrations){
//    getFloatingSpeciesInitialConcentrations();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, setFloatingSpeciesConcentrations){
//    setFloatingSpeciesConcentrations();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, setFloatingSpeciesInitialConcentrationByIndex){
//    setFloatingSpeciesInitialConcentrationByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, setFloatingSpeciesInitialConcentrations){
//    setFloatingSpeciesInitialConcentrations();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getFloatingSpeciesIds){
//    getFloatingSpeciesIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getFloatingSpeciesInitialConditionIds){
//    getFloatingSpeciesInitialConditionIds();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getNumberOfGlobalParameters){
//    getNumberOfGlobalParameters();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, setGlobalParameterByIndex){
//    setGlobalParameterByIndex();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getGlobalParameterValues){
//    getGlobalParameterValues();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, evalModel){
//    evalModel();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getNumberOfDependentSpecies){
//    getNumberOfDependentSpecies();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getReactionRates){
//    getReactionRates();
//}
//
//TEST_F(RoadRunnerAPITestsWithMCJit, getReactionIds){
//    getReactionIds();
//}




























