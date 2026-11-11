//
// Created by Ciaran on 07/05/2021.
//

#include "RoadRunnerAPITests.h"
#include "rrConfig.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

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

TEST_F(RoadRunnerAPITestsWithLLJit, getSelectedReactionRates){
    getSelectedReactionRates();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSelectedAmountRates){
    getSelectedAmountRates();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSelectedRateRuleRates){
    getSelectedRateRuleRates();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getSteadyStateValuesWithRates){
    getSteadyStateValuesWithRates();
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

TEST_F(RoadRunnerAPITestsWithLLJit, getReactantsStoichiometryMatrix) {
    getReactantsStoichiometryMatrix();
}

TEST_F(RoadRunnerAPITestsWithLLJit, getProductsStoichiometryMatrix) {
    getProductsStoichiometryMatrix();
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

TEST_F(RoadRunnerAPITestsWithLLJit, setInitAmountWithSetValue) {
    setInitAmountWithSetValue();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setInitConcentration){
    setInitConcentration();
}

TEST_F(RoadRunnerAPITestsWithLLJit, setInitConcentrationWithSetValue) {
    setInitConcentrationWithSetValue();
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


/**
 * Chains of initial assignments.
 *
 * Every parameter p_k reads the previous one twice, so inlining the code of an
 * initial assignment into each reader makes the generated code grow as 2^k.
 * The initial value resolvers cache one value per symbol, which keeps it linear.
 * Optionally, one link of the chain is an assignment rule instead.
 */
namespace {

    constexpr int ChainLevels = 40;

    // p_k = p_(k-1) + piecewise(p_(k-1), AGE < 50, 2*p_(k-1))
    std::string chainMath(int k) {
        std::string p = "<ci> p" + std::to_string(k - 1) + " </ci>";
        return "<math xmlns='http://www.w3.org/1998/Math/MathML'><apply><plus/>" + p +
               "<piecewise><piece>" + p + "<apply><lt/><ci> AGE </ci><cn> 50 </cn></apply></piece>"
               "<otherwise><apply><times/><cn> 2 </cn>" + p + "</apply></otherwise>"
               "</piecewise></apply></math>";
    }

    // ruleLevel is the link of the chain given by an assignment rule, 0 for none.
    // S reads the last two links and the rule of q reads S and p(ChainLevels - 2).
    std::string chainSBML(int ruleLevel) {
        std::string params = "<parameter id='AGE' value='35' constant='true'/>"
                             "<parameter id='p0' value='1' constant='true'/>";
        std::string assignments;
        std::string rules;
        for (int k = 1; k <= ChainLevels; k++) {
            std::string id = "p" + std::to_string(k);
            if (k == ruleLevel) {
                params += "<parameter id='" + id + "' constant='false'/>";
                rules += "<assignmentRule variable='" + id + "'>" + chainMath(k) + "</assignmentRule>";
            } else {
                params += "<parameter id='" + id + "' constant='true'/>";
                assignments += "<initialAssignment symbol='" + id + "'>" + chainMath(k) +
                               "</initialAssignment>";
            }
        }
        params += "<parameter id='q' constant='false'/>";
        assignments += "<initialAssignment symbol='S'>"
                       "<math xmlns='http://www.w3.org/1998/Math/MathML'><apply><divide/><ci> p" +
                       std::to_string(ChainLevels) + " </ci><ci> p" + std::to_string(ChainLevels - 1) +
                       " </ci></apply></math></initialAssignment>";
        rules += "<assignmentRule variable='q'>"
                 "<math xmlns='http://www.w3.org/1998/Math/MathML'><apply><plus/><ci> S </ci><ci> p" +
                 std::to_string(ChainLevels - 2) + " </ci></apply></math></assignmentRule>";
        return "<?xml version='1.0' encoding='UTF-8'?>"
               "<sbml xmlns='http://www.sbml.org/sbml/level3/version2/core' level='3' version='2'>"
               "<model id='chain'>"
               "<listOfCompartments><compartment id='c' size='1' constant='true' "
               "spatialDimensions='3'/></listOfCompartments>"
               "<listOfSpecies><species id='S' compartment='c' hasOnlySubstanceUnits='false' "
               "boundaryCondition='false' constant='false'/></listOfSpecies>"
               "<listOfParameters>" + params + "</listOfParameters>"
               "<listOfInitialAssignments>" + assignments + "</listOfInitialAssignments>"
               "<listOfRules>" + rules + "</listOfRules></model></sbml>";
    }

    // Without the cache the code generator does not finish, and a running
    // compilation cannot be cancelled, so a timeout ends the process.
    std::unique_ptr<RoadRunner> loadChain(int ruleLevel, std::chrono::seconds limit) {
        // the initial value resolvers cache whether or not this is set
        Config::setValue(Config::LLVM_SYMBOL_CACHE, false);
        auto promise = std::make_shared<std::promise<std::unique_ptr<RoadRunner>>>();
        std::future<std::unique_ptr<RoadRunner>> loaded = promise->get_future();
        std::string sbml = chainSBML(ruleLevel);
        std::thread([promise, sbml]() {
            try {
                promise->set_value(std::make_unique<RoadRunner>(sbml));
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }).detach();
        if (loaded.wait_for(limit) != std::future_status::ready) {
            std::cerr << "Loading a chain of " << ChainLevels << " initial assignments took more than "
                      << limit.count() << " s; the initial value code is not generated once per symbol."
                      << std::endl;
            std::_Exit(EXIT_FAILURE);
        }
        return loaded.get();
    }

    // Every link is the previous one times branch; S is the last ratio.
    void checkChain(RoadRunner& rr, double branch) {
        for (int k : {10, 20, 21, ChainLevels}) {
            EXPECT_NEAR(rr.getValue("p" + std::to_string(k)) / std::pow(branch, k), 1.0, 1e-12) << "p" << k;
        }
        EXPECT_NEAR(rr.getValue("S"), branch, 1e-12);
        EXPECT_NEAR(rr.getValue("init([S])"), branch, 1e-12);
        EXPECT_NEAR(rr.getValue("q") / (branch + std::pow(branch, ChainLevels - 2)), 1.0, 1e-12);
    }

    const std::chrono::seconds ChainTimeout(60);
}

TEST_F(RoadRunnerAPITestsWithLLJit, ChainOfInitialAssignments) {
    auto rr = loadChain(0, ChainTimeout);
    checkChain(*rr, 2.0);
}

TEST_F(RoadRunnerAPITestsWithLLJit, ChainOfInitialAssignmentsChangedInitialValue) {
    auto rr = loadChain(0, ChainTimeout);
    rr->setValue("init(AGE)", 60.0);
    rr->reset();
    checkChain(*rr, 3.0);
}

TEST_F(RoadRunnerAPITestsWithLLJit, ChainOfInitialAssignmentsWithAssignmentRuleInMiddle) {
    auto rr = loadChain(20, ChainTimeout);
    checkChain(*rr, 2.0);
}

TEST_F(RoadRunnerAPITestsWithLLJit, ChainOfInitialAssignmentsWithAssignmentRuleInMiddleChangedInitialValue) {
    auto rr = loadChain(20, ChainTimeout);
    rr->setValue("init(AGE)", 60.0);
    rr->reset();
    checkChain(*rr, 3.0);
}
