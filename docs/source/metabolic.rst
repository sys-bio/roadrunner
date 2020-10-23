**************************
Metabolic Control Analysis
**************************

.. currentmodule:: RoadRunner

Preliminaries
-------------

Metabolic control analysis is the study of how sensitive the system is to pertubrations in parameters and how those perturbations
progagate through the network. Two kinds of sensitivity are defined, system and local. The local sensitivities are
described by the elasticities. These are defined as follows:

.. math::

   \varepsilon^v_S = \frac{\partial v}{\partial S} \frac{S}{v} = \frac{\partial \ln v}{\partial \ln S}

Given a reaction rate :math:`v_i`, the elasticity describes how a given effector of the reaction step affects
the reaction rate. Because the definition is in terms of partial deriviatives, any effector that is perturbed assumes
that all other potential effectors are unchanged.

The system sensitivities are described by the control and response coefficients. These come in two forms, flux
and concentration. The flux control coefficients measures how senstive a given flux is to a perturbation in the
local rate of a reaction step. Often the local rate is perturbed by changing the enzyme concentration at the step. 
In this siutation the flux control coefficicent with respect to enzyme :math:`E_i` is defined as follow:

.. math::

   C^J_{E_i} = \frac{dJ}{dE_i} \frac{E_1}{J} = \frac{d\ln J}{d\ln E_i}

Likewise the concentration control coefficient is defined by:

.. math::

   C^S_{E_i} = \frac{dS}{d E_i} \frac{E_1}{S} = \frac{d\ln S}{d\ln E_i}

where :math:`S` is a given species. The response coefficients measure the sensitivity of a flux or species concentration
to a perturbation in some external effector. These are defined by:

.. math::
   R^J_{X} = \frac{dJ}{dX} \frac{X}{J} = \frac{d\ln J}{d\ln X}

.. math::
   R^S_{X} = \frac{dJ}{dX} \frac{X}{S} = \frac{d\ln S}{d\ln X}

where :math:`X` is the external effector. 

Methods
-------

The following methods allow users to obtain various metabolic control coefficients.

This first block of methods are for parameter independent coefficients.

.. autosummary::

   RoadRunner.getUnscaledFluxControlCoefficientMatrix
   RoadRunner.getUnscaledConcentrationControlCoefficientMatrix
   RoadRunner.getUnscaledElasticityMatrix
   RoadRunner.getUnscaledSpeciesElasticity
   RoadRunner.getScaledElasticityMatrix
   RoadRunner.getScaledFloatingSpeciesElasticity
   RoadRunner.getScaledFluxControlCoefficientMatrix
   RoadRunner.getScaledConcentrationControlCoefficientMatrix

Use these to obtain metabolic control coefficients.

.. autosummary::   
   
   RoadRunner.getuCC
   RoadRunner.getCC
   RoadRunner.getDiffStepSize
   RoadRunner.setDiffStepSize
   RoadRunner.getuEE
   RoadRunner.getEE
