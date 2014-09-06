#!/usr/bin/perl
# -*-Perl-*-
#  
#  @file    example1.pl
#  @brief   SBML FBC example
#  @author  Frank Bergmann
#  
# <!--------------------------------------------------------------------------
# This sample program is distributed under a different license than the rest
# of libSBML.  This program uses the open-source MIT license, as follows:
#
# Copyright (c) 2013-2014 by the California Institute of Technology
# (California, USA), the European Bioinformatics Institute (EMBL-EBI, UK)
# and the University of Heidelberg (Germany), with support from the National
# Institutes of Health (USA) under grant R01GM070923.  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Neither the name of the California Institute of Technology (Caltech), nor
# of the European Bioinformatics Institute (EMBL-EBI), nor of the University
# of Heidelberg, nor the names of any contributors, may be used to endorse
# or promote products derived from this software without specific prior
# written permission.
# ------------------------------------------------------------------------ -->
#  

use LibSBML;
use strict;

my $sbmlns = new LibSBML::FbcPkgNamespaces(3,1);
my $document = new LibSBML::SBMLDocument($sbmlns);

#  create the Model

my $model=$document->createModel();

#  create the Compartment

my $compartment = $model->createCompartment();
$compartment->setId("compartment");
$compartment->setConstant(1);
$compartment->setSize(1);

#  create the Species

my $species = $model->createSpecies();
$species->setId("Node1");
$species->setCompartment("compartment");
$species->setBoundaryCondition(0);
$species->setConstant(0);
$species->setHasOnlySubstanceUnits(0);

$species = $model->createSpecies();
$species->setId("Node2");
$species->setCompartment("compartment");
$species->setBoundaryCondition(0);
$species->setConstant(0);
$species->setHasOnlySubstanceUnits(0);

$species = $model->createSpecies();
$species->setId("Node3");
$species->setCompartment("compartment");
$species->setBoundaryCondition(0);
$species->setConstant(0);
$species->setHasOnlySubstanceUnits(0);

$species = $model->createSpecies();
$species->setId("Node4");
$species->setCompartment("compartment");
$species->setBoundaryCondition(0);
$species->setConstant(0);
$species->setHasOnlySubstanceUnits(0);

$species = $model->createSpecies();
$species->setId("Node5");
$species->setCompartment("compartment");
$species->setBoundaryCondition(0);
$species->setConstant(0);
$species->setHasOnlySubstanceUnits(0);

$species = $model->createSpecies();
$species->setId("Node6");
$species->setCompartment("compartment");
$species->setBoundaryCondition(0);
$species->setConstant(0);
$species->setHasOnlySubstanceUnits(0);

$species = $model->createSpecies();
$species->setId("Node7");
$species->setCompartment("compartment");
$species->setBoundaryCondition(0);
$species->setConstant(0);
$species->setHasOnlySubstanceUnits(0);

$species = $model->createSpecies();
$species->setId("Node8");
$species->setCompartment("compartment");
$species->setBoundaryCondition(0);
$species->setConstant(0);
$species->setHasOnlySubstanceUnits(0);

$species = $model->createSpecies();
$species->setId("Node0");
$species->setCompartment("compartment");
$species->setBoundaryCondition(1);
$species->setConstant(0);
$species->setHasOnlySubstanceUnits(0);

$species = $model->createSpecies();
$species->setId("Node9");
$species->setCompartment("compartment");
$species->setBoundaryCondition(1);
$species->setConstant(0);
$species->setHasOnlySubstanceUnits(0);

# create reactions

my $reaction = $model->createReaction();
$reaction->setId("J0");
$reaction->setReversible(0);
$reaction->setFast(0);
my $reactant = $reaction->createReactant();
$reactant->setSpecies("Node0");
$reactant->setStoichiometry(1);
$reactant->setConstant(1);
my $product = $reaction->createProduct();
$product->setSpecies("Node1");
$product->setStoichiometry(1);
$product->setConstant(1);

$reaction = $model->createReaction();
$reaction->setId("J1");
$reaction->setReversible(0);
$reaction->setFast(0);
$reactant = $reaction->createReactant();
$reactant->setSpecies("Node1");
$reactant->setStoichiometry(1);
$reactant->setConstant(1);
$product = $reaction->createProduct();
$product->setSpecies("Node2");
$product->setStoichiometry(1);
$product->setConstant(1);

$reaction = $model->createReaction();
$reaction->setId("J2");
$reaction->setReversible(0);
$reaction->setFast(0);
$reactant = $reaction->createReactant();
$reactant->setSpecies("Node2");
$reactant->setStoichiometry(1);
$reactant->setConstant(1);
$product = $reaction->createProduct();
$product->setSpecies("Node3");
$product->setStoichiometry(1);
$product->setConstant(1);

$reaction = $model->createReaction();
$reaction->setId("J3");
$reaction->setReversible(0);
$reaction->setFast(0);
$reactant = $reaction->createReactant();
$reactant->setSpecies("Node1");
$reactant->setStoichiometry(1);
$reactant->setConstant(1);
$product = $reaction->createProduct();
$product->setSpecies("Node4");
$product->setStoichiometry(1);
$product->setConstant(1);

$reaction = $model->createReaction();
$reaction->setId("J4");
$reaction->setReversible(0);
$reaction->setFast(0);
$reactant = $reaction->createReactant();
$reactant->setSpecies("Node4");
$reactant->setStoichiometry(1);
$reactant->setConstant(1);
$product = $reaction->createProduct();
$product->setSpecies("Node3");
$product->setStoichiometry(1);
$product->setConstant(1);

$reaction = $model->createReaction();
$reaction->setId("J5");
$reaction->setReversible(0);
$reaction->setFast(0);
$reactant = $reaction->createReactant();
$reactant->setSpecies("Node3");
$reactant->setStoichiometry(1);
$reactant->setConstant(1);
$product = $reaction->createProduct();
$product->setSpecies("Node5");
$product->setStoichiometry(1);
$product->setConstant(1);

$reaction = $model->createReaction();
$reaction->setId("J6");
$reaction->setReversible(0);
$reaction->setFast(0);
$reactant = $reaction->createReactant();
$reactant->setSpecies("Node5");
$reactant->setStoichiometry(1);
$reactant->setConstant(1);
$product = $reaction->createProduct();
$product->setSpecies("Node6");
$product->setStoichiometry(1);
$product->setConstant(1);

$reaction = $model->createReaction();
$reaction->setId("J7");
$reaction->setReversible(0);
$reaction->setFast(0);
$reactant = $reaction->createReactant();
$reactant->setSpecies("Node6");
$reactant->setStoichiometry(1);
$reactant->setConstant(1);
$product = $reaction->createProduct();
$product->setSpecies("Node7");
$product->setStoichiometry(1);
$product->setConstant(1);

$reaction = $model->createReaction();
$reaction->setId("J8");
$reaction->setReversible(0);
$reaction->setFast(0);
$reactant = $reaction->createReactant();
$reactant->setSpecies("Node5");
$reactant->setStoichiometry(1);
$reactant->setConstant(1);
$product = $reaction->createProduct();
$product->setSpecies("Node8");
$product->setStoichiometry(1);
$product->setConstant(1);

$reaction = $model->createReaction();
$reaction->setId("J9");
$reaction->setReversible(0);
$reaction->setFast(0);
$reactant = $reaction->createReactant();
$reactant->setSpecies("Node8");
$reactant->setStoichiometry(1);
$reactant->setConstant(1);
$product = $reaction->createProduct();
$product->setSpecies("Node7");
$product->setStoichiometry(1);
$product->setConstant(1);

$reaction = $model->createReaction();
$reaction->setId("J10");
$reaction->setReversible(0);
$reaction->setFast(0);
$reactant = $reaction->createReactant();
$reactant->setSpecies("Node7");
$reactant->setStoichiometry(1);
$reactant->setConstant(1);
$product = $reaction->createProduct();
$product->setSpecies("Node9");
$product->setStoichiometry(1);
$product->setConstant(1);

# add FBC constraints

my $mplugin = $model->getPlugin("fbc");


my $bound= $mplugin->createFluxBound();
$bound->setId("bound1");
$bound->setReaction("J0");
$bound->setOperation("equal");
$bound->setValue(10);

my $objective = $mplugin->createObjective();
$objective->setId("obj1");
$objective->setType("maximize");

$mplugin->setActiveObjectiveId("obj1");


my $fluxObjective = objective->createFluxObjective();
$fluxObjective->setReaction("J8");
$fluxObjective->setCoefficient(1);

LibSBML::writeSBML($document,"fbc_example1->xml");
