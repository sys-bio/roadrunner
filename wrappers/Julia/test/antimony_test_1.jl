using RoadRunner
using Test


ant_str = """
    S1 -> S2; k1*S1;
    k1 = 0.1; S1 = 10; S2 = 2.5
"""

rr = RoadRunner.loada(ant_str)

@testset "compartment" begin
    @test RoadRunner.getNumberOfCompartments(rr) == 1
end

@testset "reaction" begin
    @test RoadRunner.getNumberOfReactions(rr) == 1
    @test RoadRunner.getNumberOfRules(rr) == 0
    @test RoadRunner.getReactionIds(rr) == ["_J0"]
end

@testset "parameters" begin
    @test RoadRunner.getNumberOfGlobalParameters(rr) == 1
    @test RoadRunner.getGlobalParameterIds(rr) == ["k1"]
    @test RoadRunner.getGlobalParameterValues(rr) == [0.1]
end

@testset "species" begin
    @test RoadRunner.getNumberOfFloatingSpecies(rr) == 2
    @test RoadRunner.getFloatingSpeciesIds(rr) == ["S1", "S2"]
    @test RoadRunner.getNumberOfBoundarySpecies(rr) == 0
    @test RoadRunner.getBoundarySpeciesIds(rr) == []
    @test RoadRunner.getFloatingSpeciesInitialConcentrationByIndex(rr, 0) == 10
    @test RoadRunner.getFloatingSpeciesInitialConcentrationByIndex(rr, 1) == 2.5
end
