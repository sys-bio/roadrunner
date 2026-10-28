using RoadRunner
using Test


ant_str = """
    const S1
    S1 -> S2; k1*S1;
    S2 -> S3; k2*S2-k3*S3
    k1 = 0.1; k2 = 0.2; k3 = 0.3
    S1 = 10; S2 = 2.5; S3 = 3.4
"""

rr = RoadRunner.loada(ant_str)


@testset "compartment" begin
    @test RoadRunner.getNumberOfCompartments(rr) == 1
end

@testset "reaction" begin
    @test RoadRunner.getNumberOfReactions(rr) == 2
    @test RoadRunner.getNumberOfRules(rr) == 0
    @test RoadRunner.getReactionIds(rr) == ["_J0", "_J1"]
end

@testset "parameters" begin
    @test RoadRunner.getNumberOfGlobalParameters(rr) == 3
    @test RoadRunner.getGlobalParameterIds(rr) == ["k1", "k2", "k3"]
    @test RoadRunner.getGlobalParameterValues(rr) == [0.1, 0.2, 0.3]
end

@testset "species" begin
    @test RoadRunner.getNumberOfFloatingSpecies(rr) == 2
    @test RoadRunner.getFloatingSpeciesIds(rr) == ["S2", "S3"]
    @test RoadRunner.getNumberOfBoundarySpecies(rr) == 1
    @test RoadRunner.getBoundarySpeciesIds(rr) == ["S1"]
    @test RoadRunner.getFloatingSpeciesInitialConcentrationByIndex(rr, 0) == 2.5
    @test RoadRunner.getFloatingSpeciesInitialConcentrationByIndex(rr, 1) == 3.4
end
