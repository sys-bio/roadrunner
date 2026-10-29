using RoadRunner
using Test


ant_str = """
    J1: 2 S1 + 3 S2 -> 5 S3 + 7 S4; v
    v = 0
    S1 = 10; S2 = 2.5; S3 = 3.4; S4 = 0
"""

rr = RoadRunner.loada(ant_str)

@testset "compartment" begin
    @test RoadRunner.getNumberOfCompartments(rr) == 1
end

@testset "reaction" begin
    @test RoadRunner.getNumberOfReactions(rr) == 1
    @test RoadRunner.getNumberOfRules(rr) == 0
    @test RoadRunner.getReactionIds(rr) == ["J1"]
end

@testset "species" begin
    @test RoadRunner.getNumberOfFloatingSpecies(rr) == 4
    @test RoadRunner.getFloatingSpeciesIds(rr) == ["S1", "S2", "S3", "S4"]
    @test RoadRunner.getNumberOfBoundarySpecies(rr) == 0
    @test RoadRunner.getBoundarySpeciesIds(rr) == []
    @test RoadRunner.getFloatingSpeciesInitialConcentrationByIndex(rr, 0) == 10
    @test RoadRunner.getFloatingSpeciesInitialConcentrationByIndex(rr, 1) == 2.5
    @test RoadRunner.getFloatingSpeciesInitialConcentrationByIndex(rr, 2) == 3.4
    @test RoadRunner.getFloatingSpeciesInitialConcentrationByIndex(rr, 3) == 0
end
