using RoadRunner
using Test


ant_str = """
    const Xo, X1
    Xo -> S1; k1*Xo - k2*S1
    S1 -> S2; k3*S2
    S2 -> X1; k4*S2

    Xo = 1;   X1 = 0
    S1 = 0;   S2 = 0
    k1 = 0.1; k2 = 0.56
    k3 = 1.2; k4 = 0.9
"""

rr = RoadRunner.loada(ant_str)


@testset "parameters" begin
    @test RoadRunner.getNumberOfGlobalParameters(rr) == 4
    @test RoadRunner.getGlobalParameterIds(rr) == ["k1", "k2", "k3", "k4"]
end

@testset "species" begin
    @test RoadRunner.getNumberOfFloatingSpecies(rr) == 2
    @test RoadRunner.getFloatingSpeciesIds(rr) == ["S1", "S2"]
    @test RoadRunner.getNumberOfBoundarySpecies(rr) == 2
    @test RoadRunner.getBoundarySpeciesIds(rr) == ["Xo", "X1"]
end

@testset "steadystates" begin
    @test RoadRunner.steadyState(rr) < 1e-6
end
