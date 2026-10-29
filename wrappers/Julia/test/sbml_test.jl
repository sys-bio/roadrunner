using RoadRunner
using Test

# Load the contents of a SBML file into a string variable
sbmlFile = joinpath(@__DIR__, "feedback.xml")
f = open(sbmlFile)
sbmlStr = read(f,String)
close(f)

rr = RoadRunner.createRRInstance()         # Start up roadRunner
RoadRunner.loadSBML(rr, sbmlStr)


@testset "parameters" begin
    @test RoadRunner.getNumberOfGlobalParameters(rr) == 5
    @test RoadRunner.getGlobalParameterIds(rr) == ["J0_VM1", "J0_Keq1", "J0_h", "J4_V4", "J4_KS4"]
end

@testset "species" begin
    @test RoadRunner.getNumberOfFloatingSpecies(rr) == 4
    @test RoadRunner.getFloatingSpeciesIds(rr) == ["S1", "S2", "S3", "S4"]
    @test RoadRunner.getNumberOfBoundarySpecies(rr) == 2
    @test RoadRunner.getBoundarySpeciesIds(rr) == ["X0", "X1"]
end

@testset "steadystates" begin
    @test RoadRunner.steadyState(rr) < 1e-6
end
