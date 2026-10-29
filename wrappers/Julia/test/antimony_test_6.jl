using RoadRunner
using Test


ant_str = """
    k1 := sin (time)
    k2 := k1 + 3.14

    S1 -> S2; v
    S1 = 1; S2 = 0; v = 0
"""

rr = RoadRunner.loada(ant_str)

@testset "compartment" begin
    @test RoadRunner.getNumberOfCompartments(rr) == 1
end

@testset "reaction" begin
    @test RoadRunner.getNumberOfReactions(rr) == 1
end
