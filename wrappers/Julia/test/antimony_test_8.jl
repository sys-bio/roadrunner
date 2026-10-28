using RoadRunner
using Test


ant_str = """
    compartment c1, c2, c3, c4;
    c1 = 1.1; c2 = 2.2; c3 = 3.3; c4 = 4.4
"""

rr = RoadRunner.loada(ant_str)

@testset "compartment" begin
    @test RoadRunner.getNumberOfCompartments(rr) == 4
    @test RoadRunner.getCompartmentIds(rr) == ["c1", "c2", "c3", "c4"]
    @test RoadRunner.getCompartmentByIndex(rr, 0) == 1.1
    @test RoadRunner.getCompartmentByIndex(rr, 1) == 2.2
    @test RoadRunner.getCompartmentByIndex(rr, 2) == 3.3
    @test RoadRunner.getCompartmentByIndex(rr, 3) == 4.4
end
