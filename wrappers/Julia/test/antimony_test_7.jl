using RoadRunner
using Test


ant_str = """
    model rabbit()

    k1 = 1.1; k2 = 2.2; k3 = 3.3; k4 = 4.4
    k5 = 5.5; k6 = 6.6; k7 = 7.7; k8 = 8.8
    end
"""

rr = RoadRunner.loada(ant_str)


@testset "parameters" begin
    @test RoadRunner.getNumberOfGlobalParameters(rr) == 8
    @test RoadRunner.getGlobalParameterIds(rr) == ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
    @test RoadRunner.getGlobalParameterValues(rr) == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
end
