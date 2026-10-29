using RoadRunner
using Test

sbmlStr = read(joinpath(@__DIR__, "feedback.xml"), String)

rr = RoadRunner.createRRInstance()
RoadRunner.loadSBML(rr, sbmlStr)

@testset "seed" begin
    RoadRunner.setSeed(rr, 42)
    @test RoadRunner.getSeed(rr) == 42

    # a value that does not fit in 32 bits
    RoadRunner.setSeed(rr, 5000000000)
    @test RoadRunner.getSeed(rr) == 5000000000
end
