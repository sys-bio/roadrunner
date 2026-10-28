using RoadRunner
using Test


@testset "test roadrunner errors" begin
    ant_str = """
        A -> B; k1*A,
        A = 10,
    """
    @test_throws ErrorException("Error in model string, line 1:  syntax error, unexpected ',', expecting end of line or ';' or '\\n'") RoadRunner.loada(ant_str)
end

@testset "test antimony errors" begin
    ant_str = """
        A -> B; k1*A
        A = 10
        k1 = 5
    """
    rr = RoadRunner.loada(ant_str)
    @test_throws ErrorException("RoadRunner exception: No sbml element exists for symbol 'S'\n") RoadRunner.getValue(rr, "S")
end
