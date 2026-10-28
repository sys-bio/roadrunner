using RoadRunner
using Test


@testset "test roadrunner errors" begin
    ant_str = """
        A -> B; k1*A,
        A = 10,
    """
    err = try RoadRunner.loada(ant_str); nothing catch e; e end
    @test err isa ErrorException
    # the order of the expected tokens depends on the parser generator, so only the start is checked
    @test startswith(err.msg, "Error in model string, line 1:  syntax error, unexpected ','")
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
