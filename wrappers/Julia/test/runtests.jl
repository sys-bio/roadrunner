using Test

# Each test file runs in its own module, so globals from one file do not leak into the next.
function run_isolated(file)
    Base.include(Module(:RoadRunnerTestFile), joinpath(@__DIR__, file))
end

@testset "antimony test 0" begin run_isolated("antimony_test_0.jl") end

@testset "antimony test 1" begin run_isolated("antimony_test_1.jl") end

@testset "antimony test 2" begin run_isolated("antimony_test_2.jl") end

@testset "antimony test 3" begin run_isolated("antimony_test_3.jl") end

@testset "antimony test 4" begin run_isolated("antimony_test_4.jl") end

@testset "antimony test 5" begin run_isolated("antimony_test_5.jl") end

@testset "antimony test 6" begin run_isolated("antimony_test_6.jl") end

@testset "antimony test 7" begin run_isolated("antimony_test_7.jl") end

@testset "antimony test 8" begin run_isolated("antimony_test_8.jl") end

@testset "sbml test" begin run_isolated("sbml_test.jl") end

@testset "error test" begin run_isolated("error_test.jl") end

@testset "seed test" begin run_isolated("seed_test.jl") end
