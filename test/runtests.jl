using DoME
using Test


@testset "DoME.jl" begin

    # Test with a regression problem
    inputs = Float64.(hcat(1:10, 20:-2:1));
    targets = inputs[:,1].*2.4 .- inputs[:,2]./inputs[:,1] .- 10;
    validationIndices = [4,6,8];
    testIndices = [3,7,9];
    (trainingResult, validationResult, testResult, bestTree) = dome(inputs, targets;
        validationIndices = validationIndices ,
        testIndices = testIndices ,
        minimumReductionMSE = 1e-6 ,
        maximumHeight = Inf ,
        maximumNodes = 13 ,
        strategy = StrategySelectiveWithConstantOptimization ,
        useDivisionOperator = true ,
        showText = false
    );

    func = eval(Meta.parse(string("X -> ", vectorString(bestTree))));
    outputs = Base.invokelatest(func, inputs);
    trainResult  = mean((outputs[setdiff(1:length(targets),vcat(validationIndices,testIndices))] .- targets[setdiff(1:length(targets),vcat(validationIndices,testIndices))]).^2);
    valResult    = mean((outputs[validationIndices] .- targets[validationIndices]).^2);
    testResult   = mean((outputs[      testIndices] .- targets[      testIndices]).^2);
    @test isapprox(trainResult, 6.172e-29; atol=1e-7)
    @test isapprox(valResult  , 5.732e-29; atol=1e-7)
    @test isapprox(testResult , 4.312e-29; atol=1e-7)
    @test isapprox(trainResult,   trainingResult; atol=1e-7)
    @test isapprox(valResult  , validationResult; atol=1e-7)
    @test isapprox(testResult ,       testResult; atol=1e-7)

    # Test with a classification problem
    targets = targets.>=0;
    (trainingResult, validationResult, testResult, bestTree) = dome(inputs, targets;
        validationIndices = validationIndices ,
        testIndices = testIndices ,
        minimumReductionMSE = 1e-6 ,
        maximumHeight = Inf ,
        maximumNodes = 13 ,
        strategy = StrategySelectiveWithConstantOptimization ,
        useDivisionOperator = true ,
        showText = false
    );

    func = eval(Meta.parse(string("X -> ", vectorString(bestTree))));
    outputs = Base.invokelatest(func, inputs);
    trainResult  = mean((outputs[setdiff(1:length(targets),vcat(validationIndices,testIndices))].>=0) .== targets[setdiff(1:length(targets),vcat(validationIndices,testIndices))]);
    valResult    = mean((outputs[validationIndices].>=0) .== targets[validationIndices]);
    testResult   = mean((outputs[      testIndices].>=0) .== targets[      testIndices]);
    @test isapprox(trainResult,    1)
    @test isapprox(valResult  , 2/3.)
    @test isapprox(testResult ,    1)
    @test isapprox(trainResult,   trainingResult)
    @test isapprox(valResult  , validationResult)
    @test isapprox(testResult ,       testResult)

end
