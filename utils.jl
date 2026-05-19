using Printf, InfiniteOpt

function ipopt_stats(fname)
    output = read(fname, String)
    tot = parse(Float64, split(split(split(output, "OverallAlgorithm....................:")[2], "wall:")[2], ")")[1])
    ad = parse(Float64,split(split(split(output, "Function Evaluations................:")[2], "wall:")[2], ")")[1])
    symFact  = parse(Float64, split(split(split(output, "LinearSystemSymbolicFactorization..:")[2], "wall:")[2], ")")[1])
    numFact  = parse(Float64, split(split(split(output, "LinearSystemFactorization..........:")[2], "wall:")[2], ")")[1])
    linSolve = parse(Float64, split(split(split(output, "LinearSystemBackSolve..............:")[2], "wall:")[2], ")")[1])
    nvar = parse(Int, split(split(output,"Total number of variables............................:")[2], "\n")[1])
    necon = parse(Int, split(split(output,"Total number of equality constraints.................:")[2], "\n")[1])
    nicon = parse(Int, split(split(output,"Total number of inequality constraints...............:")[2], "\n")[1])
    nits = parse(Int, split(split(output,"Number of Iterations....:")[2], "\n")[1])
    ncon = necon + nicon
    factor = symFact + numFact + linSolve

    return nvar, ncon, nits, tot, factor, ad
end

function ipopt_stats_all(fname)
    output = read(fname, String)

    # Split by solve occurrences
    sections = split(output, "Total number of variables............................:")
    nvars, ncons, nits, tots, factors, ads = [], [], [], [], [], []

    # 
    for (i, sec) in enumerate(sections[2:end])
        block = "Total number of variables............................:" * sec
        try
            tot = parse(Float64, split(split(split(block, "OverallAlgorithm....................:")[2], "wall:")[2], ")")[1])
            ad  = parse(Float64, split(split(split(block, "Function Evaluations................:")[2], "wall:")[2], ")")[1])
            symFact  = parse(Float64, split(split(split(block, "LinearSystemSymbolicFactorization..:")[2], "wall:")[2], ")")[1])
            numFact  = parse(Float64, split(split(split(block, "LinearSystemFactorization..........:")[2], "wall:")[2], ")")[1])
            linSolve = parse(Float64, split(split(split(block, "LinearSystemBackSolve..............:")[2], "wall:")[2], ")")[1])
            nvar  = parse(Int, split(split(block, "Total number of variables............................:")[2], "\n")[1])
            necon = parse(Int, split(split(block, "Total number of equality constraints.................:")[2], "\n")[1])
            nicon = parse(Int, split(split(block, "Total number of inequality constraints...............:")[2], "\n")[1])
            nit = parse(Int, split(split(output,"Number of Iterations....:")[2], "\n")[1])
            ncon = necon + nicon
            factor = symFact + numFact + linSolve
            
            push!(nvars, nvar)
            push!(ncons, ncon)
            push!(nits, nit)
            push!(tots, tot)
            push!(factors, factor)
            push!(ads, ad)
        catch err
            @warn "Failed to parse IPOPT stats for solve $i: $err"
        end
    end
    return nvars, ncons, nits, tots, factors, ads
end

function madnlp_stats(fname)
    output = read(fname, String)
    tot = -1.0
    factor = -1.0
    ad = -1.0
    nvar = -1.0
    ncon = -1.0
    nits = 0
    tot = parse(Float64, split(split(output,"Total wall-clock secs                                       =")[2], "\n")[1])
    factor = parse(Float64, split(split(output,"Total wall-clock secs in linear solver                      =")[2], "\n")[1])
    ad = parse(Float64, split(split(output,"Total wall-clock secs in NLP function evaluations           =")[2], "\n")[1])
    nits = parse(Int, split(split(output,"Number of Iterations....:")[2], "\n")[1])
    try
        nvar = parse(Int, split(split(output,"Total number of variables............................:")[2], "\n")[1])
        nicon = parse(Int, split(split(output,"Total number of inequality constraints...............:")[2], "\n")[1])
        necon = parse(Int, split(split(output,"Total number of equality constraints.................:")[2], "\n")[1])
        ncon = necon + nicon
    catch
    end
    return nvar, ncon, nits, tot, factor, ad
end
