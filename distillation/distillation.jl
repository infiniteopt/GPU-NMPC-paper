module Distillation
    using InfiniteOpt, DifferentialEquations, Ipopt, Plots, Measures, CSV, DataFrames, DelimitedFiles
    using InfiniteExaModels, NLPModelsIpopt, MadNLPGPU, CUDA, ExaModels
    using Interpolations, Suppressor, HSL_jll
    include("../utils.jl")

    # System Parameters
    N = 32      # Number of trays
    FT = 17     # Feed tray
    D = 0.2     # Distillate flow rate
    F = 0.4     # Feed flow rate
    α = 1.6     # relative volatility
    xbar = 0.8958 # Setpoint for mole fraction
    ubar = 2.51459    # Setpoint for reflux ratio 
    xf = 0.5    # Liquid feed mole fraction
    # Liquid holdup
    Ac = 0.5
    At = 0.25
    Ar = 1.0

    itr = 2:FT-1        # Iterator for rectifying section
    its = FT+1:N-1      # Iterator for stripping section
    it = 1:N            # Iterator for all trays

    # Backend map
    backend_map = Dict(
    "Ipopt" => () -> Ipopt.Optimizer,
    "MOI" => () -> Ipopt.Optimizer,
    "ExaModelsCPU" => () -> ExaTranscriptionBackend(NLPModelsIpopt.IpoptSolver),
    "ExaModelsGPU" => () -> ExaTranscriptionBackend(MadNLPSolver, backend = CUDABackend())
    )

    function plotResults(tVals, xVals, yVals, uVals)
        trayVals = collect(LinRange(1, N, N))
        xset = xbar*ones(length(tVals))
        uset = ubar*ones(length(tVals))
        xSS = last.(xVals)

        # Graph the results
        xPlot = plot(tVals, xVals[1], label = "x1", xlabel = "Time (min)", lw=:3, ylabel = "Tray 1 mole fraction", ylims=(0.65, 0.91))
        plot!(tVals, xset, label = "Steady-state value", lw=:3, linestyle=:dash)
        savefig(joinpath(@__DIR__, "distill_x1.png"))
        display(xPlot)
        yPlot = plot(tVals, yVals[N], label = "y$N", xlabel = "Time (min)", lw=:3, ylabel = "Tray $N mole fraction", legend = false, color="#DB5C87", ylims=(0.15, 0.52))
        savefig(joinpath(@__DIR__, "distill_y$N.png"))
        display(yPlot)
        uPlot = plot(tVals, uVals, label = "r", xlabel = "Time (min)", lw=:3, ylabel = "Reflux ratio", ylims=(2.51, 2.67))
        plot!(tVals, uset, label = "Steady-state value", lw=:3, linestyle=:dash)
        savefig(joinpath(@__DIR__, "distill_Control.png"))
        display(uPlot)
        trayPlot = plot(trayVals, xSS, label = "x", xlabel = "Trays", lw=:3, ylabel = "Mole fraction", ylims=(0.0, 1.0))
        plot!(trayVals, last.(yVals), label = "y", lw=:3, color ="#DB5C87")
        savefig(joinpath(@__DIR__, "distill_moleFrac.png"))
        display(trayPlot)
    end

    function distillDynamics!(dx, x, u, t)
        # Define aliases for dynamics
        y = (α*x[it])./(1 .+ (α - 1)*x[it])
        L = u*D
        V = L + D
        S = L + F

        # ODEs for distillation dynamics
        dx[1] = V*(y[2] - x[1])/Ac
        dx[itr] = (L*(x[itr.-1] .- x[itr]) .- V*(y[itr] .- y[itr .+ 1]))./At
        dx[FT] = (F*xf + L*x[FT-1] - S*x[FT] - V*(y[FT] - y[FT+1]))/At
        dx[its] = (S*(x[its.-1] .- x[its]) .- V*(y[its] .- y[its.+1]))./At
        dx[N] = (S*x[N-1] - (F - D)*x[N] - V*y[N])/Ar
    end

    function distillSim(x0, u0, tspan)
        prob = ODEProblem(distillDynamics!, x0, tspan, u0)
        sol = solve(prob)
        xsol = sol.u[end]
        return xsol
    end

    function createModel(logfile, backend, xsol, MPC_params)        
        # Unpack MPC parameters
        t0, tf, Δt, Tp, Tc = MPC_params

        model = InfiniteModel(backend_map[backend]())

        @infinite_parameter(model, t ∈ [0, Tp], supports = collect(0:Δt:Tp), derivative_method = FiniteDifference(Backward()))

        # Initial condition parameter
        @finite_parameter(model, x0[i in it] == xsol[i])

        @variables(
            model,
            begin
            0 ≤ x[1:N] ≤ 1, Infinite(t), (start = xf)
            0 ≤ y[1:N] ≤ 1, Infinite(t), (start = xf)
            1 ≤ u ≤ 5, Infinite(t), (start = ubar)
            0 ≤ L ≤ 10, Infinite(t), (start = ubar*D)
            0 ≤ V ≤ 10, Infinite(t), (start = ubar*D + D)
            0 ≤ S ≤ 10, Infinite(t), (start = ubar*D + F)
            end
        )
        
        @objective(model, Min, ∫((x[1] - xbar)^2 + 0.1 * (u - ubar)^2, t))

        if Tc != Tp
            # Activate if control horizon Tc is different from prediction horizon Tp
            time_func(t_s) = (ti + Tc ≤ t_s ≤ supports(t)[end])
            time_domain = DomainRestriction(time_func, t)
            @constraint(model, u == u(ti + Tc), time_domain)
        end
        
        # Initial conditions
        @constraint(model, [i in it], x[i](supports(t)[1]) == x0[i])
        
        # System dynamics
        @constraint(model, [i in it], y[i] == (α*x[i])./(1 + (α - 1).*x[i]))
        @constraint(model, L == u*D)
        @constraint(model, V == L + D)
        @constraint(model, S == L + F)
        @constraint(model, ∂(x[1], t) == V*(y[2] - x[1])/Ac)
        @constraint(model, [i in itr], ∂(x[i], t) == (L*(x[i-1] - x[i]) - V*(y[i] - y[i+1]))/At)
        @constraint(model, ∂(x[FT], t) == (F*xf + L*x[FT-1] - S*x[FT] - V*(y[FT] - y[FT+1]))/At)
        @constraint(model, [i in its], ∂(x[i], t) == (S*(x[i-1] - x[i]) - V*(y[i] - y[i+1]))/At)
        @constraint(model, ∂(x[N], t) == (S*x[N-1] - (F - D)*x[N] - V*y[N])/Ar)
        constant_over_collocation.(u, t)

        if backend == "ExaModelsGPU"
            set_optimizer_attribute(model, "print_level", MadNLP.WARN)
        else
            if backend == "MOI"
                set_attribute(
                    model,
                    MOI.AutomaticDifferentiationBackend(),
                    MOI.Nonlinear.SymbolicMode()
                )
            end
            set_optimizer_attribute(model, "print_level", 0)
            set_optimizer_attribute(model, "file_print_level", 3)
            set_optimizer_attribute(model, "linear_solver", "ma97")
        end
        set_optimizer_attribute(model, "tol", 1e-8)
        set_optimizer_attribute(model, "output_file", logfile)
        set_optimizer_attribute(model, "print_timing_statistics", "yes")
        set_silent(model)
        return model
    end

    function distillControl(logfile, backend, xsol, MPC_params; warmstart=true, param_updates=false, OCmodel=nothing, it=1)
        setup_time = 0
        total_time = 0
        if isnothing(OCmodel)
            # Create the InfiniteOpt model
            setup_time += @elapsed OCmodel = createModel(logfile, backend, xsol, MPC_params)
            total_time += @elapsed optimize!(OCmodel)
        else
            # Extract the references
            x = OCmodel[:x]
            y = OCmodel[:y]
            u = OCmodel[:u]
            L = OCmodel[:L]
            V = OCmodel[:V]
            S = OCmodel[:S]
            x0 = OCmodel[:x0]

            if warmstart == true
                if param_updates == false
                    # Warmstart start values (will rebuild backend)
                    total_time += @elapsed set_start_values(OCmodel)
                else
                    # Warmstart start values without rebuilding backend
                    total_time += @elapsed warmstart_backend_start_values(OCmodel)
                    
                    # Adjust optimizer attributes accordingly
                    if backend == "ExaModelsGPU"
                        set_optimizer_attribute(OCmodel, "mu_init", 1E-6)
                    elseif backend == "ExaModelsCPU"
                        set_optimizer_attribute(OCmodel, "bound_push", 1e-6)
                        set_optimizer_attribute(OCmodel, "bound_frac", 1e-6)
                        set_optimizer_attribute(OCmodel, "mu_init", 1e-11)
                    else
                        set_optimizer_attribute(OCmodel, "mu_init", 1e-5)
                    end 
                end
            end
            
            # Update logger for timing stats
            if backend == "ExaModelsGPU" && param_updates == true
                logger = transformation_backend(OCmodel).solver.logger
                logger.file = logfile == "" ? nothing : open(logfile,"w+")
            end

            # Update the initial conditions
            total_time += @elapsed set_parameter_value.(x0, xsol)

            # Solve the optimal control problem
            @suppress_out begin
            total_time += @elapsed optimize!(OCmodel)
            end
        end
        
        status = termination_status(OCmodel)

        # Extract timing stats & results
        if backend == "ExaModelsGPU"
            CUDA.allowscalar(true)
            nvar, ncon, nit, sol_time, factor_time, ad_time = madnlp_stats(logfile)
        else
            nvar, ncon, nit, sol_time, factor_time, ad_time = ipopt_stats(logfile)
        end
        setup_time += total_time - sol_time
        time_results = [nvar, ncon, nit, total_time, setup_time, sol_time, factor_time, ad_time]
        uVals = value.(OCmodel[:u])

        return uVals[2], status, time_results, OCmodel
    end

    function distillMPC(backend, MPC_params, run; warmstart=true, param_updates=true, show_plots=false, compile=false)
        # Unpack the NMPC parameters
        t0, tf, Δt, Tp, Tc = MPC_params
        t_points = Int(tf/Δt +1)    # Total number of time points
        t_vals = collect(LinRange(t0, tf, t_points)/60) # Time in mins for graphing purposes

        # Initialize arrays for storing results
        x_vals = [zeros(t_points) for tray in 1:N]
        y_vals = [zeros(t_points) for tray in 1:N]
        u_vals = []
        nvars, ncons, nits, total_times, setup_times, sol_times, factor_times, ad_times = [], [], [], [], [], [], [], []
        
        # Initialize values before MPC loop
        x_k = ones(N)*xf    # For initial conditions
        y_k = (α*x_k)./(1 .+ (α - 1)*x_k)
        for j in 1:N
            x_vals[j][1] = value(x_k[j])
            y_vals[j][1] = value(y_k[j])
        end
        model = nothing     # For optimal control model

        # Initialize log file
        WS = warmstart ? "WS" : ""
        PU = param_updates ? "PU" : ""
        filename = "distill_$(backend)_$WS$PU.log"
        logfile = joinpath(@__DIR__, "logs", filename)

        # Main NMPC loop
        for i in eachindex(t_vals)
            println("Current NMPC iteration: $(i) of $(length(t_vals))")

            # Ensures new model is built from scratch per iteration
            if warmstart == false && param_updates == false
                model = nothing
            end

            # Solve the optimal control problem
            u_k, status, timing_k, model = distillControl(logfile, backend, x_k, MPC_params, warmstart=warmstart, param_updates=param_updates, OCmodel=model, it=i)

            # Check the solve status
            if !(status in (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED))
                println("Status = $(status)")
                println("Model not locally solved. Terminating.")
                break
            end

            # Ensure no negative times
            timing_k = max.(timing_k, 0)

            if backend == "ExaModelsGPU" && param_updates == true
                # Timing stats are accumulated; so recalculate
                # Fix missing or invalid timing values
                timing_k[1] = timing_k[1] == 0.0 ? (isempty(nvars) ? 0.0 : nvars[end]) : timing_k[1]
                timing_k[2] = timing_k[2] == 0.0 ? (isempty(ncons) ? 0.0 : ncons[end]) : timing_k[2]
            end

            # Update arrays with current results
            push!(u_vals, u_k)
            push!(nvars, timing_k[1])
            push!(ncons, timing_k[2])
            push!(nits, timing_k[3])
            push!(total_times, timing_k[4])
            push!(setup_times, timing_k[5])
            push!(sol_times, timing_k[6])
            push!(factor_times, timing_k[7])
            push!(ad_times, timing_k[8])

            # Simulate system forward & store state values
            ti = t_vals[i]
            tspan = (ti, ti+Δt)
            x_k = distillSim(x_k, u_k, tspan)
            y_k = (α*x_k)./(1 .+ (α - 1)*x_k)
            for j in 1:N
                x_vals[j][i] = value(x_k[j])
                y_vals[j][i] = value(y_k[j])
            end
        end

        if backend in ["Ipopt", "MOI", "ExaModelsCPU"] && param_updates == true
            # Need to extract nvar, ncon, total_time and ad_time from log file
            nvars, ncons, nits, sol_times, factor_times, ad_times = ipopt_stats_all(logfile)

            # Recalculate the model times; return 0 if negative
            setup_times = max.(total_times .- sol_times, 0.0)
        end

        # Extract the time statistics
        result_file = "distill-results-$WS$PU.csv"
        result_path = joinpath(dirname(@__DIR__), "results", backend, result_file)
        file_exists = isfile(result_path)

        # Make a table to save the results to
        its = collect(1:1:length(nvars))
        nrows = length(its)+1
        ncols = 9
        table = Matrix{String}(undef, nrows, ncols)
        table[1, :] = append!(["iteration", "nvar", "ncon", "nits", "total_time", "setup_time", "solve_time", "factor_time","ad_time"])
        table[2:end, :] = string.(hcat(its, nvars, ncons, nits, total_times, setup_times, sol_times, factor_times, ad_times))
        
        # Save the matrix as a CSV
        if compile == false # Skip saving results during compilation runs
            open(result_path, file_exists && run != 1 ? "a" : "w") do io
                if file_exists
                    # skip the header
                    writedlm(io, table[2:end, :], ",")
                else
                    writedlm(io, table, ",")
                end
            end
        end

        if show_plots == true
            plotResults(t_vals, x_vals, y_vals, u_vals)
        end
    end
end