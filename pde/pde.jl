module PDE
    using InfiniteOpt, Ipopt, Plots, DifferentialEquations, LinearAlgebra, Statistics, Measures, CSV, DataFrames, DelimitedFiles
    using InfiniteExaModels, NLPModelsIpopt, MadNLPGPU, CUDA
    using Interpolations, Suppressor, HSL_jll
    include("../utils.jl")

    # System Parameters
    alp = 230  # Thermal conductivity (W/mK)
    L = 0.1       # Length (m)
    T_boundary = 50.0 # Boundary temperature
    T_init = 20.0
    Q_init = 60.0
    n = 19     # Internal Spatial Points
    N = 21     # Number of External
    dx = L / (n + 1) # Spacing between spatial points
    λ = alp / dx^2   # Finite Difference Discretization Parameter
    heat_loss = 10

    backend_map = Dict(
        "Ipopt" => () -> TranscriptionBackend(Ipopt.Optimizer, update_parameter_functions = true),
        "MOI" => () -> TranscriptionBackend(Ipopt.Optimizer, update_parameter_functions = true),
        "ExaModelsCPU" => () -> ExaTranscriptionBackend(NLPModelsIpopt.IpoptSolver),
        "ExaModelsGPU" => () -> ExaTranscriptionBackend(MadNLPSolver, backend = CUDABackend())
    )

    function setpoint(t)
        if t < 0.5
            return 58
        elseif t < 1.5
            return 55
        else
            return 63
        end
    end

    function plotResults(time_vals, T_side, T_bot, T_avg, T_sp, T)
        domain_x = (-0.5, 0.5)
        domain_y = (-0.5, 0.5)

        # Control history
        p1 = plot(time_vals[1:length(T_bot)],  Array{Float64}([T_side T_bot T_avg T_sp]),
                label=["Side Temperature" "Bottom Temperature" "Average Domain Temperature" "Setpoint Temperature"], xlabel="Time", ylabel="Temperature")
        
        # Final temperature distribution
        if !isempty(T)
            T_final = T
            x_coords = range(-1, 1, length = N)
            y_coords = range(-1, 1, length = N)
            
            p2 = heatmap(x_coords, y_coords, T_final',
                        title="Final Temperature Distribution",
                        xlabel="x₁", ylabel="x₂", colorbar_title="Temperature")
            
            # Mark tracking domain
            track_x = [domain_x[1], domain_x[2], domain_x[2], domain_x[1], domain_x[1]]
            track_y = [domain_y[1], domain_y[1], domain_y[2], domain_y[2], domain_y[1]]
            plot!(p2, track_x, track_y, lc=:red, lw=2, label="Tracking Domain", linestyle=:dash)
        end
        
        combined_plot = plot(p1, p2, layout=(2,1), size=(800,600))
        savefig(joinpath(@__DIR__, "pde_graph.png"))
        display(combined_plot)
    end

    function pdeDynamics!(dT, T, p, t)
        T_side, T_bottom = p
        T_mat = reshape(T, n, n)
        T_all = zeros(eltype(T), n+2, n+2)
        T_all[2:end-1, 2:end-1] = T_mat
        T_all[1, :] .= T_side
        T_all[:, end] .= T_boundary
        T_all[end, :] .= T_boundary
        T_all[:, 1] .= T_bottom

        dT_mat = zeros(eltype(T), n, n)

        for i in 1:n, j in 1:n
            dT_mat[i,j] = λ*(T_all[i+1,j+2] + T_all[i+1,j] + 
                            T_all[i+2,j+1] + T_all[i,j+1] - 
                            4*T_all[i+1,j+1])
        end

        dT[:] = dT_mat[:]
    end

    function pdeSim(T0_vec, Tside_val, Tbottom_val, tspan)
        prob = ODEProblem(pdeDynamics!, T0_vec, tspan, [Tside_val, Tbottom_val])
        sol = solve(prob, TRBDF2(), reltol=1e-6, abstol=1e-8)
        T_interior = reshape(sol.u[end], n, n)
        T_full = zeros(n+2, n+2)
        T_full[2:end-1, 2:end-1] = T_interior
        T_full[:,end] .= T_boundary
        T_full[1, :] .= Tside_val
        T_full[end, :] .= T_boundary
        T_full[:, 1] .= Tbottom_val
        return T_full
    end

    function createModel(logfile, backend, T_sol, Tsp_val, MPC_params)    
        # Unpack MPC parameters
        t0, tf, Δt, Tp, Tc = MPC_params

        # Create the InfiniteOpt model
        model = InfiniteModel(backend_map[backend]())

        @infinite_parameter(model, t in [0, Tp], supports = collect(0:Δt:Tp))
        @infinite_parameter(model, x[1:2] in [-1, 1], independent = true, num_supports = N, derivative_method = FiniteDifference(Central()))

        # Setpoint parameter for objective function
        @finite_parameter(model, Tsp == Tsp_val)

        @variable(model, T, Infinite(t, x...), start = T_init)
        @variable(model, 50 <= Q, Infinite(t), start = Q_init)

        # Objective function
        @objective(model, Min, integral(integral(integral((T - Tsp)^2, x[1], -0.5, 0.5), x[2], -0.5, 0.5) + 1e-8 * Q^2, t))

        # Heat diffusion PDE
        @constraint(model, deriv(T, t) == alp * (@deriv(T, x[1]^2) + @deriv(T, x[2]^2)) - 5*heat_loss^2)

        # Time boundary conditions
        xs = supports.(x)
        # initial temperature across the whole plate
        interp = linear_interpolation((xs[1], xs[2]), T_sol)
        func = (x1, x2) -> interp(x1, x2)
        @parameter_function(model, T0_pf == func(x[1], x[2]))
        @constraint(model, T(0, x...) == T0_pf)

        # Dirchlet boundary conditions
        add_generative_supports(t) # only needed if orthogonal collocation is used
        ts = supports(t)
        t0_func(t_s) = (ts[2] ≤ t_s ≤ Tp)
        side_func(t_s, x2_s) = (ts[2] ≤ t_s ≤ Tp) && (xs[2][2] ≤ x2_s ≤ 1)
        top_func(t_s, x1_s) = (ts[1] ≤ t_s ≤ Tp) && (xs[1][2] ≤ x1_s ≤ xs[1][end-1])
        time_domain = DomainRestriction(t0_func, t)
        side_wall_domain = DomainRestriction(side_func, t, x[2])
        top_wall_domain = DomainRestriction(top_func, t, x[1])
        @constraint(model, T(t, -1, x[2]) == 4.1*Q^(1/3), side_wall_domain) # Left wall
        @constraint(model, T(t, 1, x[2]) == T_boundary, side_wall_domain)  # Right wall
        @constraint(model, T(t, x[1], -1) == sqrt(Q), time_domain)   # Bottom wall
        @constraint(model, T(t, x[1], 1) == T_boundary, top_wall_domain)  # Top wall

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
            set_optimizer_attribute(model, "linear_solver", "ma27")
        end
        set_optimizer_attribute(model, "output_file", logfile)
        set_optimizer_attribute(model, "print_timing_statistics", "yes")
        set_silent(model)
        return model
    end

    function pdeControl(logfile, backend, T_sol, Tsp_val, MPC_params; warmstart = true, param_updates = false, OCmodel = nothing, it=1)
        setup_time = 0
        total_time = 0
        if isnothing(OCmodel)
            # Create the infinite model
            setup_time += @elapsed OCmodel = createModel(logfile, backend, T_sol, Tsp_val, MPC_params)
            total_time += @elapsed optimize!(OCmodel)
        else
            # Extract the references
            x = OCmodel[:x]
            T = OCmodel[:T]
            Q = OCmodel[:Q]
            Tsp = OCmodel[:Tsp]
            T0_pf = OCmodel[:T0_pf]
            xs = supports.(x)

            if warmstart == true
                if param_updates == false
                    # Warmstart start values (will rebuild backend)
                    total_time += @elapsed set_start_values(OCmodel)
                    if backend == "ExaModelsCPU"
                        OCmodel.backend.prev_options = Dict{Symbol, Any}()
                    end
                    # New logfile to go with new backend
                    logfile = replace(logfile, ".log" => "$it.log")
                    set_optimizer_attribute(OCmodel, "output_file", logfile)
                else
                    # Update start values without rebuilding backend
                    total_time += @elapsed warmstart_backend_start_values(OCmodel)

                    # Adjust optimizer attributes accordingly
                    if backend == "ExaModelsGPU"
                        set_optimizer_attribute(OCmodel, "mu_init", 1E-6)
                    elseif backend == "ExaModelsCPU"
                        set_optimizer_attribute(OCmodel, "bound_push", 1e-11)
                        set_optimizer_attribute(OCmodel, "bound_frac", 1e-11)
                        set_optimizer_attribute(OCmodel, "mu_init", 1e-10)
                    else
                        set_optimizer_attribute(OCmodel, "mu_init", 1e-4)    
                    end
                end
            end

            # Update logger for timing stats
            if backend == "ExaModelsGPU" && param_updates == true
                logger = transformation_backend(OCmodel).solver.logger
                logger.file = logfile == "" ? nothing : open(logfile,"w+")
            end

            # Update the initial condition
            interp = linear_interpolation((xs[1], xs[2]), T_sol)
            newFunc = (x1, x2) -> interp(x1, x2)
            total_time += @elapsed set_parameter_value(T0_pf, newFunc)

            # Update the setpoint
            total_time += @elapsed set_parameter_value(Tsp, Tsp_val)

            # Solve the optimal control problem
            @suppress_out begin
            total_time += @elapsed optimize!(OCmodel)
            end
        end

        status = termination_status(OCmodel)

        # Extract timing stats & results
        if backend == "ExaModelsGPU"
            CUDA.allowscalar(true)
            nvar, ncon, it, sol_time, ad_time = madnlp_stats(logfile)
        else
            nvar, ncon, it, sol_time, ad_time = ipopt_stats(logfile)
        end
        setup_time += total_time - sol_time
        time_results = [nvar, ncon, it, total_time, setup_time, sol_time, ad_time]
        Q_vals = value.(OCmodel[:Q])
        Tside_vals = 4.1*(Q_vals.^(1/3))
        Tbot_vals = sqrt.(Q_vals)

        return Tside_vals[2], Tbot_vals[2], status, time_results, OCmodel
    end

    function pdeMPC(backend, MPC_params, run; warmstart=true, param_updates=true, show_plots=false, compile=false)
        # Unpack the NMPC parameters
        t0, tf, Δt, Tp, Tc = MPC_params
        t_vals = t0:Δt:tf

        # Initialize arrays for storing results
        Tavg_vals, Tside_vals, Tbot_vals = [], [], []
        nvars, ncons, nits, total_times, setup_times, sol_times, ad_times = [], [], [], [], [], [], []

        # Initialize values before MPC loop
        T_k = fill(T_boundary, N, N)  # For initial conditions
        T_k[2:end-1, 2:end-1] .= T_init
        model = nothing     # For optimal control model

        # Initialize log file
        WS = warmstart ? "WS" : ""
        PU = param_updates ? "PU" : ""
        filename = "pde_$(backend)_$WS$PU.log"
        if WS != "" && PU == ""
            logfile = joinpath(@__DIR__, "logs", "WS", filename)
        else
            logfile = joinpath(@__DIR__, "logs", filename)
        end

        # Main NMPC loop
        for i in eachindex(t_vals)
            println("Current NMPC iteration: $(i) of $(length(t_vals))")

            ti = t_vals[i]
            Tsp = setpoint(ti)

            # Ensures new model is built from scratch per iteration
            if warmstart == false && param_updates == false
                model = nothing
            end

            # Solve the optimal control problem
            Tside_k, Tbot_k, status, timing_k, model = pdeControl(logfile, backend, T_k, Tsp, MPC_params, warmstart=warmstart, param_updates=param_updates, OCmodel=model, it=i)

            # Check the solve status
            if !(status in (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED))
                println("Status = $(status)")
                println("Model not locally solved. Terminating.")
                break
            end

            # Ensure no negative times
            timing_k = max.(timing_k, 0.0)

            if backend == "ExaModelsGPU" && param_updates == true
                # Timing stats are accumulated; so recalculate
                # Also fix missing or invalid timing values
                timing_k[1] = timing_k[1] == 0.0 ? (isempty(nvars) ? 0.0 : nvars[end]) : timing_k[1]
                timing_k[2] = timing_k[2] == 0.0 ? (isempty(ncons) ? 0.0 : ncons[end]) : timing_k[2]
                timing_k[3] = isempty(nits) ? timing_k[3] : Int(timing_k[3] - nits[end])
                timing_k[5] = timing_k[5] <= 0 ? 0.0 : timing_k[5]
                timing_k[6] -= isempty(sol_times) ? 0.0 : sum(sol_times)
                timing_k[7] -= isempty(ad_times) ? 0.0 : sum(ad_times)
            end

            # Update arrays with current results
            push!(Tside_vals, Tside_k)
            push!(Tbot_vals, Tbot_k)
            push!(nvars, timing_k[1])
            push!(ncons, timing_k[2])
            push!(nits, timing_k[3])
            push!(total_times, timing_k[4])
            push!(setup_times, timing_k[5])
            push!(sol_times, timing_k[6])
            push!(ad_times, timing_k[7])
            
            # Simulate system forward (using only interior points)
            tspan = (ti, ti+Δt)
            T_next_full = pdeSim(T_k[2:end-1, 2:end-1][:], Tside_k, Tbot_k, tspan)
            T_k = T_next_full
            T_k_avg = mean(T_k[6:16, 6:16])

            # Store current state values
            push!(Tavg_vals, T_k_avg) 
        end
        
        if backend in ["Ipopt", "MOI", "ExaModelsCPU"] && param_updates == true
            # Need to extract nvar, ncon, total_time and ad_time from log file
            nvars, ncons, nits, sol_times, ad_times = ipopt_stats_all(logfile)

            # Recalculate the model times; return 0 if negative
            setup_times = max.(total_times .- sol_times, 0.0)
        end

        # Extract the time statistics
        result_file = "pde-results-$WS$PU.csv"
        result_path = joinpath(dirname(@__DIR__), "results", backend, result_file)
        file_exists = isfile(result_path)

        # Make a table to save the results to
        its = collect(1:1:length(nvars))
        nrows = length(its)+1
        ncols = 8
        table = Matrix{String}(undef, nrows, ncols)
        table[1, :] = append!(["iteration", "nvar", "ncon", "nits", "total_time", "setup_time", "solve_time", "ad_time"])
        table[2:end, :] = string.(hcat(its, nvars, ncons, nits, total_times, setup_times, sol_times, ad_times))
        
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
            Tsp_vals = setpoint.(t_vals)
            plotResults(t_vals, Tside_vals, Tbot_vals, Tavg_vals, Tsp_vals, T_k)
        end
    end
end