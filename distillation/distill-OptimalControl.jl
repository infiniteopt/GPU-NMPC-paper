using OptimalControl, Plots, DifferentialEquations, Measures, CSV, DataFrames, DelimitedFiles
using MadNLPGPU, CUDA
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

# MPC Parameters; units are in seconds
t0 = 0
tf = 300    # Total simulation time
Δt = 2      # Control time step (s)
Tp = 180    # Prediction horizon (s)
Tc = Tp     # Control horizon (s)
t_points = Int(tf/Δt + 1)    # Total number of time points
nsupps = Int(Tp/Δt) + 1  # Number of supports in the time horizon
t_vals = collect(LinRange(t0, tf, t_points))/60 # Time in mins for graphing purposes
itr = 2:FT-1        # Iterator for rectifying section
its = FT+1:N-1      # Iterator for stripping section
it = 1:N            # Iterator for all trays

# Initialize log file
filename = "distill_OptimalControl.log"
logfile = joinpath(@__DIR__, "logs", filename)

function distillDynamics!(dx, x, u, t)
    y = (α*x[it])./(1 .+ (α - 1)*x[it])
    L = u*D
    V = L + D
    S = L + F
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

function OptControlModel(x0)
    ocp = @def begin
        t ∈ [0, Tp], time
        x ∈ R^N, state  
        u ∈ R, control

        # Aliases for dynamics
        y = [(α*x[i](t))/(1 + (α - 1)*x[i](t)) for i in 1:N]
        L = u(t)*D
        V = L + D
        S = L + F
        dx1 = V*(y[2] - x[1](t))/Ac
        dxr = [(L*(x[i-1](t) - x[i](t)) - V*(y[i] - y[i+1]))/At for i in itr]
        dxf = (F*xf + L*x[FT-1](t) - S*x[FT](t) - V*(y[FT] - y[FT+1]))/At
        dxs = [(S*(x[i-1](t) - x[i](t)) - V*(y[i] - y[i+1]))/At for i in its]
        dxn = (S*x[N-1](t) - (F - D)*x[N](t) - V*y[N])/Ar

        # Variable bounds
        0 ≤ x[1](t) ≤ 1
        0 ≤ x[2](t) ≤ 1
        0 ≤ x[3](t) ≤ 1
        0 ≤ x[4](t) ≤ 1
        0 ≤ x[5](t) ≤ 1
        0 ≤ x[6](t) ≤ 1
        0 ≤ x[7](t) ≤ 1
        0 ≤ x[8](t) ≤ 1
        0 ≤ x[9](t) ≤ 1
        0 ≤ x[10](t) ≤ 1
        0 ≤ x[11](t) ≤ 1
        0 ≤ x[12](t) ≤ 1
        0 ≤ x[13](t) ≤ 1
        0 ≤ x[14](t) ≤ 1
        0 ≤ x[15](t) ≤ 1
        0 ≤ x[16](t) ≤ 1
        0 ≤ x[17](t) ≤ 1
        0 ≤ x[18](t) ≤ 1
        0 ≤ x[19](t) ≤ 1
        0 ≤ x[20](t) ≤ 1
        0 ≤ x[21](t) ≤ 1
        0 ≤ x[22](t) ≤ 1
        0 ≤ x[23](t) ≤ 1
        0 ≤ x[24](t) ≤ 1
        0 ≤ x[25](t) ≤ 1
        0 ≤ x[26](t) ≤ 1
        0 ≤ x[27](t) ≤ 1
        0 ≤ x[28](t) ≤ 1
        0 ≤ x[29](t) ≤ 1
        0 ≤ x[30](t) ≤ 1
        0 ≤ x[31](t) ≤ 1
        0 ≤ x[32](t) ≤ 1
        0 ≤ y[1] ≤ 1
        0 ≤ y[2] ≤ 1
        0 ≤ y[3] ≤ 1
        0 ≤ y[4] ≤ 1
        0 ≤ y[5] ≤ 1
        0 ≤ y[6] ≤ 1
        0 ≤ y[7] ≤ 1
        0 ≤ y[8] ≤ 1
        0 ≤ y[9] ≤ 1
        0 ≤ y[10] ≤ 1
        0 ≤ y[11] ≤ 1
        0 ≤ y[12] ≤ 1
        0 ≤ y[13] ≤ 1
        0 ≤ y[14] ≤ 1
        0 ≤ y[15] ≤ 1
        0 ≤ y[16] ≤ 1
        0 ≤ y[17] ≤ 1
        0 ≤ y[18] ≤ 1
        0 ≤ y[19] ≤ 1
        0 ≤ y[20] ≤ 1
        0 ≤ y[21] ≤ 1
        0 ≤ y[22] ≤ 1
        0 ≤ y[23] ≤ 1
        0 ≤ y[24] ≤ 1
        0 ≤ y[25] ≤ 1
        0 ≤ y[26] ≤ 1
        0 ≤ y[27] ≤ 1
        0 ≤ y[28] ≤ 1
        0 ≤ y[29] ≤ 1
        0 ≤ y[30] ≤ 1
        0 ≤ y[31] ≤ 1
        0 ≤ y[32] ≤ 1
        1 ≤ u(t) ≤ 5
        0 ≤ L ≤ 10    # L
        0 ≤ V ≤ 10    # V
        0 ≤ S ≤ 10   # S
        
        # Assign tray dynamic ODEs for rectifying, feed & stripping sections
        ∂(x[1])(t) == dx1
        ∂(x[2])(t) == dxr[1]
        ∂(x[3])(t) == dxr[2]
        ∂(x[4])(t) == dxr[3]
        ∂(x[5])(t) == dxr[4]
        ∂(x[6])(t) == dxr[5]
        ∂(x[7])(t) == dxr[6]
        ∂(x[8])(t) == dxr[7]
        ∂(x[9])(t) == dxr[8]
        ∂(x[10])(t) == dxr[9]
        ∂(x[11])(t) == dxr[10]
        ∂(x[12])(t) == dxr[11]
        ∂(x[13])(t) == dxr[12]
        ∂(x[14])(t) == dxr[13]
        ∂(x[15])(t) == dxr[14]
        ∂(x[16])(t) == dxr[15]
        ∂(x[17])(t) == dxf
        ∂(x[18])(t) == dxs[1]
        ∂(x[19])(t) == dxs[2]
        ∂(x[20])(t) == dxs[3]
        ∂(x[21])(t) == dxs[4]
        ∂(x[22])(t) == dxs[5]
        ∂(x[23])(t) == dxs[6]
        ∂(x[24])(t) == dxs[7]
        ∂(x[25])(t) == dxs[8]
        ∂(x[26])(t) == dxs[9]
        ∂(x[27])(t) == dxs[10]
        ∂(x[28])(t) == dxs[11]
        ∂(x[29])(t) == dxs[12]
        ∂(x[30])(t) == dxs[13]
        ∂(x[31])(t) == dxs[14]
        ∂(x[32])(t) == dxn

        # Initial conditions
        x[1](0) == x0[1]
        x[2](0) == x0[2]
        x[3](0) == x0[3]
        x[4](0) == x0[4]
        x[5](0) == x0[5]
        x[6](0) == x0[6]
        x[7](0) == x0[7]
        x[8](0) == x0[8]
        x[9](0) == x0[9]
        x[10](0) == x0[10]
        x[11](0) == x0[11]
        x[12](0) == x0[12]
        x[13](0) == x0[13]
        x[14](0) == x0[14]
        x[15](0) == x0[15]
        x[16](0) == x0[16]
        x[17](0) == x0[17]
        x[18](0) == x0[18]
        x[19](0) == x0[19]
        x[20](0) == x0[20]
        x[21](0) == x0[21]
        x[22](0) == x0[22]
        x[23](0) == x0[23]
        x[24](0) == x0[24]
        x[25](0) == x0[25]
        x[26](0) == x0[26]
        x[27](0) == x0[27]
        x[28](0) == x0[28]
        x[29](0) == x0[29]
        x[30](0) == x0[30]
        x[31](0) == x0[31]
        x[32](0) == x0[32]
        
        # Objective
        integral((x[1](t) - xbar)^2 + (u(t) - ubar)^2) → min
    end
    return ocp
end

# NMPC loop
function distillMPC(warmstart::Bool)
    # Initialize arrays to store results in
    x_vals = [zeros(t_points) for tray in 1:N]
    y_vals = [zeros(t_points) for tray in 1:N]
    u_vals = []
    nvars, ncons, total_times, nits, setup_times, sol_times, factor_times, ad_times = [], [], [], [], [], [], [], []

    # Initialize initial guesses
    xinit = ones(N)*xf  # Start value for state
    uinit = ubar           # Start value for control
    x0 = ones(N)*xf     # Initial condition for state

    # Initial compilation run
    dummy_x0 = x0
    ocp = nothing
    sol = nothing
    for i in 1:2
        println("Compile solve $i out of 2")
        ocp = OptControlModel(dummy_x0)
        sol = solve(
            ocp,
            init=sol,
            :exa,
            :madnlp,
            disc_method=:euler_implicit,
            grid_size=nsupps,
            exa_backend=CUDABackend(),
            output_file=logfile,
            print_level = MadNLP.WARN,
            tol=1e-8
        )

        # Simulate system forward & update initial condition for state
        tk = t_vals[i]
        tspan = (tk, tk+Δt)
        u = control(sol)
        uk = u(Δt)
        dummy_x0 = distillSim(dummy_x0, uk, tspan)
    end
    
    # Main NMPC loop
    for i in eachindex(t_vals)
        println("Current iteration: $(i) of $(length(t_vals))")

        # Create the OCP with updated initial conditions
        total_time = @elapsed ocp = OptControlModel(x0)

        # Solve the optimal control problem
        if i == 1 || warmstart == false
            println("Normal solve")
            total_time += @elapsed sol = solve(
                ocp,
                init=(state=xinit, control=uinit),
                :exa,
                :madnlp,
                disc_method=:euler_implicit,
                grid_size=nsupps,
                exa_backend=CUDABackend(),
                output_file=logfile,
                print_level=MadNLP.WARN,
                tol=1e-8
            )
        else
            println("Warmstart solve")
            # Overwrite sol to use in next iteration
            total_time += @elapsed sol = solve(
                ocp,
                init=sol,
                :exa,
                :madnlp,
                disc_method=:euler_implicit,
                grid_size=nsupps,
                exa_backend=CUDABackend(),
                output_file=logfile,
                print_level=MadNLP.WARN,
                mu_init = 1E-6,
                tol=1e-8
            )
        end

        # Check the solve status
        if !successful(sol)
            println("Status = $(status(sol))")
            println("Model not locally solved. Terminating.")
            break
        end

        # Get timing stats for current iteration
        nvar, ncon, its, sol_time, factor_time, ad_time = madnlp_stats(logfile)
        setup_time = total_time - sol_time
        push!(nvars, nvar)
        push!(ncons, ncon)
        push!(total_times, total_time)
        push!(nits, its)
        push!(setup_times, setup_time)
        push!(sol_times, sol_time)
        push!(factor_times, factor_time)
        push!(ad_times, ad_time)

        # Simulate system forward & update initial condition for state
        tk = t_vals[i]
        tspan = (tk, tk+Δt)
        u = control(sol)
        uk = u(Δt)
        push!(u_vals, uk)
        x0 = distillSim(x0, uk, tspan)
        yk = (α*x0)./(1 .+ (α - 1)*x0)

        # Store state/control values
        for j in 1:N
            x_vals[j][i] = value(x0[j])
            y_vals[j][i] = value(yk[j])
        end
    end

    # Extract the time statistics
    # make a table to save the results to
    its = collect(1:1:length(nvars))
    nrows = length(its)+1
    ncols = 9
    table = Matrix{String}(undef, nrows, ncols)
    table[1, :] = append!(["iteration", "nvar", "ncon", "nits", "total_time", "setup_time", "solve_time", "factor_time", "ad_time"])
    table[2:end, :] = string.(hcat(its, nvars, ncons, nits, total_times, setup_times, sol_times, factor_times,ad_times))

    # save the matrix as a CSV
    open("$(dirname(@__DIR__))/results/OptimalControl/distill-results.csv", "w") do io
        writedlm(io, table, ",")
    end
end

distillMPC(true)