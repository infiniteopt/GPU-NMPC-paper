include("distillation/distillation.jl")
include("pde/pde.jl")

# Define MPC parameters for benchmarks 
# In the order of: [t0, tf, Δt, Tp, Tc]
initial_distill = [0, 2, 2, 180, 180]   # For initial compilation of distillation benchmark
initial_pde = [0.0, 0.08, 0.08, 1, 1]     # For initial compilation of PDE benchmark
MPC_distill = [0, 300, 2, 180, 180]
MPC_pde = [0.0, 3, 0.08, 1, 1]

# Settings for keyword arguments
# In the form: [warmstart::Bool, updateParams::Bool, show_plots::Bool]
pl = false
settings = [
    [false, false, pl],
    [false, true, pl],
    [true, false, pl],
    [true, true, pl]
]

backends = ["Ipopt", "MOI", "ExaModelsCPU", "ExaModelsGPU"]

# Run the case studies
for backend in backends
    for setting in settings
        WS, UP, PL = setting
        for run in 1:3
            println("Current backend: $backend")
            Distillation.distillMPC(backend, initial_distill, run, warmstart=WS, param_updates=UP, show_plots=PL, compile=true)    # For initial compilation
            Distillation.distillMPC(backend, MPC_distill, run, warmstart=WS, param_updates=UP, show_plots=PL)    # Proper solve

            PDE.pdeMPC(backend, initial_pde, warmstart=WS, run, param_updates=UP, show_plots=PL, compile=true)   # For initial compilation
            PDE.pdeMPC(backend, MPC_pde, warmstart=WS, run, param_updates=UP, show_plots=PL)  # Proper solve
        end
    end
end