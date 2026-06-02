import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/MPCGPU/distillation_mpcgpu_nmpc.csv")
prefix = "results/MPCGPU"

xbar = 0.8958
ubar = 2.51459

plt.figure()
plt.plot(df["t_min"], df["x1"], label="x1", linewidth=3)
plt.axhline(xbar, linestyle="--", label="Steady-state value", linewidth=3, color="#ff7846")
plt.xlabel("Time (min)")
plt.ylabel("Tray 1 mole fraction")
plt.ylim(0.65, 0.91)
plt.legend()
plt.savefig(f"{prefix}/distill_x1_mpcgpu.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(df["t_min"], df["y32"], label="", linewidth=3, color="#DB5C87")
plt.xlabel("Time (min)")
plt.ylabel("Tray 32 mole fraction")
plt.ylim(0.15, 0.52)
plt.savefig(f"{prefix}/distill_y32_mpcgpu.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(df["t_min"], df["u"], label="r", linewidth=3)
plt.axhline(ubar, linestyle="--", label="Steady-state value", linewidth=3, color="#ff7846")
plt.xlabel("Time (min)")
plt.ylabel("Reflux ratio")
plt.ylim(2.51, 2.67)
plt.legend()
plt.savefig(f"{prefix}/distill_Control_mpcgpu.png", dpi=200, bbox_inches="tight")
plt.show()

final = df.iloc[-1]
trays = list(range(1, 33))
x_vals = [final[f"x{i}"] for i in trays]
y_vals = [final[f"y{i}"] for i in trays]

plt.figure()
plt.plot(trays, x_vals, label="x", linewidth=3)
plt.plot(trays, y_vals, label="y", linewidth=3, color="#DB5C87")
plt.xlabel("Trays")
plt.ylabel("Mole fraction")
plt.ylim(0.0, 1.0)
plt.legend()
plt.savefig(f"{prefix}/distill_moleFrac_mpcgpu.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(df["iter"], df["solve_time_s"], linewidth=3)
plt.xlabel("NMPC iteration")
plt.ylabel("OCP solve time (s)")
plt.savefig(f"{prefix}/distill_solve_time_mpcgpu.png", dpi=200, bbox_inches="tight")
plt.show()