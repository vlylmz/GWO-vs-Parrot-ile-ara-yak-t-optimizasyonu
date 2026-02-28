import numpy as np
import matplotlib.pyplot as plt


def smooth_curve(curve, window=5):
    """
    Moving average smoothing (for visualization only)
    """
    smoothed = np.copy(curve)
    for i in range(window, len(curve)):
        smoothed[i] = np.mean(curve[i - window:i])
    return smoothed


def print_optimizer_solution(problem_name, optimizer_name, best_cost, best_position):
    """
    Prints the solution found by an optimizer in a clean format
    """
    print(f"\n--- {problem_name} | {optimizer_name} RESULT ---")
    print(f"Best Cost : {best_cost:.6f}")


def plot_convergence(
    curves,
    labels,
    title="Convergence Curve",
    smooth=False,
    window=5
):
    """
    curves : list of convergence arrays
    labels : list of strings
    smooth : apply smoothing only for plotting
    """
    # Her grafik için yeni bir figür oluşturur
    plt.figure(figsize=(8, 5))

    for c, label in zip(curves, labels):
        if smooth:
            c = smooth_curve(c, window)
        plt.semilogy(c, linewidth=2, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Best Cost (log scale)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    

def format_cost(value):
    """
    Pretty print cost values
    """
    if abs(value) >= 1000:
        return f"{value:10.2f}"
    elif abs(value) >= 1:
        return f"{value:10.4f}"
    elif abs(value) >= 0.0001:
        return f"{value:10.6f}"
    else:
        return f"{value:10.8f}"


def print_results_table(results):
    """
    Ödev formatına uygun GENİŞLETİLMİŞ sonuç tablosu.
    """
    print("\n" + "="*105)
    print(f"{'Problem':12s} | {'Optimizer':8s} | {'Best Cost':>12s} | {'Mean Cost':>12s} | {'Std Dev':>12s} | {'Time (s)':>10s}")
    print("-" * 105)

    from collections import defaultdict
    best_per_problem = defaultdict(lambda: float("inf"))

    for r in results:
        best_per_problem[r["Problem"]] = min(
            best_per_problem[r["Problem"]],
            r["BestCost"]
        )

    for r in results:
        best_str = format_cost(r["BestCost"])
        mean_str = format_cost(r["MeanCost"])
        std_str  = format_cost(r["StdDev"])
        time_str = f"{r['Time']:.4f}"
        
        mark = " ★" if r["BestCost"] == best_per_problem[r["Problem"]] else ""
        
        print(
            f"{r['Problem']:12s} | "
            f"{r['Optimizer']:8s} | "
            f"{best_str}{mark:2s} | "
            f"{mean_str:>12s} | "
            f"{std_str:>12s} | "
            f"{time_str:>10s}"
        )
    print("="*105 + "\n")