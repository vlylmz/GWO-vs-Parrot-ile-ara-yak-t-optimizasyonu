import numpy as np
import time
import matplotlib.pyplot as plt

# Kendi yazdığımız yardımcı modülleri içe aktarıyoruz
from benchmark_functions import get_benchmark_functions
from car_fuel_opt_problem import car_fuel_opt_fcn
from gwo import gwo
from parrot_optimizer import parrot_optimizer
from results_utils import plot_convergence, print_results_table

# -------- NumPy Çıktı Ayarları --------
# Konsola basılan sayıların bilimsel gösterim yerine (0.0001 gibi) 
np.set_printoptions(
    suppress=True,
    precision=4,
    floatmode="fixed"
)

def plot_best_speed_profile(best_v, problem_name, optimizer_name):
    """
    Optimizasyon sonucunda elde edilen en iyi 'Hız Profilini' (Hız-Yol Grafiği) çizer.
    Bu grafik, aracın yol boyunca hangi segmentte hangi hızda gittiğini gösterir.
    """
    plt.figure(figsize=(10, 5))
    segments = np.arange(1, len(best_v) + 1)
    
    # Basamaklı grafik (step plot) kullanımı, segment bazlı hız değişimini daha iyi yansıtır.
    plt.step(segments, best_v, where='post', color='darkblue', linewidth=2, label=f'Optimum Hız ({optimizer_name})')
    
    # Kıyaslama için ortalama hızı kesikli kırmızı çizgi ile ekliyoruz.
    plt.axhline(y=np.mean(best_v), color='red', linestyle='--', label=f'Ortalama Hız: {np.mean(best_v):.2f} m/s')
    
    plt.title(f"{problem_name} - {optimizer_name} Tarafından Bulunan Hız Planı")
    plt.xlabel("Yol Segmenti (Her biri 100m)")
    plt.ylabel("Hız (m/s)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()

def run_multiple_times(problem, optimizer, n_runs=30):
    """
    Algoritmaların kararlılığını ölçmek için problemi n_runs kez (varsayılan 30) çalıştırır.
    İstatistiksel verileri ve en iyi bulunan çözümü (hız vektörü) döndürür.
    """
    costs, curves, times = [], [], []
    absolute_best_cost = float("inf")
    absolute_best_pos = None 
    
    print(f"> {optimizer.__name__.upper()} calistiriliyor ({problem['Name']})... ", end="")
    
    for i in range(n_runs):
        start_time = time.time()
        
        # Algoritmayı çağırıyoruz: optimizer fonksiyonu (GWO veya APO)
        # Bize: en iyi pozisyon (hızlar), en iyi maliyet ve yakınsama eğrisini döndürür.
        best_pos, best_cost_run, curve = optimizer(
            cost_function=problem["CostFunction"],
            n_var=problem["nVar"],
            var_min=problem["VarMin"],
            var_max=problem["VarMax"],
            max_iter=300, 
            pop_size=30,  
            seed=None 
        )
        
        end_time = time.time()
        
        # Eğer bu çalıştırma şimdiye kadarki en iyi maliyeti verdiyse, o çözümü sakla.
        if best_cost_run < absolute_best_cost:
            absolute_best_cost = best_cost_run
            absolute_best_pos = best_pos
            
        costs.append(best_cost_run)
        curves.append(curve)
        times.append(end_time - start_time)
        print(".", end="", flush=True) 

    print(" Tamamlandı.")
    
    # İstatistiksel hesaplamalar: Ortalama, Standart Sapma ve Ortalama Yakınsama Eğrisi
    mean_cost = np.mean(costs)
    std_cost = np.std(costs)
    avg_curve = np.mean(np.array(curves), axis=0)
    avg_time = np.mean(times)
    
    return absolute_best_cost, mean_cost, std_cost, avg_curve, avg_time, absolute_best_pos


def main():

    results = []    # Tablo oluşturmak için tüm sonuçların tutulduğu liste
    N_RUNS = 30     # 30 bağımsız çalıştırma 

    # --- 1. BÖLÜM: MATEMATİKSEL BENCHMARK TESTLERİ ---
    # Algoritmaların genel performansını ölçmek için standart fonksiyonlar (Sphere, Ackley vb.)
    problems = get_benchmark_functions(default_dim=30)

    for problem in problems:
        # Klasik GWO algoritması ile test [cite: 27]
        g_best, g_mean, g_std, g_curve, g_time, _ = run_multiple_times(problem, gwo, n_runs=N_RUNS)
        
        # 2024 sonrası yeni nesil Parrot Optimizer ile test [cite: 30]
        p_best, p_mean, p_std, p_curve, p_time, _ = run_multiple_times(problem, parrot_optimizer, n_runs=N_RUNS)

        # Tabloya eklemek üzere sonuçları sözlük yapısında saklıyoruz
        results.append({"Problem": problem["Name"], "Optimizer": "GWO", "BestCost": g_best, "MeanCost": g_mean, "StdDev": g_std, "Time": g_time})
        results.append({"Problem": problem["Name"], "Optimizer": "PARROT", "BestCost": p_best, "MeanCost": p_mean, "StdDev": p_std, "Time": p_time})

        # Her test fonksiyonu için yakınsama grafiğini çiziyoruz [cite: 37]
        plot_convergence([g_curve, p_curve], [f"GWO (Avg)", f"Parrot (Avg)"], title=f"Convergence – {problem['Name']}")

    # --- 2. BÖLÜM: GERÇEK DÜNYA OPTİMİZASYONU (ARAÇ YAKIT TÜKETİMİ) ---
    print("\n--- Problem: CarFuel ---")
    car_problem = {
        "Name": "CarFuel",
        "CostFunction": car_fuel_opt_fcn, # Fizik tabanlı yakıt maliyet fonksiyonu [cite: 19]
        "nVar": 30,                       # 30 farklı yol segmenti için hız değişkeni [cite: 17]
        "VarMin": 5.0,                    # Minimum hız sınırı (m/s) [cite: 35]
        "VarMax": 30.0,                   # Maksimum hız sınırı (m/s) [cite: 35]
    }

    # Gerçek dünya problemi için algoritmaları çalıştırıyoruz
    g_best, g_mean, g_std, g_curve, g_time, g_best_pos = run_multiple_times(car_problem, gwo, n_runs=N_RUNS)
    p_best, p_mean, p_std, p_curve, p_time, p_best_pos = run_multiple_times(car_problem, parrot_optimizer, n_runs=N_RUNS)

    results.append({"Problem": "CarFuel", "Optimizer": "GWO", "BestCost": g_best, "MeanCost": g_mean, "StdDev": g_std, "Time": g_time})
    results.append({"Problem": "CarFuel", "Optimizer": "PARROT", "BestCost": p_best, "MeanCost": p_mean, "StdDev": p_std, "Time": p_time})

    # --- 3. BÖLÜM: SONUÇLARIN GÖRSELLEŞTİRİLMESİ VE RAPORLANMASI ---
    
    # Konsola en iyi hız profillerini (sayısal değerler) yazdırıyoruz
    print("\n" + "="*50)
    print("GWO EN IYI HIZ PROFILI (m/s):")
    print(g_best_pos) # Algoritmanın bulduğu en verimli hız dizisi
    print("\nPARROT EN IYI HIZ PROFILI (m/s):")
    print(p_best_pos)
    print("="*50)

    # Yakıt optimizasyonu yakınsama grafiği [cite: 46]
    plot_convergence([g_curve, p_curve], ["GWO (Avg)", "Parrot (Avg)"], title="Convergence – Car Fuel Optimization")

    # Yol bazlı hız değişim grafikleri
    plot_best_speed_profile(g_best_pos, "Car Fuel Optimization", "GWO")
    plot_best_speed_profile(p_best_pos, "Car Fuel Optimization", "Parrot")

    # Karşılaştırmalı sonuç tablosu (Best, Mean, Std, Time) 
    print_results_table(results)
    
    print("Tüm grafikler gösteriliyor...")
    plt.show()

if __name__ == "__main__":
    main()