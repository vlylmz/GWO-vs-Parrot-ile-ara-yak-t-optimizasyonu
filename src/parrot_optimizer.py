import numpy as np

def parrot_optimizer(
    cost_function,
    n_var,
    var_min,
    var_max,
    max_iter=300,
    pop_size=30,
    seed=None
):
    """
    Advanced Parrot Optimizer (APO)
    - Keşif (Exploration): Rastgele gürültü ve popülasyon sıfırlama ile geniş alan tarama.
    - Sömürü (Exploitation): gBest'e odaklanma ve sadece daha iyi çözümleri kabul etme (Greedy).
    """

    if seed is not None:
        np.random.seed(seed)

    # ---------- Başlangıç (Global Keşif) ----------
    # İlk adımda papağanlar arama uzayına tamamen rastgele dağıtılarak 
    # KEŞİF (Exploration) süreci başlatılır.
    parrots = np.random.uniform(var_min, var_max, (pop_size, n_var))
    fitness = np.array([cost_function(p) for p in parrots])

    best_idx = np.argmin(fitness)
    gbest = parrots[best_idx].copy()
    gbest_cost = fitness[best_idx]

    convergence = np.zeros(max_iter)
    
    # Sıkışma takibi (Yerel minimumdan kaçış için gerekli)
    stagnation_counter = 0
    last_best_cost = float('inf')

    # ---------- Ana Döngü ----------
    for t in range(max_iter):
        
        # 1. Non-Lineer Parametre Güncellemesi
        # 'w' (atalet ağırlığı): Başta yüksektir (Keşif odaklı), sona doğru azalır (Sömürü odaklı).
        w = 0.9 * (1 - (t / max_iter)**2) 
        c1 = 1.5 # Sosyal bileşen: En iyiye olan çekim gücü
        
        # 'noise_scale': İterasyon ilerledikçe azalır, böylece algoritma 
        # başlangıçta büyük sıçramalar (Keşif) yaparken sonunda küçük adımlarla (Sömürü) iyileştirme yapar.
        noise_scale = 0.2 * (1 - t / max_iter)

        for i in range(pop_size):
            r = np.random.rand(n_var)
            
            # --- KEŞİF (Exploration) ---
            # Gaussian Noise: Papağanlara rastgele hareket kabiliyeti kazandırarak 
            # bilinmeyen bölgeleri keşfetmelerini sağlar.
            noise = noise_scale * np.random.randn(n_var) * (var_max - var_min)

            # --- SÖMÜRÜ (Exploitation) ---
            # Pozisyon Güncelleme: Papağan mevcut konumu ile global en iyi (gbest) 
            # arasında bir denge kurarak hedefe odaklanır.
            candidate = (
                w * parrots[i]
                + c1 * r * (gbest - parrots[i])
                + noise
            )

            candidate = np.clip(candidate, var_min, var_max)
            candidate_cost = cost_function(candidate)

            # Greedy Selection (Açgözlü Seçim): SÖMÜRÜYÜ güçlendirir.
            # Sadece mevcut durumdan daha iyi bir aday bulunursa konum güncellenir.
            if candidate_cost < fitness[i]:
                parrots[i] = candidate
                fitness[i] = candidate_cost

                if candidate_cost < gbest_cost:
                    gbest_cost = candidate_cost
                    gbest = candidate.copy()

        # 2. gBest Pertürbasyonu (Local Escape / Yerel Kaçış)
        # Sıkışmayı önlemek için %10 şansla en iyi çözümü sarsarak (Perturbation)
        # yeni bölgelerde daha iyi bir nokta olup olmadığını KEŞFEDER.
        if np.random.rand() < 0.1:
            perturbation_scale = 0.05 * (1 - t / max_iter)
            temp_gbest = gbest + perturbation_scale * (var_max - var_min) * np.random.randn(n_var)
            temp_gbest = np.clip(temp_gbest, var_min, var_max)
            temp_cost = cost_function(temp_gbest)
            
            if temp_cost < gbest_cost:
                gbest = temp_gbest
                gbest_cost = temp_cost

        # 3. Akıllı Çeşitlilik Enjeksiyonu (Stagnation Detection)
        # Eğer çözüm gelişmiyorsa (Stagnation), algoritma KEŞİF moduna geçer.
        if abs(last_best_cost - gbest_cost) < 1e-9:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            last_best_cost = gbest_cost

        # Eğer 10 iterasyondur ilerleme yoksa, en başarısız papağanları resetler.
        # Bu, çeşitliliği artırarak algoritmayı yerel tuzaklardan kurtarır (KEŞİF).
        if stagnation_counter >= 10 and t < 0.8 * max_iter:
            n_reset = max(1, int(0.2 * pop_size))
            sorted_indices = np.argsort(fitness)
            worst_indices = sorted_indices[-n_reset:] 
            
            parrots[worst_indices] = np.random.uniform(
                var_min, var_max, (n_reset, n_var)
            )
            for idx in worst_indices:
                fitness[idx] = cost_function(parrots[idx])
            
            stagnation_counter = 0

        # Yakınsama Kaydı
        if t == 0:
            convergence[t] = gbest_cost
        else:
            convergence[t] = min(convergence[t-1], gbest_cost)

    return gbest, gbest_cost, convergence