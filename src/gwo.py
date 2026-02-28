import numpy as np

def gwo(
    cost_function,
    n_var,
    var_min,
    var_max,
    max_iter=300,
    pop_size=30,
    seed=None
):
    """
    Enhanced Grey Wolf Optimizer (GWO)
    - Keşif (Exploration): Geniş alanları tarama yeteneği.
    - Sömürü (Exploitation): Bulunan en iyi bölgedeki hassas arama.
    """

    if seed is not None:
        np.random.seed(seed)

    # ---------- Başlangıç (Rastgele Arama Alanı Oluşturma) ----------
    # Popülasyonu rastgele dağıtarak geniş bir KEŞİF (Exploration) başlatılır.
    wolves = np.random.uniform(var_min, var_max, (pop_size, n_var))
    fitness = np.array([cost_function(w) for w in wolves])

    # En iyi 3 kurt hiyerarşiyi belirler: Alfa, Beta ve Delta
    idx = np.argsort(fitness)
    alpha = wolves[idx[0]].copy()
    beta  = wolves[idx[1]].copy()
    delta = wolves[idx[2]].copy()
    alpha_cost = fitness[idx[0]]

    # Global en iyi çözümün hafızada tutulması
    gbest = alpha.copy()
    gbest_cost = alpha_cost

    convergence = np.zeros(max_iter)

    # ---------- Ana Döngü ----------
    for t in range(max_iter):

        # 'a' parametresi: SÖMÜRÜ ve KEŞİF arasındaki dengeyi kurar.
        # İterasyon ilerledikçe azalır: Başta yüksek (keşif), sonda düşük (sömürü).
        a = 2 * (1 - (t / max_iter) ** 2)

        for i in range(pop_size):
            for j in range(n_var):

                # --- SÖMÜRÜ (Exploitation) MEKANİZMASI ---
                # Kurtlar, Alfa, Beta ve Delta'nın (liderlerin) etrafında toplanır.
                # A ve C katsayıları, liderlerin etrafındaki çemberi daraltarak 
                # hedefe (optimuma) odaklanmayı sağlar.
                
                # Alfa kurduna göre güncelleme
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                X1 = alpha[j] - A1 * abs(C1 * alpha[j] - wolves[i, j])

                # Beta kurduna göre güncelleme
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                X2 = beta[j] - A2 * abs(C2 * beta[j] - wolves[i, j])

                # Delta kurduna göre güncelleme
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                X3 = delta[j] - A3 * abs(C3 * delta[j] - wolves[i, j])

                # Yeni pozisyon: 3 liderin ortalaması (Sömürü yoğunlaşır)
                wolves[i, j] = (X1 + X2 + X3) / 3.0

        # Sınırların dışına çıkan kurtları arama alanına geri çekme
        wolves = np.clip(wolves, var_min, var_max)

        # Fitness güncelleme ve hiyerarşiyi yeniden belirleme
        fitness = np.array([cost_function(w) for w in wolves])
        idx = np.argsort(fitness)

        alpha = wolves[idx[0]].copy()
        beta  = wolves[idx[1]].copy()
        delta = wolves[idx[2]].copy()
        alpha_cost = fitness[idx[0]]

        # --- gBest (Global En İyi) Güncelleme ---
        if alpha_cost < gbest_cost:
            gbest_cost = alpha_cost
            gbest = alpha.copy()

        # --- Alpha Perturbation (Ekstra KEŞİF) ---
        # Alfa kurdu bir noktada takılı kalırsa (local minimum), 
        # küçük bir sarsıntı ile yeni bölgeleri KEŞFETMESİ sağlanır.
        if t < 0.6 * max_iter and np.random.rand() < 0.05:
            candidate = alpha + 0.1 * (var_max - var_min) * np.random.randn(n_var)
            candidate = np.clip(candidate, var_min, var_max)
            candidate_cost = cost_function(candidate)

            if candidate_cost < gbest_cost:
                gbest_cost = candidate_cost
                gbest = candidate.copy()

        # --- Çeşitlilik Enjeksiyonu (Gelişmiş KEŞİF) ---
        # Popülasyonun bir kısmını rastgele sıfırlayarak algoritmanın 
        # tek bir noktaya sıkışıp kalmasını (Stagnation) önler.
        if t % 20 == 0 and 0 < t < 0.7 * max_iter:
            n_reset = max(1, int(0.1 * pop_size))
            idx_reset = np.random.choice(pop_size, n_reset, replace=False)
            wolves[idx_reset] = np.random.uniform(
                var_min, var_max, (n_reset, n_var)
            )

        # Yakınsama (Convergence) kaydı: Her zaman en iyiyi korur (Monotonik)
        if t == 0:
            convergence[t] = gbest_cost
        else:
            convergence[t] = min(convergence[t-1], gbest_cost)

    return gbest, gbest_cost, convergence