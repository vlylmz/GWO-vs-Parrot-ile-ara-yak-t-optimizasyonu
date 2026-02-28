import numpy as np

def car_fuel_opt_fcn(v, params=None):
    """
    Araç Yakıt Optimizasyonu Maliyet Fonksiyonu
    Karar değişkeni: v[i] (m/s) -> Her bir yol segmentindeki araç hızları
    Amaç: Toplam yakıt tüketimini minimize ederken zaman ve konfor kısıtlarına uymak.
    """

    v = np.asarray(v, dtype=float)

    # --- Varsayılan Araç ve Yol Parametreleri ---
    if params is None:
        params = {
            "N": len(v),                   # Segment sayısı
            "segment_length_m": 100.0,     # Her segmentin uzunluğu (metre)
            "mass_kg": 1300.0,             # Araç kütlesi (kg)
            "Crr": 0.012,                  # Yuvarlanma direnci katsayısı
            "CdA": 0.62,                   # Hava direnci katsayısı * Ön alan (m^2)
            "rho_air": 1.225,              # Hava yoğunluğu (kg/m^3)
            "g": 9.81,                     # Yerçekimi ivmesi (m/s^2)
            "drivetrain_eff": 0.28,        # Aktarma organları ve motor verimi (%28)
            "fuel_LHV_J_per_L": 32e6,      # Yakıtın alt ısıl değeri (Joule/Litre)
            "grade": 0.0,                  # Yol eğimi (radyan)
            "v_eps": 0.5,                  # Sıfıra bölmeyi önlemek için minimum hız (m/s)
            "comfort_a_max": 2.0,          # Maksimum konforlu ivme sınırı (m/s^2)
            "target_trip_time_s": 360.0,   # Hedeflenen toplam seyahat süresi (saniye)
            "time_penalty_weight": 200.0,  # Zaman sapması için ceza katsayısı
            "acc_penalty_weight": 50.0,    # Sert ivmelenme/frenleme için ceza katsayısı
        }

    # Parametreleri yerel değişkenlere aktarma
    N = params["N"]
    L = params["segment_length_m"]
    m = params["mass_kg"]
    Crr = params["Crr"]
    CdA = params["CdA"]
    rho = params["rho_air"]
    g = params["g"]
    eff = params["drivetrain_eff"]
    LHV = params["fuel_LHV_J_per_L"]
    grade = params["grade"]
    v_eps = params["v_eps"]
    a_max = params["comfort_a_max"]
    w_time = params["time_penalty_weight"]
    w_acc = params["acc_penalty_weight"]
    T_target = params["target_trip_time_s"]

    # Hatalı giriş kontrolü (boyut uyumsuzluğu)
    if len(v) != N:
        return 1e12 + 1e9 * abs(len(v) - N)

    # Hızın sıfır olmasını engelle (zaman hesaplamasında payda olduğu için)
    v_safe = np.maximum(v, v_eps)

    # --- Zaman ve İvme Hesaplamaları ---
    # Her segmentte geçen süre: t = yol / hız
    dt = L / v_safe
    total_time = float(np.sum(dt))

    # İvme hesabı: a = (v_son^2 - v_ilk^2) / (2 * yol)
    a = np.zeros(N)
    a[1:] = (v_safe[1:]**2 - v_safe[:-1]**2) / (2.0 * L)

    # --- Kuvvet Hesaplamaları (Fizik Modeli) ---
    F_roll = m * g * Crr * np.cos(grade)          # Yuvarlanma Direnci Kuvveti
    F_grade = m * g * np.sin(grade)               # Yerçekimi (Eğim) Kuvveti
    F_aero = 0.5 * rho * CdA * (v_safe**2)        # Aerodinamik Hava Direnci
    F_inertia = m * a                             # Eylemsizlik Kuvveti (F=m*a)

    # Tekerleklere iletilmesi gereken toplam çekiş kuvveti
    F_trac = F_roll + F_grade + F_aero + F_inertia
    
    # Negatif kuvvetleri (frenleme) yakıt tüketimine katma (İdeal enerji geri kazanımı yok sayılır)
    F_trac_pos = np.maximum(F_trac, 0.0)

    # --- Enerji ve Yakıt Hesabı ---
    P = F_trac_pos * v_safe                       # Güç: P = Kuvvet * Hız
    E = float(np.sum(P * dt))                     # Toplam Enerji (Joule): E = Güç * Zaman

    # Yakıt Tüketimi (Litre): Enerji / (Verim * Yakıt Enerji Değeri)
    fuel_L = E / (max(eff, 1e-6) * LHV)

    # --- Cezalar (Penalties) ---
    # 1. Konfor Cezası: Belirlenen ivme sınırı (a_max) aşılırsa ceza puanı ekle
    acc_violation = np.maximum(np.abs(a) - a_max, 0.0)
    penalty_acc = w_acc * float(np.sum(acc_violation**2))

    # 2. Zaman Cezası: Hedeflenen süreden (T_target) ne kadar sapıldığının karesel cezası
    penalty_time = w_time * float(
        ((total_time - T_target) / max(T_target, 1.0))**2
    )

    # Toplam Maliyet = Yakıt Tüketimi + İvme Cezası + Zaman Cezası
    return float(fuel_L + penalty_acc + penalty_time)