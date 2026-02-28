# Araç Yakıt Tüketimi Optimizasyonu (Meta-Sezgisel)

Bu proje, bir araç için **yol segmentlerine göre hız profilini** optimize ederek toplam yakıt tüketimini minimize etmeyi amaçlar. 
Çözüm için iki meta-sezgisel algoritma karşılaştırılmıştır:

- **Grey Wolf Optimizer (GWO)** (2014)
- **Advanced Parrot Optimizer (APO / Parrot Optimizer)** (2024)

Ayrıca, algoritmaların davranışını görmek için standart **benchmark fonksiyonları** (Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel, Zakharov) üzerinde de test yapılır.

## İçerik
- `src/main.py`: Benchmark + gerçek problem deneylerini çalıştırır, yakınsama eğrilerini ve hız profillerini çizer.
- `src/car_fuel_opt_problem.py`: Araç yakıt tüketimi maliyet fonksiyonu (zaman ve konfor ceza terimleri ile).
- `src/gwo.py`: Grey Wolf Optimizer implementasyonu.
- `src/parrot_optimizer.py`: Advanced Parrot Optimizer implementasyonu.
- `src/benchmark_functions.py`: Benchmark fonksiyonları.
- `reports/odev-2-veli-yilmaz-rapor.docx`: Ödev raporu (metodoloji ve sayısal sonuçlar).

## Kurulum
Python 3.10+ önerilir.

```bash
pip install -r requirements.txt
```

## Çalıştırma
```bash
python src/main.py
```

> Not: Kod, grafiklerini `matplotlib` ile açar (plot pencereleri). Uzak sunucuda çalıştırıyorsan `matplotlib` backend ayarı gerekebilir.

## Çıktılar
Çalıştırma sonunda:
- Benchmark fonksiyonları için **yakınsama eğrileri**
- Araç yakıt optimizasyonu için **yakınsama eğrisi**
- GWO ve Parrot için **optimum hız profilleri** grafikleri
ekrana çizilir ve konsola özet tablo basılır.

## Parametreler
Varsayılan deney ayarları:
- Popülasyon: 30
- İterasyon: 300
- Çalıştırma: 30 (istatistik için)

İstersen `src/main.py` içinden değiştirebilirsin.

## Lisans
Eğitim ve portfolyo amaçlıdır.
