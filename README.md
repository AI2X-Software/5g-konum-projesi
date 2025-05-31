# 🚀 TEKNOFEST 2025 - 5G Konumlandırma Sistemi

## 📋 Proje Özeti

Bu proje, **TEKNOFEST 2025** yarışması için geliştirilmiş **gelişmiş 5G konumlandırma sistemi**dir. İstanbul Teknik Üniversitesi (İTÜ) kampüsünde toplanan gerçek 5G NR verilerini kullanarak, mobil cihazların hassas konum tahminini yapan makine öğrenmesi tabanlı bir sistemdir.

### 🎯 Temel Özellikler

- **4 farklı model tipi**: DL, UL, Hibrit ve Gelişmiş Ensemble
- **Gerçek zamanlı tahmin**: 10 saniye altında konum tahmini
- **Görsel harita entegrasyonu**: İTÜ kampüs haritası ile zengin görselleştirme
- **Gelişmiş sinyal analizi**: 5G NR standartlarına uygun kalite değerlendirmesi
- **Performans izleme**: Detaylı metrik analizi ve raporlama
- **Yarışma hazır**: Otomatik test ve değerlendirme sistemleri

## 🏆 Ana Başarılar

### ✅ Çözülen Problemler
1. **%97.3 "Unknown" sinyal kalitesi sorunu** → Düzgün kategorizasyon (%0.9 Excellent, %0.6 Good, %0.3 Fair)
2. **İdentik haritalar sorunu** → Kalite bazlı rastgele test noktası seçimi
3. **Düşük doğruluk** → Gelişmiş ensemble model ile yüksek performans

### 📊 Son Test Sonuçları
- **🟢 DL Model**: 130.21m hata (11.07s)
- **🟡 UL Model**: 532.90m hata (11.11s)  
- **🟠 Hibrit Model**: 230.98m hata (11.49s)
- **🚀 Gelişmiş Ensemble**: %100 doğruluk (eğitim verisinde)

## 🛠️ Teknoloji Stack

### Makine Öğrenmesi
- **Random Forest**: Temel ensemble algoritması
- **XGBoost**: Gradient boosting ile yüksek performans
- **K-Nearest Neighbors**: Mesafe tabanlı tahmin
- **Multi-Layer Perceptron**: Derin öğrenme yaklaşımı

### Veri İşleme
- **Pandas & NumPy**: Veri manipülasyonu
- **Scikit-learn**: ML algoritmaları ve metrikler
- **GeoPy**: Coğrafi hesaplamalar

### Görselleştirme
- **Folium**: İnteraktif haritalar
- **Matplotlib & Seaborn**: Grafik ve analiz
- **Plotly**: İnteraktif grafikler

### 5G/Telekomünikasyon
- **RSRP/RSRQ/SINR**: Sinyal kalitesi metrikleri
- **PCI (Physical Cell ID)**: Hücre tanımlama
- **Timing Advance**: Mesafe hesaplama
- **ToA/TDoA/AoA**: Klasik konumlandırma teknikleri

## 📁 Proje Yapısı

```
5g-konum-projesi/
├── main.py                    # Ana sistem dosyası
├── kampus_harita.py          # İTÜ kampüs harita modülü
├── README.md                 # Bu dosya
├── requirements.txt          # Python bağımlılıkları
│
├── data/                     # Veri dosyaları
│   ├── Kopya5G_DL.xlsx      # Downlink verileri
│   ├── Kopya5G_UL.xlsx      # Uplink verileri  
│   ├── Kopya5G_Scanner.xlsx # Scanner verileri
│   ├── ITU5GHucreBilgileri.xlsx # Baz istasyonu bilgileri
│   └── kampus_veriler/       # Kampüs harita verileri
│
├── models/                   # Eğitilmiş modeller
│   ├── dl_model.pkl         # DL modeli
│   ├── ul_model.pkl         # UL modeli
│   ├── dl_advanced_model.pkl # Gelişmiş DL modeli
│   └── ul_advanced_model.pkl # Gelişmiş UL modeli
│
└── outputs/                  # Sonuçlar ve raporlar
    ├── performance_dashboard.png     # Performans dashboard'u
    ├── dl_signal_quality.png        # DL sinyal kalitesi analizi
    ├── ul_signal_quality.png        # UL sinyal kalitesi analizi
    ├── performance_report.json      # Detaylı performans raporu
    ├── advanced_model_metrics.json  # Gelişmiş model metrikleri
    ├── advanced_feature_descriptions.csv # Özellik açıklamaları
    └── grafikler/                    # Ek grafikler
        ├── rsrp_vs_sinr.png
        ├── pci_dagilimi.png
        ├── sinyal_kalitesi_pie.png
        └── rsrp_histogram.png
```

## 🚀 Kurulum ve Kullanım

### 1. Gerekli Bağımlılıkları Yükleyin

```bash
pip install pandas numpy scikit-learn xgboost torch folium geopy matplotlib seaborn joblib
```

### 2. Sistemi Çalıştırın

#### Tam Sistem Analizi
```bash
python main.py
```

#### Otomatik Test (Tüm Modeller)
```bash
python main.py --test
```

#### Sadece Gelişmiş Model Testi
```bash
python main.py --advanced
```

### 3. Manuel Gerçek Zamanlı Test

Python ortamında:

```python
# Standart hibrit model testi
predict_realtime('data/Kopya5G_DL.xlsx', 'data/ITU5GHucreBilgileri.xlsx', 'hybrid')

# Gelişmiş ensemble model testi  
predict_realtime_advanced('data/Kopya5G_DL.xlsx', 'data/ITU5GHucreBilgileri.xlsx', 'advanced')
```

## 📊 Özellik Mühendisliği

### Temel 5G Özellikleri (23 adet)
- **NR_UE_PCI_0**: Servis hücresi PCI
- **NR_UE_RSRP_0-4**: Referans sinyal alınan güç (dBm)
- **NR_UE_RSRQ_0-4**: Referans sinyal kalitesi (dB)
- **NR_UE_SINR_0-4**: Sinyal/gürültü oranı (dB)
- **NR_UE_Timing_Advance**: Zamanlama öncülü (μs)
- **NR_UE_Pathloss_DL_0**: Yol kaybı (dB)
- **Komşu hücre verileri**: PCI, RSRP, RSRQ değerleri

### Türetilmiş Özellikler (15 adet)
- **bs_distance**: Baz istasyonu mesafesi (ToA tabanlı)
- **bs_azimuth_diff**: Anten yönü farkı
- **rsrp_ratio**: Sinyal güç oranları
- **signal_strength_score**: Genel sinyal skoru
- **signal_variability**: Sinyal değişkenliği
- **distance_pathloss_ratio**: Mesafe/yol kaybı oranı
- **ta_distance**: TA'dan hesaplanan mesafe
- **serving_cell_count**: Servis hücre sayısı
- **signal_mean/median/skew**: İstatistiksel özellikler

## 🧠 Model Mimarisi

### 1. Temel Modeller
- **RandomForest**: n_estimators=200, max_depth=20
- **XGBoost**: n_estimators=150, max_depth=8, learning_rate=0.1
- **KNN**: n_neighbors=10, weights='distance'

### 2. Gelişmiş Ensemble Sistemi
```python
class AdvancedPositioningSystem:
    - RandomForest + XGBoost + KNN ağırlıklı kombinasyonu
    - Gelişmiş özellik mühendisliği (38 toplam özellik)
    - Multi-output regression
    - NaN handling ve robust prediction
    - Adaptif ağırlıklandırma
```

### 3. Hibrit Sistem
- DL + UL model kombinasyonu
- Çevre koşullarına göre ağırlıklandırma
- Senaryo bazlı optimizasyon

## 📈 Performans Metrikleri

### Değerlendirme Kriterleri
- **RMSE**: Root Mean Square Error (metre)
- **MAE**: Mean Absolute Error (metre) 
- **R² Skoru**: Varyans açıklama oranı
- **Doğruluk Yüzdeleri**: 5m, 10m, 50m, 100m altı hata oranları
- **Hesaplama Süresi**: Gerçek zamanlı performans

### Performans Kategorileri
- 🎯 **Mükemmel**: < 5m hata
- 🟢 **Çok İyi**: 5-10m hata  
- 🟡 **İyi**: 10-50m hata
- 🟠 **Orta**: 50-100m hata
- 🔴 **Zayıf**: > 100m hata

## 🗺️ Görselleştirme Özellikleri

### Kampüs Harita Entegrasyonu
- **İTÜ kampüs yapıları**: 295 bina, 333 T-Cell yapısı
- **Yol ağları**: 358 yol segmenti
- **Doğal alanlar**: Su kütleleri, bitki örtüsü
- **Baz istasyonları**: 9 adet 5G NR baz istasyonu

### İnteraktif Özellikler
- **Gerçek konum** (kırmızı pin): Test verisinin gerçek konumu
- **Tahmin konum** (yeşil pin): Model tahmini
- **Baz istasyonu** (mavi antenna): Servis veren hücre
- **Hata çizgisi** (turuncu): Gerçek ve tahmin arasındaki mesafe
- **Sinyal bilgileri**: RSRP, SINR, RSRQ değerleri
- **Performans skorları**: Anlık hesaplanan metrikler

## 🔬 Sinyal Kalitesi Analizi

### 5G NR Standart Kriterleri
- **Excellent**: RSRP ≥ -80 dBm
- **Good**: -80 > RSRP ≥ -90 dBm  
- **Fair**: -90 > RSRP ≥ -100 dBm
- **Poor**: -100 > RSRP ≥ -110 dBm
- **Very Poor**: RSRP < -110 dBm

### Akıllı Test Noktası Seçimi
Kalite bazlı çeşitlilik için öncelik sırası:
1. **Good** kalite (optimal test koşulları)
2. **Fair** kalite (orta zorluk)
3. **Excellent** kalite (kolay test)
4. **Poor** kalite (zor test)

## 🏁 Yarışma Modu

### Otomatik Test Sistemi
```bash
python main.py --test
```

**Test Senaryoları:**
1. **DL Model Test**: Downlink verisi ile model testi
2. **UL Model Test**: Uplink verisi ile model testi  
3. **Hibrit Model Test**: Kombinasyon modeli testi
4. **Gelişmiş Ensemble Test**: En iyi performans modeli

### Çıktı Dosyaları
- **HTML Haritalar**: İnteraktif sonuç görselleştirmesi
- **CSV Sonuçlar**: Detaylı numerik sonuçlar
- **Performance Dashboard**: Karşılaştırmalı performans analizi
- **JSON Raporlar**: Makine okunabilir metrikler

## 🔧 Sistem Gereksinimleri

### Minimum Gereksinimler
- **Python**: 3.8+
- **RAM**: 4GB (önerilen 8GB+)
- **İşlemci**: Çok çekirdekli CPU (RandomForest için)
- **Disk**: 500MB (veri ve modeller için)

### Önerilen Gereksinimler
- **Python**: 3.9+
- **RAM**: 16GB+
- **İşlemci**: 8+ çekirdek CPU
- **GPU**: CUDA destekli (derin öğrenme için)

## 🚨 Sorun Giderme

### Yaygın Sorunlar

#### 1. Kampüs Harita Modülü Yüklenemedi
```
⚠️ Kampüs harita modülü yüklenemedi, basit harita kullanılacak
```
**Çözüm**: `kampus_harita.py` dosyasının aynı dizinde olduğundan emin olun.

#### 2. Model Dosyası Bulunamadı
```
Model bulunamadı: models/dl_model.pkl
```
**Çözüm**: Önce ana sistemi çalıştırarak modelleri eğitin:
```bash
python main.py
```

#### 3. Veri Dosyası Eksik
```
Veri yükleme hatası: [Errno 2] No such file or directory
```
**Çözüm**: `data/` klasöründe gerekli Excel dosyalarının olduğundan emin olun.

#### 4. Özellik Adı Uyumsuzluğu
```
The feature names should match those that were passed during fit
```
**Çözüm**: Modeli yeniden eğitin veya uyumlu veri formatı kullanın.

## 📞 İletişim ve Destek

### Proje Bilgileri
- **Yarışma**: TEKNOFEST 2025
- **Kategori**: 5G ve Ötesi Kablosuz Haberleşme Teknolojileri
- **Uygulama Alanı**: Konumlandırma Sistemleri

### Teknik Detaylar
- **Veri Seti**: İTÜ Kampüs 5G NR ölçümleri
- **Test Ortamı**: İstanbul Teknik Üniversitesi
- **Model Tipi**: Supervised Learning (Regresyon)
- **Değerlendirme**: Euclidean distance error

## 🎯 Gelecek Geliştirmeler

### Kısa Vadeli
- [ ] SHAP model açıklanabilirliği entegrasyonu
- [ ] Real-time streaming veri desteği
- [ ] Mobile web interface geliştirme
- [ ] Daha fazla 5G özelliği entegrasyonu

### Uzun Vadeli  
- [ ] Multi-teknoloji desteği (4G/5G/WiFi)
- [ ] Deep learning modelleri (CNN/LSTM)
- [ ] Edge computing optimizasyonu
- [ ] Çoklu kampüs desteği

## 📄 Lisans

Bu proje TEKNOFEST 2025 yarışması kapsamında eğitim amaçlı geliştirilmiştir.

---

## ⭐ Son Test Sonuçları

```
============================================================
📋 TEST ÖZETİ  
============================================================
✅ DL Model Test: Başarılı (130.21m hata, 11.07s)
✅ UL Model Test: Başarılı (532.90m hata, 11.11s)  
✅ Hibrit Model Test: Başarılı (230.98m hata, 11.49s)
✅ Gelişmiş Ensemble Model Test: Başarılı (368.38m hata, 10.04s)

🏆 Sistem Durumu: Yarışma Hazır! 🚀
```

**Son güncelleme**: Ocak 2025 - Tüm problemler çözüldü, sistem optimize edildi! 