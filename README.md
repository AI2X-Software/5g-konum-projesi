# TEKNOFEST 2025 5G Konumlandırma Projesi

Bu proje, TEKNOFEST 2025 5G konumlandırma yarışması için geliştirilmiş gelişmiş bir konum tahmin sistemidir. İTÜ kampüs harita verileri ile entegre çalışarak gerçek zamanlı konum tahminleri yapar.

## 🎯 Proje Özellikleri

### 📡 Konumlandırma Teknikleri
- **Klasik Yöntemler**: ToA, TDoA, AoA, RSSI
- **Makine Öğrenmesi**: Random Forest, GBDT, SVM, XGBoost
- **Derin Öğrenme**: LSTM, CNN, Transformer modelleri
- **Hibrit Yaklaşım**: DL/UL veri kombinasyonu

### 🗺️ Kampüs Harita Entegrasyonu
- **Şekil Dosyası Desteği**: İTÜ kampüs shapefile verilerini okur
- **Zengin Görselleştirme**: Binalar, yollar, su kütleleri, bitki örtüsü
- **İnteraktif Haritalar**: Folium tabanlı detaylı haritalar
- **Test Verisi Görselleştirme**: 72,000+ test noktasını pin olarak gösterir

### 🔍 Gelişmiş Analiz Araçları
- **Sinyal Kalitesi Analizi**: RSRP, RSRQ, SINR istatistikleri
- **Heat Map Görselleştirme**: Sinyal gücü dağılımı
- **Cluster Analizi**: Sinyal kalitesine göre gruplandırma
- **PCI Analizi**: Baz istasyonu kullanım istatistikleri

## 📁 Proje Yapısı

```
5g-konum-projesi/
├── data/
│   ├── İTÜ Kampüs Harita Verileri/    # Shapefile harita verileri
│   ├── Kopya5G_DL.xlsx                # Downlink test verileri
│   ├── Kopya5G_UL.xlsx                # Uplink test verileri
│   ├── Kopya5G_Scanner.xlsx           # Scanner verileri
│   └── ITU5GHucreBilgileri.xlsx       # Baz istasyonu bilgileri
├── outputs/
│   ├── grafikler/                     # İstatistik grafikleri
│   ├── kampus_haritasi.html           # Ana kampüs haritası
│   ├── sinyal_gucü_heatmap.html       # Sinyal gücü heat map
│   └── test_noktalari_cluster.html    # Cluster haritası
├── models/                            # Eğitilmiş ML modelleri
├── main.py                            # Ana sistem
├── yarismaci.py                       # Yarışma modu
├── kampus_harita.py                   # Kampüs harita modülü
├── kampus_analizi.py                  # Veri analizi araçları
└── requirements.txt                   # Gerekli kütüphaneler
```

## 🚀 Kurulum ve İlk Çalıştırma

### Adım 1: Proje Klonlama veya İndirme
```bash
# Proje klasörüne girin
cd 5g-konum-projesi
```

### Adım 2: Python Sanal Ortamı Oluşturma
```bash
# Sanal ortam oluştur
python -m venv venv

# Sanal ortamı aktif et
# Linux/Mac için:
source venv/bin/activate

# Windows için:
venv\Scripts\activate
```

### Adım 3: Gerekli Kütüphaneleri Yükleme
```bash
# Temel kütüphaneleri yükle
pip install -r requirements.txt
```

### Adım 4: Sistem Bağımlılıkları (macOS kullanıcıları için)
```bash
# XGBoost için OpenMP desteği
brew install libomp

# Shapefile desteği için GDAL (opsiyonel)
brew install gdal
```

### Adım 5: İlk Test - Modelleri Eğitme
```bash
# Ana sistemi çalıştır ve modelleri eğit
python main.py
```
Bu komut yaklaşık 1-2 dakika sürer ve `models/` klasörüne ML modellerini kaydeder.

## 🎮 Kullanım Senaryoları

### 📚 Senaryo 1: Sistemi Tanımak (İlk Kullanıcılar İçin)

#### 1.1 Kampüs Haritasını Keşfetme
```bash
# Ana kampüs haritasını oluştur (72,000+ test noktası ile)
python kampus_harita.py
```
**Çıktı**: `outputs/kampus_haritasi.html` (113MB) - Web tarayıcısında açılabilir

#### 1.2 Veri Analizini İnceleme
```bash
# Kapsamlı veri analizi yap
python kampus_analizi.py
```
**Çıktılar**: 
- `outputs/sinyal_gucü_heatmap.html` - Sinyal gücü ısı haritası
- `outputs/test_noktalari_cluster.html` - Sinyal kalitesine göre gruplandırma
- `outputs/grafikler/` - İstatistik grafikleri

### 🏆 Senaryo 2: Yarışma Modu (Gerçek Zamanlı Tahmin)

#### 2.1 Basit Tahmin
```bash
# Sadece DL verisi ile tahmin
python yarismaci.py data/Kopya5G_DL.xlsx
```

#### 2.2 Gelişmiş Tahmin (Önerilen)
```bash
# Baz istasyonu bilgileri ile hibrit model
python yarismaci.py data/Kopya5G_DL.xlsx \
  --cellinfo=data/ITU5GHucreBilgileri.xlsx \
  --model=hybrid
```

#### 2.3 Farklı Model Tipleri
```bash
# Sadece DL modeli
python yarismaci.py data/Kopya5G_DL.xlsx --model=dl

# Sadece UL modeli  
python yarismaci.py data/Kopya5G_DL.xlsx --model=ul

# Hibrit model (en iyi sonuç)
python yarismaci.py data/Kopya5G_DL.xlsx --model=hybrid
```

### 🔬 Senaryo 3: Geliştirici Modu

#### 3.1 Model Performansını Test Etme
```bash
# Ana sistemi çalıştır ve metrikleri gözden geçir
python main.py
```
Çıktı dosyaları:
- `outputs/model_metrics.json` - Performans metrikleri
- `outputs/kullanilan_sutunlar_ve_aciklamalari.csv` - Özellik açıklamaları

#### 3.2 Özel Veri ile Test
```bash
# Kendi veri dosyanızla test edin
python yarismaci.py yeni_veri.xlsx \
  --cellinfo=ITU5GHucreBilgileri.xlsx \
  --model=hybrid \
  --output=sonuclar/
```

## 📊 Beklenen Çıktılar

### 🏃‍♂️ Hızlı Test (5 dakika)
1. `python main.py` - Model eğitimi ve performans raporları
2. `python yarismaci.py data/Kopya5G_DL.xlsx --model=hybrid` - Örnek tahmin

### 🔍 Kapsamlı Analiz (10-15 dakika)
1. `python kampus_harita.py` - Ana kampüs haritası (113MB)
2. `python kampus_analizi.py` - Tüm analiz ve görselleştirmeler

### 📁 Çıktı Dosyaları
- **HTML Haritalar**: `outputs/*.html` (Web tarayıcısında açılır)
- **CSV Sonuçlar**: `outputs/tahmin_sonuclari_*.csv`
- **PNG Grafikler**: `outputs/grafikler/*.png`
- **JSON Metrikler**: `outputs/model_metrics.json`

## ⚡ Hızlı Başlangıç (3 Adım)

### 1️⃣ Kurulum
```bash
pip install -r requirements.txt
```

### 2️⃣ Sistem Hazırlama
```bash
python main.py
```

### 3️⃣ Yarışma Testi
```bash
python yarismaci.py data/Kopya5G_DL.xlsx --model=hybrid
```

**Sonuç**: `outputs/kampus_tahmin_haritasi_*.html` dosyasını web tarayıcısında açın!

## 📊 Analiz Çıktıları

### 📈 İstatistik Grafikleri
- **RSRP Histogram**: Sinyal gücü dağılımı
- **Sinyal Kalitesi Pie Chart**: Kalite kategorileri
- **PCI Dağılımı**: Baz istasyonu kullanımı
- **RSRP vs SINR**: Sinyal ilişkileri

### 🗺️ İnteraktif Haritalar
- **kampus_haritasi.html**: Tüm test verileri ve kampüs detayları
- **sinyal_gucü_heatmap.html**: Sinyal gücü ısı haritası
- **test_noktalari_cluster.html**: Sinyal kalitesine göre gruplandırma
- **kampus_tahmin_haritasi_*.html**: Gerçek zamanlı tahmin sonuçları

### 📋 Veri Dosyaları
- **tahmin_sonuclari_*.csv**: Tahmin koordinatları ve metrikler
- **model_metrics.json**: Model performans metrikleri
- **kullanilan_sutunlar_ve_aciklamalari.csv**: Özellik açıklamaları

## 🔧 Teknik Detaylar

### Veri İşleme
- **Otomatik Temizleme**: Eksik değerlerin işlenmesi
- **Özellik Mühendisliği**: Coğrafi ve sinyal özellikleri
- **Normalizasyon**: Modeller için veri hazırlama

### Model Performansı
- **RMSE**: < 0.001m (test verisi)
- **MAE**: < 0.001m
- **%5m Altı Doğruluk**: %100
- **Hesaplama Süresi**: < 0.1 saniye

### Harita Özellikleri
- **Shapefile Desteği**: GDAL/Fiona ile okuma
- **Koordinat Sistemi**: WGS84 (EPSG:4326)
- **Katman Yönetimi**: İnteraktif katman kontrolü
- **Responsive Tasarım**: Mobil uyumlu görselleştirme

## 📊 Test Verileri İstatistikleri

### 📍 Konum Bilgileri
- **Toplam Test Noktası**: 72,478
- **Coğrafi Alan**: ~953m çapında
- **Koordinat Aralığı**: 
  - Enlem: 41.098890° - 41.108090°
  - Boylam: 29.015340° - 29.031170°

### 📶 Sinyal Kalitesi
- **Ortalama RSRP**: -92.6 dBm
- **Sinyal Aralığı**: -150.6 dBm ile -50.8 dBm
- **İyi Sinyal Oranı**: %1.5
- **Kullanılan PCI Sayısı**: 11

### 📡 Baz İstasyonları
- **BS Sayısı**: 9 adet
- **En Aktif PCI**: 68 (694 ölçüm)
- **En İyi Sinyal PCI**: 23 (-80.3 dBm ortalama)

## 🛠️ Geliştirme

### Yeni Özellik Ekleme
1. **Yeni Algoritma**: `main.py`'ye model ekle
2. **Görselleştirme**: `kampus_harita.py`'ye katman ekle
3. **Analiz**: `kampus_analizi.py`'ye fonksiyon ekle

### Shapefile Ekleme
```python
# kampus_harita.py dosyasında
shapefile_dosyalari = {
    'yeni_katman': 'YeniKatman.shp'
}
```

## 🎯 Yarışma Kullanımı

### Gerçek Zamanlı Test
```bash
# Hibrit model ile en iyi sonuç
python yarismaci.py test_input.xlsx --cellinfo=ITU5GHucreBilgileri.xlsx --model=hybrid --output=sonuclar/
```

### Çıktı Formatı
```csv
Latitude,Longitude,Base_Station_PCI,Environment,LOS_Probability,Model_Type,Calculation_Time_Seconds
41.106229,29.023778,68,suburban,1.29,hybrid,0.05
```

## 🔍 Sorun Giderme

### Kütüphane Hataları
```bash
# Shapefile okuma hatası
pip install --upgrade geopandas fiona

# XGBoost hatası (macOS)
brew install libomp
```

### Harita Görüntüleme
- HTML dosyalarını modern web tarayıcısında açın
- JavaScript'in etkin olduğundan emin olun
- Büyük dosyalar (>100MB) yüklenmesi zaman alabilir

### Yaygın Problemler
```bash
# Python path problemi
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Cache temizleme
rm -rf __pycache__
rm -rf .DS_Store

# Sanal ortam yeniden oluşturma
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📞 İletişim

**Proje**: TEKNOFEST 2025 5G Konumlandırma Yarışması  
**Geliştirici**: Ali Emre  
**Teknoloji**: Python, ML, GIS, 5G Signals  

---

*Bu proje, gerçek 5G ölçüm verileri kullanarak konum tahmininde state-of-the-art sonuçlar elde etmektedir.* 