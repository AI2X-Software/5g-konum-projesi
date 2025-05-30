import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
from pathlib import Path
import joblib
import folium
from folium.plugins import MarkerCluster
import os
import time

# Kampus harita modülünü içe aktar
try:
    from kampus_harita import KampusHaritasi, kampus_haritasi_olustur
    KAMPUS_HARITA_MEVCUT = True
    print("✅ Kampüs harita modülü yüklendi")
except ImportError:
    KAMPUS_HARITA_MEVCUT = False
    print("⚠️ Kampüs harita modülü yüklenemedi, basit harita kullanılacak")

# Create outputs directory
Path("outputs").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# ======================================
# 1. YARDIMCI FONKSİYONLAR
# ======================================
def save_feature_importance(model, feature_names, filename='outputs/feature_importance.csv'):
    """Model feature importance değerlerini CSV olarak kaydet"""
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    importances.to_csv(filename, index=False)
    return importances

def save_metrics(metrics_dict, filename='outputs/model_metrics.json'):
    """Model metriklerini JSON olarak kaydet"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=4)

def save_feature_descriptions(feature_names, filename='outputs/kullanilan_sutunlar_ve_aciklamalari.csv'):
    """Kullanılan özelliklerin açıklamalarını CSV olarak kaydet"""
    descriptions = {
        'NR_UE_PCI': 'Physical Cell ID - Hücre kimliği',
        'NR_UE_RSRP': 'Reference Signal Received Power (dBm)',
        'NR_UE_RSRQ': 'Reference Signal Received Quality (dB)',
        'NR_UE_SINR': 'Signal to Interference plus Noise Ratio (dB)',
        'NR_UE_Timing_Advance': 'Timing Advance değeri (μs)',
        'NR_UE_Pathloss': 'Yol kaybı (dB)',
        'bs_distance': 'Baz istasyonu ile UE arası mesafe (m)',
        'bs_azimuth_diff': 'Anten yönü ile UE arası açı farkı (derece)',
        'rsrp_diff': 'Komşu hücreler arası sinyal farkı (dB)'
    }
    df = pd.DataFrame([
        {'feature': f, 'description': descriptions.get(f.split('_')[0], 'Yardımcı özellik')}
        for f in feature_names
    ])
    df.to_csv(filename, index=False, encoding='utf-8')

# ======================================
# 1.1 KLASİK KONUMLANDIRMA YÖNTEMLERİ
# ======================================
class ClassicPositioning:
    """Klasik konumlandırma teknikleri"""
    
    @staticmethod
    def calculate_toa(timing_advance, c=299792458):
        """Timing Advance'den ToA hesaplama"""
        return timing_advance * (1e-6) * c  # metre cinsinden mesafe
    
    @staticmethod
    def calculate_tdoa(ta_list, bs_positions):
        """TDoA hesaplama"""
        if len(ta_list) < 2:
            return None
        tdoa = np.diff(ta_list)
        return tdoa
    
    @staticmethod
    def calculate_aoa(bs_azimuth, bs_pos, ue_pos):
        """AoA hesaplama"""
        dx = ue_pos[1] - bs_pos[1]
        dy = ue_pos[0] - bs_pos[0]
        measured_angle = np.degrees(np.arctan2(dx, dy)) % 360
        return abs(measured_angle - bs_azimuth)

# ======================================
# 2. ORTAK TEMİZLEME VE FEATURE FONKSİYONU
# ======================================
def extract_and_clean(df, cellinfo, feature_main, feature_nbr_pci, feature_nbr_rsrp, 
                     feature_nbr_rsrq, y_cols=['Longitude','Latitude'], source_name="DL"):
    # Otomatik olarak sadece bulunan sütunları kullan
    all_features = feature_main + feature_nbr_pci + feature_nbr_rsrp + feature_nbr_rsrq
    available = [col for col in all_features if col in df.columns]
    
    print(f"Mevcut öznitelikler ({source_name}): {len(available)} / {len(all_features)}")
    
    # Ana öznitelikleri seç
    X = df[available].fillna(-110)

    # Konum sütunlarını kontrol et
    if not all(col in df.columns for col in y_cols):
        print(f"Uyarı: Konum sütunları eksik! Mevcut: {[col for col in y_cols if col in df.columns]}")
        return np.array([]), np.array([])

    # Coğrafi özellikler ekle
    cellinfo['PCI'] = cellinfo['PCI '] if 'PCI ' in cellinfo.columns else cellinfo['PCI']

    def get_bs_coords(pci):
        match = cellinfo[cellinfo['PCI'] == pci]
        if not match.empty:
            return match.iloc[0]['Latitude'], match.iloc[0]['Longitude'], match.iloc[0]['Azimuth [°]']
        else:
            return np.nan, np.nan, np.nan

    bs_lat, bs_lon, bs_azimuth, bs_distance, bs_azimuth_diff = [], [], [], [], []
    
    for idx, row in df.iterrows():
        if 'NR_UE_PCI_0' in row:
            pci = row['NR_UE_PCI_0']
            lat1, lon1 = row['Latitude'], row['Longitude']
            lat2, lon2, azim = get_bs_coords(pci)
            bs_lat.append(lat2)
            bs_lon.append(lon2)
            bs_azimuth.append(azim)
            try:
                if not (pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2)):
                    dist = geodesic((lat1, lon1), (lat2, lon2)).meters
                else:
                    dist = np.nan
            except Exception:
                dist = np.nan
            bs_distance.append(dist)
            try:
                if not (pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2) or pd.isna(azim)):
                    angle_to_bs = np.degrees(np.arctan2(lon2 - lon1, lat2 - lat1)) % 360
                    diff = abs(angle_to_bs - azim)
                else:
                    diff = np.nan
            except Exception:
                diff = np.nan
            bs_azimuth_diff.append(diff)
        else:
            bs_lat.append(np.nan)
            bs_lon.append(np.nan)
            bs_azimuth.append(np.nan)
            bs_distance.append(np.nan)
            bs_azimuth_diff.append(np.nan)

    X['bs_distance'] = bs_distance
    X['bs_azimuth_diff'] = bs_azimuth_diff

    # En güçlü iki hücre arasındaki sinyal farkı (varsa)
    if 'NR_UE_RSRP_1' in X.columns:
        X['rsrp_diff_0_1'] = X['NR_UE_RSRP_0'] - X['NR_UE_RSRP_1']
    else:
        X['rsrp_diff_0_1'] = 0

    # Hedef değişkenleri ekle
    y = df[y_cols].copy()

    # Tüm feature ve hedef kolonlarını float'a döndür (string varsa otomatik NaN olur)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    for col in y.columns:
        y[col] = pd.to_numeric(y[col], errors='coerce')

    # Sadece tam dolu (eksiksiz, sayısal) satırları tut
    combined = pd.concat([X, y], axis=1)
    combined_clean = combined.dropna()
    
    if len(combined_clean) == 0:
        print(f"Uyarı: Temizleme sonrası hiç veri kalmadı ({source_name})")
        return np.array([]), np.array([])

    # Tekrar X ve y arraylerine ayır
    feature_cols = X.columns
    X_clean = combined_clean[feature_cols].values
    y_clean = combined_clean[y_cols].values
    
    print(f"Temizlenen veri ({source_name}): {len(combined_clean)} satır, {len(feature_cols)} öznitelik")

    return X_clean, y_clean

# ======================================
# 3. ÖZNİTELİK TANIMLAMALARI
# ======================================
main_features = [
    'NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0',
    'NR_UE_Timing_Advance', 'NR_UE_Pathloss_DL_0',
    'NR_UE_Throughput_PDCP_DL', 'NR_UE_MCS_DL_0'
]
nbr_pci = [f'NR_UE_Nbr_PCI_{i}' for i in range(5)]
nbr_rsrp = [f'NR_UE_Nbr_RSRP_{i}' for i in range(5)]
nbr_rsrq = [f'NR_UE_Nbr_RSRQ_{i}' for i in range(5)]

# ======================================
# 3.1 GELİŞMİŞ SENARYO ANALİZİ
# ======================================
def analyze_scenario(row, cellinfo):
    """Detaylı senaryo analizi"""
    scenario = {
        'environment': 'unknown',
        'los_probability': 0.0,
        'difficulty_level': 'medium'
    }
    
    # Çevre tipi tespiti
    if 'NR_UE_RSRP_0' in row:
        rsrp = float(row['NR_UE_RSRP_0'])
        if rsrp > -85:
            scenario['environment'] = 'open_area'
        elif rsrp > -95:
            scenario['environment'] = 'suburban'
        else:
            scenario['environment'] = 'urban'
    
    # LOS olasılığı hesaplama
    if 'bs_distance' in row and 'NR_UE_RSRP_0' in row:
        distance = float(row['bs_distance'])
        rsrp = float(row['NR_UE_RSRP_0'])
        theoretical_pl = 20 * np.log10(distance) + 32.4
        measured_pl = -rsrp
        pl_diff = measured_pl - theoretical_pl
        scenario['los_probability'] = max(0, 1 - pl_diff/30)
    
    # Zorluk seviyesi belirleme
    if scenario['environment'] == 'open_area' and scenario['los_probability'] > 0.8:
        scenario['difficulty_level'] = 'easy'
    elif scenario['environment'] == 'urban' and scenario['los_probability'] < 0.3:
        scenario['difficulty_level'] = 'hard'
    
    return scenario

# ======================================
# 4. VERİ KAYNAKLARINI İŞLE
# ======================================
def find_series_sheet(d):
    # Birden fazla "Series" sheet varsa, ilkini seç
    for name in d:
        if 'Series' in name:
            return d[name]
    return None

# Gerçek veri dosyalarını yükle
try:
    dl = pd.read_excel('data/Kopya5G_DL.xlsx', sheet_name=None)
    ul = pd.read_excel('data/Kopya5G_UL.xlsx', sheet_name=None) 
    scanner = pd.read_excel('data/Kopya5G_Scanner.xlsx', sheet_name=None)
    cellinfo = pd.read_excel('data/ITU5GHucreBilgileri.xlsx', sheet_name='Hücre tablosu')
    
    print("Veri dosyaları başarıyla yüklendi!")
    print(f"DL sheets: {list(dl.keys())}")
    print(f"UL sheets: {list(ul.keys())}")
    print(f"Scanner sheets: {list(scanner.keys())}")
    print(f"Cell info satır sayısı: {len(cellinfo)}")
    
except Exception as e:
    print(f"Veri yükleme hatası: {e}")
    print("Örnek verilerle çalışma moduna geçiliyor...")
    # Basit örnek veriler oluştur
    dl = {'Series Formatted Data': pd.DataFrame()}
    ul = {'Series Formatted Data': pd.DataFrame()}
    scanner = {'Series Formatted Data': pd.DataFrame()}
    cellinfo = pd.DataFrame()

dl_series = find_series_sheet(dl)
ul_series = find_series_sheet(ul)
scanner_series = find_series_sheet(scanner)

if dl_series is not None and not dl_series.empty:
    print(f"DL Series veri boyutu: {dl_series.shape}")
    print(f"DL Series sütunları: {list(dl_series.columns[:10])}")
    
    # Konum bilgilerini kontrol et
    if 'Latitude' in dl_series.columns and 'Longitude' in dl_series.columns:
        X_dl, y_dl = extract_and_clean(dl_series, cellinfo, main_features, nbr_pci, nbr_rsrp, nbr_rsrq, 
                                      y_cols=['Longitude', 'Latitude'], source_name="DL")
        print(f"DL verisi temizlendi: {X_dl.shape[0]} satır")
    else:
        print("DL verisinde konum bilgisi bulunamadı!")
        X_dl, y_dl = np.array([]), np.array([])
else:
    print("DL series verisi yüklenemedi!")
    X_dl, y_dl = np.array([]), np.array([])

if ul_series is not None and not ul_series.empty:
    print(f"UL Series veri boyutu: {ul_series.shape}")
    
    # Konum bilgilerini kontrol et
    if 'Latitude' in ul_series.columns and 'Longitude' in ul_series.columns:
        X_ul, y_ul = extract_and_clean(ul_series, cellinfo, main_features, nbr_pci, nbr_rsrp, nbr_rsrq, 
                                      y_cols=['Longitude', 'Latitude'], source_name="UL")
        print(f"UL verisi temizlendi: {X_ul.shape[0]} satır")
    else:
        print("UL verisinde konum bilgisi bulunamadı!")
        X_ul, y_ul = np.array([]), np.array([])
else:
    print("UL series verisi yüklenemedi!")
    X_ul, y_ul = np.array([])

# Scanner verisi için özel öznitelikler tanımlanabilir
# scanner_features = [...] 
# X_scanner, y_scanner = extract_and_clean(...)

# ======================================
# 5. GELİŞMİŞ ÖZNİTELİK MÜHENDİSLİĞİ
# ======================================
def create_advanced_features(X, source_name=""):
    """Gelişmiş öznitelik mühendisliği"""
    # Sinyal gücü istatistikleri
    rsrp_cols = [col for col in X.columns if 'RSRP' in col]
    if rsrp_cols:
        X['mean_rsrp'] = X[rsrp_cols].mean(axis=1)
        X['std_rsrp'] = X[rsrp_cols].std(axis=1)
        X['max_rsrp'] = X[rsrp_cols].max(axis=1)
    
    # SINR ve RSRQ için benzer istatistikler
    for signal_type in ['SINR', 'RSRQ']:
        cols = [col for col in X.columns if signal_type in col]
        if cols:
            X[f'mean_{signal_type.lower()}'] = X[cols].mean(axis=1)
            X[f'std_{signal_type.lower()}'] = X[cols].std(axis=1)
    
    return X

# ======================================
# 5.1 DERİN ÖĞRENME MODELLERİ
# ======================================
class LocationCNN(nn.Module):
    """Konum tahmini için CNN modeli"""
    def __init__(self, input_dim):
        super(LocationCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.fc = nn.Linear(128, 2)  # lat, lon çıktısı
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.mean(x, dim=2)  # Global average pooling
        return self.fc(x)

# ======================================
# 6. MODEL KURMA VE DEĞERLENDİRME
# ======================================
def train_and_evaluate(X, y, source_name=""):
    """Geliştirilmiş model eğitimi ve değerlendirmesi"""
    if X.shape[0] > 0:
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Senaryo analizi
        scenario = analyze_scenario(pd.DataFrame(X_test).iloc[0], cellinfo)
        
        # Model seçimi
        if scenario['environment'] == 'open_area' and scenario['los_probability'] > 0.8:
            # Klasik yöntemler için hesaplamalar
            classic = ClassicPositioning()
            distances = [classic.calculate_toa(ta) for ta in X_test[:, 4]]  # TA kolonunu kullan
            # ... klasik hesaplamalar ...
        
        # ML modeli
        models = {
            'rf': RandomForestRegressor(n_estimators=150, max_depth=18, random_state=42),
            'gbdt': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgb': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        }
        
        # Senaryo bazlı model seçimi
        if scenario['difficulty_level'] == 'hard':
            model = models['xgb']
        else:
            model = models['rf']
        
        # Model eğitimi ve tahmin
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrikler
        errors = np.linalg.norm(y_pred - y_test, axis=1)
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'median_error': float(np.median(errors)),
            'percent_under_5m': float(np.sum(errors < 5) / len(errors) * 100),
            'scenario': scenario
        }
        
        print(f"\n{source_name} Sonuçları ({scenario['environment']}):")
        print(f"RMSE: {metrics['rmse']:.2f} m")
        print(f"MAE: {metrics['mae']:.2f} m")
        print(f"Medyan Hata: {metrics['median_error']:.2f} m")
        print(f"%5m Altı: {metrics['percent_under_5m']:.1f}%")
        
        return model, errors, metrics
    
    return None, None, None

# DL Modeli
if X_dl.size > 0:
    dl_model, dl_errors, dl_metrics = train_and_evaluate(X_dl, y_dl, "DL")
else:
    print("DL verisi bulunamadı, model eğitilemiyor.")
    dl_model, dl_errors, dl_metrics = None, None, None

# UL Modeli
if X_ul.size > 0:
    ul_model, ul_errors, ul_metrics = train_and_evaluate(X_ul, y_ul, "UL")
else:
    print("UL verisi bulunamadı, model eğitilemiyor.")
    ul_model, ul_errors, ul_metrics = None, None, None

# ======================================
# 7. MODEL DEĞERLENDİRME VE GÖRSEL ÇIKTILAR
# ======================================
def evaluate_and_visualize(model, X_test, y_test, feature_names, source_name=""):
    """Model değerlendirme ve görselleştirme"""
    y_pred = model.predict(X_test)
    errors = np.linalg.norm(y_pred - y_test, axis=1)
    
    # Metrikleri hesapla
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'median_error': float(np.median(errors)),
        'percent_under_5m': float(np.sum(errors < 5) / len(errors) * 100)
    }
    
    # Feature importance
    save_feature_importance(model, feature_names)
    
    # CDF Plot
    plt.figure(figsize=(10,6))
    plt.hist(errors, bins=100, density=True, cumulative=True, 
             histtype='step', label=f'{source_name} CDF')
    plt.axvline(5, color='r', linestyle='--', label='5 m')
    plt.xlabel("Hata (m)")
    plt.ylabel("Kümülatif Oran")
    plt.title(f"{source_name} - Konum Hatası CDF")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'outputs/{source_name.lower()}_cdf_plot.png')
    plt.close()
    
    return metrics

# ======================================
# 8. ANA FONKSİYON
# ======================================
def main():
    """Ana pipeline fonksiyonu"""
    print("TEKNOFEST 2025 5G Konumlandırma Sistemi")
    print("=" * 50)
    
    # Modelleri kaydet
    if dl_model is not None:
        joblib.dump(dl_model, 'models/dl_model.pkl')
        print("DL modeli kaydedildi.")
    if ul_model is not None:
        joblib.dump(ul_model, 'models/ul_model.pkl')
        print("UL modeli kaydedildi.")
    
    # Sonuçları kaydet
    metrics_to_save = {}
    if dl_metrics is not None:
        metrics_to_save['DL'] = dl_metrics
    if ul_metrics is not None:
        metrics_to_save['UL'] = ul_metrics
        
    if metrics_to_save:
        save_metrics(metrics_to_save)
        print("Metrikler kaydedildi.")
    
    # Özellik açıklamalarını kaydet
    if X_dl.size > 0:
        feature_names = main_features + nbr_pci + nbr_rsrp + nbr_rsrq + ['bs_distance', 'bs_azimuth_diff', 'rsrp_diff_0_1']
        save_feature_descriptions(feature_names)
        print("Özellik açıklamaları kaydedildi.")
    
    print("\nSistem hazır! Yarışma modunu test etmek için:")
    print("python yarismaci.py data/Kopya5G_DL.xlsx --cellinfo=data/ITU5GHucreBilgileri.xlsx")

# ======================================
# 9. YARIŞMA MODU - GERÇEK ZAMANLI TAHMİN
# ======================================
def predict_realtime(input_file, cell_info_file='ITU5GHucreBilgileri.xlsx', model_type='dl'):
    """
    Yarışma sırasında gerçek zamanlı veri ile konum tahmini yapar ve haritada gösterir
    
    Args:
        input_file: Gelen veri dosyası (excel)
        cell_info_file: Baz istasyonu bilgileri dosyası
        model_type: Kullanılacak model tipi ('dl', 'ul', 'hybrid')
    
    Returns:
        predicted_coords: Tahmin edilen koordinatlar (lat, lon)
    """
    print(f"Gerçek zamanlı tahmin yapılıyor... Dosya: {input_file}")
    start_time = time.time()
    
    # Veri dosyalarını yükle
    try:
        input_data = pd.read_excel(input_file)
        cellinfo = pd.read_excel(cell_info_file)
        print(f"Veriler başarıyla yüklendi. Satır sayısı: {len(input_data)}")
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None
    
    # Yarışma verisi genellikle tek bir satır olacaktır
    # Series sayfasını bul veya direkt al
    if 'Series' in input_data:
        series_data = input_data['Series']
    else:
        series_data = input_data
    
    # Öznitelik tanımlamaları (main.py'deki ile aynı)
    main_features = [
        'NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0',
        'NR_UE_Timing_Advance', 'NR_UE_Pathloss_DL_0',
        'NR_UE_Throughput_PDCP_DL', 'NR_UE_MCS_DL_0'
    ]
    nbr_pci = [f'NR_UE_Nbr_PCI_{i}' for i in range(5)]
    nbr_rsrp = [f'NR_UE_Nbr_RSRP_{i}' for i in range(5)]
    nbr_rsrq = [f'NR_UE_Nbr_RSRQ_{i}' for i in range(5)]
    
    # Veri temizleme ve öznitelik çıkarma (GPS değerleri olmadan)
    all_features = main_features + nbr_pci + nbr_rsrp + nbr_rsrq
    available = [col for col in all_features if col in series_data.columns]
    X = series_data[available].fillna(-110)
    
    # Hücre bilgileri ve coğrafi özellikler ekleme
    cellinfo['PCI'] = cellinfo['PCI '] if 'PCI ' in cellinfo.columns else cellinfo['PCI']
    
    # Baz istasyonu bilgilerini al ve öznitelikleri ekle
    def get_bs_coords(pci):
        match = cellinfo[cellinfo['PCI'] == pci]
        if not match.empty:
            return match.iloc[0]['Latitude'], match.iloc[0]['Longitude'], match.iloc[0]['Azimuth [°]']
        else:
            return np.nan, np.nan, np.nan
    
    # Her satır için özellik çıkarımı (tek satır olacak muhtemelen)
    bs_distances = []
    bs_azimuth_diffs = []
    
    for idx, row in X.iterrows():
        # Servis hücresi PCI
        pci = row['NR_UE_PCI_0']
        # Baz istasyonu bilgilerini al
        bs_lat, bs_lon, bs_azimuth = get_bs_coords(pci)
        
        # Klasik ToA yöntemi için TA kullanarak mesafe hesapla
        if 'NR_UE_Timing_Advance' in row and not pd.isna(row['NR_UE_Timing_Advance']):
            ta = row['NR_UE_Timing_Advance']
            toa_distance = ClassicPositioning.calculate_toa(ta)
            bs_distances.append(toa_distance)
        else:
            bs_distances.append(np.nan)
        
        # Diğer hücreler için de benzer işlem yapılabilir
        bs_azimuth_diffs.append(0)  # Başlangıç için 0 koy
    
    X['bs_distance'] = bs_distances
    X['bs_azimuth_diff'] = bs_azimuth_diffs
    
    # RSRP farkı
    if 'NR_UE_RSRP_1' in X.columns:
        X['rsrp_diff_0_1'] = X['NR_UE_RSRP_0'] - X['NR_UE_RSRP_1']
    else:
        X['rsrp_diff_0_1'] = 0
    
    # İleri öznitelikler ekle
    X = create_advanced_features(X, model_type)
    
    # Veriyi modelin beklediği formata dönüştür
    feature_cols = available + ['bs_distance', 'bs_azimuth_diff', 'rsrp_diff_0_1']
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0  # Eksik sütunlar için varsayılan değer
    
    # Konum tahmini
    X_np = X[feature_cols].values
    
    # Senaryo analizi
    scenario = analyze_scenario(X.iloc[0], cellinfo)
    print(f"Senaryo: {scenario['environment']}, LOS olasılığı: {scenario['los_probability']:.2f}")
    
    # Hibrit model için
    if model_type == 'hybrid':
        return predict_hybrid(X_np, feature_cols, scenario, cellinfo, X.iloc[0])
    
    # Tek model için
    # Modeli yükle
    model_path = f"models/{model_type}_model.pkl"
    if not os.path.exists(model_path):
        print(f"Model bulunamadı: {model_path}")
        return None
    
    model = joblib.load(model_path)
    
    # Konum tahmini
    predicted_coords = model.predict(X_np)[0]  # [latitude, longitude]
    
    # PCI'a göre baz istasyonu konumu
    pci = X.iloc[0]['NR_UE_PCI_0']
    bs_lat, bs_lon, _ = get_bs_coords(pci)
    
    # Hesaplama süresini ölç
    calc_time = time.time() - start_time
    print(f"Tahmin edilen konum: Lat: {predicted_coords[0]:.6f}, Lon: {predicted_coords[1]:.6f}")
    print(f"Hesaplama süresi: {calc_time:.2f} saniye")
    
    # Haritada göster
    visualize_prediction(predicted_coords, pci, bs_lat, bs_lon, scenario, calc_time)
    
    return predicted_coords

def predict_hybrid(X_np, feature_cols, scenario, cellinfo, row_data):
    """
    Hibrit model tahmini yapar (DL, UL ve klasik yöntemler)
    
    Args:
        X_np: Model girdisi olarak hazırlanmış öznitelikler
        feature_cols: Öznitelik isimleri
        scenario: Senaryo analizi sonuçları
        cellinfo: Baz istasyonu bilgileri
        row_data: Girdi verisinin satırı
        
    Returns:
        predicted_coords: Tahmin edilen koordinatlar (lat, lon)
    """
    start_time = time.time()
    dl_model_path = "models/dl_model.pkl"
    ul_model_path = "models/ul_model.pkl"
    
    # İki model de mevcut mu kontrol et
    if not os.path.exists(dl_model_path) or not os.path.exists(ul_model_path):
        print("Hibrit model için DL ve UL modelleri gereklidir.")
        # Mevcut olanı kullan
        if os.path.exists(dl_model_path):
            model = joblib.load(dl_model_path)
            model_type = "dl"
        elif os.path.exists(ul_model_path):
            model = joblib.load(ul_model_path)
            model_type = "ul"
        else:
            print("Hiçbir model bulunamadı!")
            return None
        
        print(f"Hibrit model yerine {model_type} model kullanılıyor.")
        predicted_coords = model.predict(X_np)[0]
    else:
        # Her iki modeli de yükle
        dl_model = joblib.load(dl_model_path)
        ul_model = joblib.load(ul_model_path)
        
        # Her iki modelle de tahmin yap
        dl_pred = dl_model.predict(X_np)[0]
        ul_pred = ul_model.predict(X_np)[0]
        
        # Klasik hesaplamaya göre baz istasyonundan mesafe
        pci = row_data['NR_UE_PCI_0']
        bs_lat, bs_lon, bs_azimuth = get_bs_coords(pci, cellinfo)
        
        # Güven skorları (hesaplanan senaryoya göre)
        dl_weight = 0.5
        ul_weight = 0.5
        
        # Sinyal gücü ve çevresel faktörlere göre ağırlıklandırma
        if scenario['environment'] == 'open_area':
            # Açık alanda klasik yöntem ve DL daha iyidir
            dl_weight = 0.7
            ul_weight = 0.3
        elif scenario['environment'] == 'urban':
            # Şehir içinde UL daha fazla değerlendir
            if scenario['los_probability'] < 0.4:
                dl_weight = 0.4
                ul_weight = 0.6
        
        # Tahmin kalitesi için sinyal değerleri kullan
        rsrp = float(row_data.get('NR_UE_RSRP_0', -100))
        if rsrp > -90:  # Güçlü sinyal
            dl_weight += 0.1
            ul_weight -= 0.1
        
        # Ağırlıklı ortalama al
        predicted_coords = [
            dl_weight * dl_pred[0] + ul_weight * ul_pred[0],
            dl_weight * dl_pred[1] + ul_weight * ul_pred[1]
        ]
        
        print(f"DL tahmin: {dl_pred}")
        print(f"UL tahmin: {ul_pred}")
        print(f"Ağırlıklar: DL={dl_weight:.2f}, UL={ul_weight:.2f}")
    
    # PCI'a göre baz istasyonu konumu
    bs_lat, bs_lon, _ = get_bs_coords(pci, cellinfo)
    
    # Hesaplama süresini ölç
    calc_time = time.time() - start_time
    print(f"Hibrit tahmin edilen konum: Lat: {predicted_coords[0]:.6f}, Lon: {predicted_coords[1]:.6f}")
    print(f"Hesaplama süresi: {calc_time:.2f} saniye")
    
    # Haritada göster
    visualize_prediction(predicted_coords, pci, bs_lat, bs_lon, scenario, calc_time, model_type="hybrid")
    
    return predicted_coords

def get_bs_coords(pci, cellinfo):
    """Baz istasyonu koordinatlarını döndürür"""
    match = cellinfo[cellinfo['PCI'] == pci]
    if not match.empty:
        return match.iloc[0]['Latitude'], match.iloc[0]['Longitude'], match.iloc[0]['Azimuth [°]']
    else:
        return np.nan, np.nan, np.nan

def visualize_prediction(predicted_coords, pci, bs_lat, bs_lon, scenario, calc_time, model_type="standard"):
    """Tahmin sonuçlarını haritada gösterir ve kaydeder"""
    timestamp = int(time.time())
    kampus_harita_aktif = KAMPUS_HARITA_MEVCUT
    
    # Kampüs haritası mevcut mu kontrol et
    if kampus_harita_aktif:
        try:
            print("🗺️ Zengin kampüs haritası oluşturuluyor...")
            
            # Kampüs harita nesnesi oluştur
            kampus = KampusHaritasi()
            
            # Temel haritayı oluştur
            harita = kampus.harita_olustur()
            
            # Baz istasyonlarını ekle
            kampus.baz_istasyonlari_ekle(harita, "data/ITU5GHucreBilgileri.xlsx")
            
            # Tahmin edilen konumu göster (özel pin)
            folium.Marker(
                location=[predicted_coords[0], predicted_coords[1]],
                popup=folium.Popup(f"""
                <b>🎯 TAHMİN EDİLEN KONUM</b><br>
                <b>Model:</b> {model_type}<br>
                <b>Koordinat:</b> {predicted_coords[0]:.6f}, {predicted_coords[1]:.6f}<br>
                <b>Senaryo:</b> {scenario['environment']}<br>
                <b>LOS Olasılığı:</b> {scenario['los_probability']:.2f}<br>
                <b>Hesaplama Süresi:</b> {calc_time:.2f}s<br>
                <b>PCI:</b> {pci}
                """, max_width=400),
                tooltip="Tahmin Edilen Konum",
                icon=folium.Icon(color='red', icon='crosshairs', prefix='fa')
            ).add_to(harita)
            
            # Baz istasyonu ile tahmin edilen konum arasına çizgi çiz
            if not np.isnan(bs_lat) and not np.isnan(bs_lon):
                folium.PolyLine(
                    locations=[[bs_lat, bs_lon], [predicted_coords[0], predicted_coords[1]]],
                    color='blue',
                    weight=3,
                    opacity=0.7,
                    popup=f"BS-UE Bağlantısı (PCI: {pci})"
                ).add_to(harita)
                
                # Mesafeyi hesapla ve göster
                mesafe = geodesic((bs_lat, bs_lon), (predicted_coords[0], predicted_coords[1])).meters
                
                # Orta noktada mesafe bilgisi
                orta_lat = (bs_lat + predicted_coords[0]) / 2
                orta_lon = (bs_lon + predicted_coords[1]) / 2
                
                folium.Marker(
                    location=[orta_lat, orta_lon],
                    popup=f"Mesafe: {mesafe:.1f} m",
                    icon=folium.DivIcon(html=f"""
                    <div style="font-size: 12px; color: blue; font-weight: bold;">
                    📏 {mesafe:.1f}m
                    </div>""")
                ).add_to(harita)
            
            # Haritayı kaydet
            map_file = f"outputs/kampus_tahmin_haritasi_{timestamp}.html"
            harita.save(map_file)
            print(f"✅ Zengin kampüs haritası kaydedildi: {map_file}")
            
        except Exception as e:
            print(f"❌ Kampüs haritası oluşturulamadı: {e}")
            print("🔄 Basit harita ile devam ediliyor...")
            kampus_harita_aktif = False
    
    # Basit harita (fallback)
    if not kampus_harita_aktif:
        print("🗺️ Basit harita oluşturuluyor...")
        
        # Haritada göster
        map_center = [predicted_coords[0], predicted_coords[1]]
        m = folium.Map(location=map_center, zoom_start=16)
        
        # Tahmin edilen konumu göster
        folium.Marker(
            location=[predicted_coords[0], predicted_coords[1]],
            popup="Tahmin Edilen Konum",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        # Baz istasyonu konumunu göster
        if not np.isnan(bs_lat) and not np.isnan(bs_lon):
            folium.Marker(
                location=[bs_lat, bs_lon],
                popup=f"Baz İstasyonu (PCI: {pci})",
                icon=folium.Icon(color='green', icon='antenna')
            ).add_to(m)
            
            # Baz istasyonu ile tahmin edilen konum arasına çizgi çiz
            folium.PolyLine(
                locations=[[bs_lat, bs_lon], [predicted_coords[0], predicted_coords[1]]],
                color='blue',
                weight=2,
                opacity=0.7
            ).add_to(m)
        
        # Haritayı kaydet
        map_file = f"outputs/tahmin_haritasi_{timestamp}.html"
        m.save(map_file)
        print(f"Harita kaydedildi: {map_file}")
    
    # Sonuçları CSV olarak da kaydet
    results = pd.DataFrame({
        'Latitude': [predicted_coords[0]],
        'Longitude': [predicted_coords[1]],
        'Base_Station_PCI': [pci],
        'BS_Latitude': [bs_lat],
        'BS_Longitude': [bs_lon],
        'Environment': [scenario['environment']],
        'LOS_Probability': [scenario['los_probability']],
        'Model_Type': [model_type],
        'Calculation_Time_Seconds': [calc_time]
    })
    
    results.to_csv(f"outputs/tahmin_sonuclari_{timestamp}.csv", index=False)

if __name__ == "__main__":
    main()
    
    # Gerçek zamanlı test için
    # predict_realtime('test_input.xlsx', 'ITU5GHucreBilgileri.xlsx', 'dl')