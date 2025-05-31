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
import random
from sklearn.ensemble import VotingRegressor, BaggingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')

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
                     feature_nbr_rsrq, y_cols=['Latitude', 'Longitude'], source_name="DL"):
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
# 3.2 SİNYAL KALİTESİ ANALİZİ (TAM YENİLENMİŞ)
# ======================================
def analyze_signal_quality(rsrp_value):
    """RSRP değerine göre sinyal kalitesi analizi - tam düzeltilmiş"""
    try:
        if pd.isna(rsrp_value):
            return "Unknown"
        
        rsrp = float(rsrp_value)
        
        # Makul RSRP aralığı kontrolü (-200 dBm ile 0 dBm arası)
        if rsrp > 0 or rsrp < -200:
            return "Unknown"
        
        # 5G NR için RSRP değerlendirme kriterleri (3GPP standartları)
        if rsrp >= -80:
            return "Excellent"
        elif rsrp >= -90:
            return "Good"
        elif rsrp >= -100:
            return "Fair"
        elif rsrp >= -110:
            return "Poor"
        else:
            return "Very Poor"
            
    except (ValueError, TypeError, AttributeError):
        return "Unknown"

def robust_data_extraction(df, source_name=""):
    """Güçlü veri çıkarma ve temizleme - tam yenilenmiş"""
    print(f"\n🔍 {source_name} - Güçlü Veri Analizi Başlıyor...")
    print(f"Ham veri boyutu: {df.shape}")
    
    # RSRP sütununu bul ve analiz et
    rsrp_column = None
    possible_rsrp_cols = ['NR_UE_RSRP_0', 'RSRP_0', 'RSRP', 'NR_RSRP_0', 'UE_RSRP_0']
    
    for col in possible_rsrp_cols:
        if col in df.columns:
            rsrp_column = col
            break
    
    if rsrp_column is None:
        print(f"❌ RSRP sütunu bulunamadı! Mevcut sütunlar: {list(df.columns[:10])}")
        return None
    
    print(f"✅ RSRP sütunu bulundu: {rsrp_column}")
    
    # Ham RSRP verilerini incele
    rsrp_raw = df[rsrp_column].copy()
    print(f"Ham RSRP veri tipi: {rsrp_raw.dtype}")
    print(f"Null değer sayısı: {rsrp_raw.isnull().sum()}")
    print(f"Benzersiz değer sayısı: {rsrp_raw.nunique()}")
    
    # Geçerli RSRP değerleri
    valid_rsrp = rsrp_raw.dropna()
    valid_rsrp = valid_rsrp[(valid_rsrp >= -200) & (valid_rsrp <= 0)]
    
    print(f"✅ Geçerli RSRP değeri sayısı: {valid_rsrp.notna().sum()}")
    print(f"Geçersiz/eksik değer: {len(rsrp_raw) - len(valid_rsrp)}")
    
    if len(valid_rsrp) > 0:
        print(f"RSRP istatistikleri:")
        print(f"  Min: {valid_rsrp.min():.1f} dBm")
        print(f"  Max: {valid_rsrp.max():.1f} dBm")
        print(f"  Ortalama: {valid_rsrp.mean():.1f} dBm")
        print(f"  Medyan: {valid_rsrp.median():.1f} dBm")
    
    return valid_rsrp

def generate_signal_quality_report(df, source_name=""):
    """Sinyal kalitesi raporu oluştur - tamamen düzeltilmiş"""
    print(f"\n📊 {source_name} Sinyal Kalitesi Analizi:")
    print("=" * 60)
    
    # RSRP sütununu bul
    rsrp_column = 'NR_UE_RSRP_0'
    if rsrp_column not in df.columns:
        print(f"❌ {rsrp_column} bulunamadı!")
        return None, None
    
    # RSRP verilerini analiz et
    rsrp_data = df[rsrp_column].copy()
    
    # Geçerli değerleri filtrele
    valid_rsrp = rsrp_data.dropna()
    valid_rsrp = valid_rsrp[(valid_rsrp >= -200) & (valid_rsrp <= 0)]
    
    print(f"📈 GENEL İSTATİSTİKLER:")
    print(f"  Toplam ölçüm: {len(rsrp_data):,}")
    print(f"  Geçerli RSRP: {len(valid_rsrp):,} (%{len(valid_rsrp)/len(rsrp_data)*100:.1f})")
    print(f"  Eksik veri: {len(rsrp_data)-len(valid_rsrp):,} (%{(len(rsrp_data)-len(valid_rsrp))/len(rsrp_data)*100:.1f})")
    
    if len(valid_rsrp) == 0:
        print(f"❌ {source_name}: Geçerli RSRP verisi yok!")
        return None, None
    
    # Kalite kategorileri hesapla - SADECE GEÇERLİ VERİLER İÇİN
    quality_categories = valid_rsrp.apply(analyze_signal_quality)
    quality_counts = quality_categories.value_counts()
    quality_percentages = quality_categories.value_counts(normalize=True) * 100
    
    print(f"\n📊 SİNYAL KALİTESİ DAĞILIMI (Sadece Geçerli Veriler):")
    categories_order = ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor', 'Unknown']
    for category in categories_order:
        count = quality_counts.get(category, 0)
        percentage = quality_percentages.get(category, 0)
        emoji = {'Excellent': '🟢', 'Good': '🟡', 'Fair': '🟠', 'Poor': '🔴', 'Very Poor': '⚫', 'Unknown': '⚪'}
        if count > 0:
            print(f"  {emoji.get(category, '📊')} {category:12}: {count:6,} ({percentage:5.1f}%)")
    
    # Geçerli veri istatistikleri
    print(f"\n📡 RSRP İSTATİSTİKLERİ:")
    print(f"  Ortalama: {valid_rsrp.mean():6.1f} dBm")
    print(f"  Medyan:   {valid_rsrp.median():6.1f} dBm")
    print(f"  Minimum:  {valid_rsrp.min():6.1f} dBm")
    print(f"  Maksimum: {valid_rsrp.max():6.1f} dBm")
    print(f"  Std Sap:  {valid_rsrp.std():6.1f} dBm")
    
    # Percentile analizi
    print(f"  25%tile:  {valid_rsrp.quantile(0.25):6.1f} dBm")
    print(f"  75%tile:  {valid_rsrp.quantile(0.75):6.1f} dBm")
    
    # Gelişmiş grafik oluştur
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Sol grafik: Kalite kategorileri (sadece geçerli veriler)
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red', 'lightgray']
        valid_categories = [cat for cat in categories_order if cat in quality_counts.index and quality_counts[cat] > 0]
        valid_counts = [quality_counts[cat] for cat in valid_categories]
        valid_colors = colors[:len(valid_categories)]
        
        bars = ax1.bar(valid_categories, valid_counts, color=valid_colors, alpha=0.8)
        ax1.set_title(f'{source_name} - Sinyal Kalitesi Dağılımı (DÜZELTILMIŞ)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Kalite Kategorisi', fontsize=12)
        ax1.set_ylabel('Ölçüm Sayısı', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Bar üzerinde değerler
        for bar, count in zip(bars, valid_counts):
            height = bar.get_height()
            percentage = count / len(valid_rsrp) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Sağ grafik: RSRP histogramı (sadece geçerli değerler)
        ax2.hist(valid_rsrp, bins=min(50, len(valid_rsrp)//10+1), color='skyblue', alpha=0.7, edgecolor='black')
        ax2.axvline(valid_rsrp.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Ortalama: {valid_rsrp.mean():.1f} dBm')
        ax2.axvline(valid_rsrp.median(), color='green', linestyle='--', linewidth=2,
                   label=f'Medyan: {valid_rsrp.median():.1f} dBm')
        ax2.set_title(f'{source_name} - RSRP Dağılımı (Geçerli Veriler)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('RSRP (dBm)', fontsize=12)
        ax2.set_ylabel('Frekans', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Kalite bölgeleri göster
        ax2.axvspan(-200, -110, alpha=0.1, color='red', label='Very Poor')
        ax2.axvspan(-110, -100, alpha=0.1, color='orange', label='Poor')
        ax2.axvspan(-100, -90, alpha=0.1, color='yellow', label='Fair')
        ax2.axvspan(-90, -80, alpha=0.1, color='lightgreen', label='Good')
        ax2.axvspan(-80, 0, alpha=0.1, color='green', label='Excellent')
        
        plt.tight_layout()
        plt.savefig(f'outputs/{source_name.lower()}_signal_quality_FINAL.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Düzeltilmiş grafik kaydedildi: outputs/{source_name.lower()}_signal_quality_FINAL.png")
        
    except Exception as e:
        print(f"⚠️ Grafik oluşturma hatası: {e}")
    
    return quality_counts, quality_percentages

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
                                      y_cols=['Latitude', 'Longitude'], source_name="DL")
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
                                      y_cols=['Latitude', 'Longitude'], source_name="UL")
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
# 5.2 GELİŞMİŞ ENSEMBLE VE OPTİMİZASYON
# ======================================
class AdvancedPositioningSystem:
    """Gelişmiş konumlandırma sistemi"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.feature_scaler = StandardScaler()
        self.best_params = {}
        
    def create_ensemble_models(self):
        """Ensemble model koleksiyonu oluştur"""
        base_models = {
            'rf': RandomForestRegressor(
                n_estimators=200, 
                max_depth=20, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'gbdt': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2
            ),
            'knn': KNeighborsRegressor(
                n_neighbors=10,
                weights='distance',
                algorithm='ball_tree'
            )
        }
        return base_models
    
    def advanced_feature_engineering(self, X_df):
        """Gelişmiş özellik mühendisliği"""
        X_new = X_df.copy()
        
        # Sütun adlarını string'e çevir
        if hasattr(X_new, 'columns'):
            X_new.columns = [str(col) for col in X_new.columns]
        
        # 1. Sinyal oranları ve farkları
        if 'NR_UE_RSRP_0' in X_new.columns and 'NR_UE_RSRP_1' in X_new.columns:
            X_new['rsrp_ratio'] = X_new['NR_UE_RSRP_0'] / (X_new['NR_UE_RSRP_1'] + 1e-6)
            X_new['rsrp_dominance'] = X_new['NR_UE_RSRP_0'] - X_new['NR_UE_RSRP_1']
        
        # 2. Sinyal kalitesi skorları
        rsrp_cols = [col for col in X_new.columns if 'RSRP' in str(col)]
        if rsrp_cols:
            X_new['signal_strength_score'] = X_new[rsrp_cols].max(axis=1)
            X_new['signal_variability'] = X_new[rsrp_cols].std(axis=1)
            X_new['signal_count'] = X_new[rsrp_cols].notna().sum(axis=1)
        
        # 3. Geometrik özellikler
        if 'bs_distance' in X_new.columns and 'NR_UE_RSRP_0' in X_new.columns:
            # Path loss model based features
            X_new['estimated_pathloss'] = -X_new['NR_UE_RSRP_0']
            X_new['distance_pathloss_ratio'] = X_new['bs_distance'] / (X_new['estimated_pathloss'] + 1e-6)
        
        # 4. Timing based features
        if 'NR_UE_Timing_Advance' in X_new.columns:
            X_new['ta_distance'] = X_new['NR_UE_Timing_Advance'] * 150  # TA to distance conversion
            if 'bs_distance' in X_new.columns:
                X_new['ta_distance_diff'] = abs(X_new['ta_distance'] - X_new['bs_distance'])
        
        # 5. Multi-cell features
        pci_cols = [col for col in X_new.columns if 'PCI' in str(col)]
        if len(pci_cols) > 1:
            X_new['serving_cell_count'] = X_new[pci_cols].notna().sum(axis=1)
        
        # 6. Statistical features
        signal_cols = [col for col in X_new.columns if any(sig in str(col) for sig in ['RSRP', 'RSRQ', 'SINR'])]
        if signal_cols:
            X_new['signal_mean'] = X_new[signal_cols].mean(axis=1)
            X_new['signal_median'] = X_new[signal_cols].median(axis=1)
            X_new['signal_skew'] = X_new[signal_cols].skew(axis=1)
        
        return X_new
    
    def optimize_hyperparameters(self, X_train, y_train, model_name='rf'):
        """Hiperparametre optimizasyonu"""
        param_grids = {
            'rf': {
                'n_estimators': [100, 150, 200],
                'max_depth': [15, 20, 25],
                'min_samples_split': [3, 5, 7]
            },
            'xgb': {
                'n_estimators': [100, 150],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.08, 0.1, 0.12]
            }
        }
        
        if model_name in param_grids:
            base_model = self.create_ensemble_models()[model_name]
            grid_search = GridSearchCV(
                base_model,
                param_grids[model_name],
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            self.best_params[model_name] = grid_search.best_params_
            return grid_search.best_estimator_
        else:
            return self.create_ensemble_models()[model_name]
    
    def train_ensemble(self, X_train, y_train):
        """Ensemble model eğitimi"""
        print("🔧 Gelişmiş ensemble model eğitimi başlatılıyor...")
        
        # Feature engineering
        X_train_advanced = self.advanced_feature_engineering(pd.DataFrame(X_train))
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train_advanced.fillna(0))
        
        # Create base models (only use reliable ones for multi-output)
        reliable_models = {
            'rf': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            'xgb': xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            'knn': KNeighborsRegressor(n_neighbors=min(5, len(X_train_scaled)), weights='distance')
        }
        
        # Train individual models
        trained_models = []
        model_scores = {}
        
        for name, model in reliable_models.items():
            print(f"  🏋️ {name} modeli eğitiliyor...")
            try:
                model.fit(X_train_scaled, y_train)
                
                # Simple validation score using training data
                y_pred_train = model.predict(X_train_scaled)
                train_error = np.sqrt(mean_squared_error(y_train, y_pred_train))
                
                model_scores[name] = train_error
                trained_models.append((name, model))
                print(f"    ✅ {name} Train RMSE: {train_error:.4f}")
            except Exception as e:
                print(f"    ❌ {name} eğitim hatası: {e}")
        
        # Store trained models
        self.trained_models = dict(trained_models)
        
        # Calculate weights based on inverse error (lower error = higher weight)
        if model_scores:
            # Inverse weights (better models get higher weights)
            total_inv_error = sum(1.0 / (score + 1e-6) for score in model_scores.values())
            self.model_weights = {}
            for name, score in model_scores.items():
                self.model_weights[name] = (1.0 / (score + 1e-6)) / total_inv_error
        else:
            self.model_weights = {}
        
        print(f"🎯 Ensemble model hazır! {len(trained_models)} model birleştirildi.")
        if self.model_weights:
            weights_str = ", ".join([f"{name}: {weight:.3f}" for name, weight in self.model_weights.items()])
            print(f"📊 Model ağırlıkları: {weights_str}")
        
        return model_scores
    
    def predict_advanced(self, X_test):
        """Gelişmiş tahmin"""
        X_test_advanced = self.advanced_feature_engineering(pd.DataFrame(X_test))
        X_test_scaled = self.feature_scaler.transform(X_test_advanced.fillna(0))
        
        if hasattr(self, 'trained_models') and self.trained_models:
            # Weighted ensemble prediction
            predictions = []
            weights = []
            
            for name, model in self.trained_models.items():
                try:
                    pred = model.predict(X_test_scaled)
                    
                    # Check for NaN values
                    if not np.isnan(pred).any():
                        predictions.append(pred)
                        weights.append(self.model_weights.get(name, 1.0))
                    else:
                        print(f"⚠️ {name} model returned NaN, skipping")
                        
                except Exception as e:
                    print(f"⚠️ {name} model prediction error: {e}")
            
            if predictions:
                # Weighted average
                weights = np.array(weights)
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)  # Normalize weights
                    final_prediction = np.average(predictions, axis=0, weights=weights)
                    
                    # Final NaN check
                    if np.isnan(final_prediction).any():
                        print("⚠️ Final prediction contains NaN, using first valid prediction")
                        final_prediction = predictions[0]
                    
                    return final_prediction
                else:
                    return predictions[0]  # Use first prediction if no weights
            else:
                raise ValueError("Hiçbir model tahmin yapamadı!")
        else:
            raise ValueError("Model henüz eğitilmemiş!")

# ======================================
# 5.3 MODEL AÇIKLANABİLİRİĞİ (EXPLAIİNABLE AIİ)
# ======================================
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP yüklü değil, model açıklamaları atlanacak")

def explain_prediction(model, X_test, feature_names, save_path='outputs/'):
    """Model tahminlerini açıkla"""
    if not SHAP_AVAILABLE:
        print("SHAP gerekli, açıklama atlanıyor")
        return
    
    try:
        # SHAP explainer oluştur
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:100])  # İlk 100 örnek
        
        # Summary plot kaydet
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, show=False)
        plt.savefig(f'{save_path}shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, plot_type="bar", show=False)
        plt.savefig(f'{save_path}shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ SHAP açıklamaları kaydedildi: {save_path}")
        
    except Exception as e:
        print(f"❌ SHAP açıklama hatası: {e}")

# ======================================
# 5.4 PERFORMANS İZLEME VE ANALİTİK
# ======================================
class PerformanceAnalyzer:
    """Performans analiz ve izleme sistemi"""
    
    def __init__(self):
        self.metrics_history = []
        self.prediction_history = []
    
    def analyze_model_performance(self, y_true, y_pred, model_name="Model"):
        """Detaylı model performans analizi"""
        errors = np.linalg.norm(y_pred - y_true, axis=1)
        
        metrics = {
            'model_name': model_name,
            'timestamp': time.time(),
            'n_samples': len(y_true),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'median_error': float(np.median(errors)),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors)),
            'std_error': float(np.std(errors)),
            'r2_lat': float(r2_score(y_true[:, 0], y_pred[:, 0])),
            'r2_lon': float(r2_score(y_true[:, 1], y_pred[:, 1])),
            'percent_under_5m': float(np.sum(errors < 5) / len(errors) * 100),
            'percent_under_10m': float(np.sum(errors < 10) / len(errors) * 100),
            'percent_under_50m': float(np.sum(errors < 50) / len(errors) * 100),
            'percent_under_100m': float(np.sum(errors < 100) / len(errors) * 100)
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def create_performance_dashboard(self, save_path='outputs/'):
        """Performans dashboard'u oluştur"""
        if not self.metrics_history:
            print("Henüz performans verisi yok")
            return
        
        # Performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # RMSE comparison
        models = [m['model_name'] for m in self.metrics_history]
        rmse_values = [m['rmse'] for m in self.metrics_history]
        
        axes[0, 0].bar(models, rmse_values, color='skyblue')
        axes[0, 0].set_title('RMSE Karşılaştırması')
        axes[0, 0].set_ylabel('RMSE (m)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Accuracy percentages
        accuracy_metrics = ['percent_under_5m', 'percent_under_10m', 'percent_under_50m']
        for i, metric in enumerate(accuracy_metrics):
            values = [m[metric] for m in self.metrics_history]
            axes[0, 1].plot(models, values, marker='o', label=f'{metric.replace("percent_under_", "").replace("m", "m accuracy")}')
        
        axes[0, 1].set_title('Doğruluk Oranları')
        axes[0, 1].set_ylabel('Doğruluk (%)')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² scores
        r2_lat = [m['r2_lat'] for m in self.metrics_history]
        r2_lon = [m['r2_lon'] for m in self.metrics_history]
        
        x_pos = np.arange(len(models))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, r2_lat, width, label='Latitude R²', color='lightcoral')
        axes[1, 0].bar(x_pos + width/2, r2_lon, width, label='Longitude R²', color='lightblue')
        axes[1, 0].set_title('R² Skorları')
        axes[1, 0].set_ylabel('R² Skoru')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].legend()
        
        # Error distribution
        latest_model = self.metrics_history[-1]
        error_stats = ['min_error', 'median_error', 'max_error']
        error_values = [latest_model[stat] for stat in error_stats]
        
        axes[1, 1].bar(error_stats, error_values, color='lightgreen')
        axes[1, 1].set_title(f'Hata Dağılımı ({latest_model["model_name"]})')
        axes[1, 1].set_ylabel('Hata (m)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Performans dashboard'u kaydedildi: {save_path}performance_dashboard.png")
    
    def save_performance_report(self, save_path='outputs/performance_report.json'):
        """Performans raporunu kaydet"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, ensure_ascii=False, indent=4)
        print(f"✅ Performans raporu kaydedildi: {save_path}")

# Global performance analyzer
performance_analyzer = PerformanceAnalyzer()

# ======================================
# 6. MODEL KURMA VE DEĞERLENDİRME
# ======================================
def train_and_evaluate_advanced(X, y, source_name=""):
    """Geliştirilmiş model eğitimi ve değerlendirmesi"""
    if X.shape[0] > 0:
        print(f"\n🚀 {source_name} - Gelişmiş model eğitimi başlıyor...")
        
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"  📊 Eğitim: {X_train.shape[0]} satır, Test: {X_test.shape[0]} satır")
        
        # Gelişmiş konumlandırma sistemi
        advanced_system = AdvancedPositioningSystem()
        
        # Ensemble model eğitimi
        model_scores = advanced_system.train_ensemble(X_train, y_train)
        
        # Tahmin yap
        y_pred = advanced_system.predict_advanced(X_test)
        
        # Performans analizi
        metrics = performance_analyzer.analyze_model_performance(y_test, y_pred, f"{source_name}_Advanced")
        
        # Senaryo analizi
        scenario = analyze_scenario(pd.DataFrame(X_test).iloc[0], cellinfo)
        metrics['scenario'] = scenario
        
        # Sonuçları yazdır
        print(f"\n📈 {source_name} Gelişmiş Model Sonuçları:")
        print(f"  🎯 RMSE: {metrics['rmse']:.2f} m")
        print(f"  📏 MAE: {metrics['mae']:.2f} m")
        print(f"  📊 Medyan Hata: {metrics['median_error']:.2f} m")
        print(f"  🔥 %5m Altı: {metrics['percent_under_5m']:.1f}%")
        print(f"  🔥 %10m Altı: {metrics['percent_under_10m']:.1f}%")
        print(f"  🔥 %50m Altı: {metrics['percent_under_50m']:.1f}%")
        print(f"  📈 R² (Lat): {metrics['r2_lat']:.3f}")
        print(f"  📈 R² (Lon): {metrics['r2_lon']:.3f}")
        
        # Model açıklaması (SHAP varsa)
        if hasattr(advanced_system.ensemble_model, 'estimators_'):
            feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
            explain_prediction(advanced_system.ensemble_model, X_test, feature_names)
        
        return advanced_system, metrics, model_scores
    
    return None, None, None

# DL Modeli
if X_dl.size > 0:
    dl_model, dl_errors, dl_metrics = train_and_evaluate_advanced(X_dl, y_dl, "DL")
else:
    print("DL verisi bulunamadı, model eğitilemiyor.")
    dl_model, dl_errors, dl_metrics = None, None, None

# UL Modeli
if X_ul.size > 0:
    ul_model, ul_errors, ul_metrics = train_and_evaluate_advanced(X_ul, y_ul, "UL")
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
    print("TEKNOFEST 2025 5G Konumlandırma Sistemi - Gelişmiş Sürüm")
    print("=" * 60)
    
    # Sinyal kalitesi analizleri
    print("\n🔍 Sinyal kalitesi analizleri yapılıyor...")
    if dl_series is not None and not dl_series.empty:
        generate_signal_quality_report(dl_series, "DL")
    
    if ul_series is not None and not ul_series.empty:
        generate_signal_quality_report(ul_series, "UL")
    
    if scanner_series is not None and not scanner_series.empty:
        generate_signal_quality_report(scanner_series, "Scanner")
    
    # Gelişmiş modelleri kaydet
    print("\n💾 Gelişmiş modeller kaydediliyor...")
    if dl_model is not None:
        joblib.dump(dl_model, 'models/dl_advanced_model.pkl')
        print("✅ DL gelişmiş modeli kaydedildi.")
    if ul_model is not None:
        joblib.dump(ul_model, 'models/ul_advanced_model.pkl')
        print("✅ UL gelişmiş modeli kaydedildi.")
    
    # Performans analizi ve dashboard
    print("\n📊 Performans analizi ve dashboard oluşturuluyor...")
    performance_analyzer.create_performance_dashboard()
    performance_analyzer.save_performance_report()
    
    # Sonuçları kaydet
    metrics_to_save = {}
    if dl_metrics is not None:
        metrics_to_save['DL_Advanced'] = dl_metrics
    if ul_metrics is not None:
        metrics_to_save['UL_Advanced'] = ul_metrics
        
    if metrics_to_save:
        save_metrics(metrics_to_save, 'outputs/advanced_model_metrics.json')
        print("✅ Gelişmiş metrikler kaydedildi.")
    
    # Özellik açıklamalarını kaydet
    if X_dl.size > 0:
        feature_names = main_features + nbr_pci + nbr_rsrp + nbr_rsrq + ['bs_distance', 'bs_azimuth_diff', 'rsrp_diff_0_1']
        save_feature_descriptions(feature_names, 'outputs/advanced_feature_descriptions.csv')
        print("✅ Gelişmiş özellik açıklamaları kaydedildi.")
    
    # Sistem özeti
    print("\n" + "="*60)
    print("🎯 SİSTEM ÖZET RAPORU")
    print("="*60)
    
    if performance_analyzer.metrics_history:
        latest_dl = next((m for m in performance_analyzer.metrics_history if 'DL' in m['model_name']), None)
        latest_ul = next((m for m in performance_analyzer.metrics_history if 'UL' in m['model_name']), None)
        
        print("📈 En İyi Performans Skorları:")
        if latest_dl:
            print(f"  🟢 DL Model RMSE: {latest_dl['rmse']:.2f}m (%5m altı: {latest_dl['percent_under_5m']:.1f}%)")
        if latest_ul:
            print(f"  🟡 UL Model RMSE: {latest_ul['rmse']:.2f}m (%5m altı: {latest_ul['percent_under_5m']:.1f}%)")
        
        if latest_dl and latest_ul:
            best_model = "DL" if latest_dl['rmse'] < latest_ul['rmse'] else "UL"
            print(f"  🏆 En İyi Model: {best_model}")
    
    print("\n🎯 Sistem hazır! Gelişmiş yarışma modunu test etmek için:")
    print("python main.py --test  # Otomatik test")
    print("# Gelişmiş gerçek zamanlı test:")
    print("predict_realtime_advanced('data/Kopya5G_DL.xlsx', 'hybrid')")

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
        # Excel dosyasının tüm sheet'lerini yükle
        input_sheets = pd.read_excel(input_file, sheet_name=None)
        cellinfo = pd.read_excel(cell_info_file)
        print(f"Veriler başarıyla yüklendi. Sheet sayısı: {len(input_sheets)}")
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None
    
    # Series sayfasını bul
    series_data = None
    for sheet_name, data in input_sheets.items():
        if 'Series' in sheet_name:
            series_data = data
            print(f"Kullanılan sheet: {sheet_name}, Satır sayısı: {len(data)}")
            break
    
    if series_data is None:
        print("Series verisi bulunamadı!")
        return None
    
    # İyileştirilmiş test noktası seçimi - kalite bazlı çeşitlilik
    test_row = None
    actual_coords = None
    
    # Kaliteli test verilerini bul ve kategorilere ayır
    uygun_satirlar = []
    excellent_quality = []  # RSRP > -85
    good_quality = []       # -85 >= RSRP > -95
    fair_quality = []       # -95 >= RSRP > -105
    poor_quality = []       # RSRP <= -105
    
    for idx, row in series_data.iterrows():
        # Geçerli PCI ve sinyal değeri var mı?
        if pd.notna(row.get('NR_UE_PCI_0')) and pd.notna(row.get('NR_UE_RSRP_0')):
            # Konum bilgisi de var mı?
            if pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')):
                rsrp = float(row.get('NR_UE_RSRP_0', -120))
                
                # Sinyal kalitesine göre kategorilere ayır
                if rsrp > -85:
                    excellent_quality.append((idx, row, rsrp))
                elif rsrp > -95:
                    good_quality.append((idx, row, rsrp))
                elif rsrp > -105:
                    fair_quality.append((idx, row, rsrp))
                else:
                    poor_quality.append((idx, row, rsrp))
                
                uygun_satirlar.append((idx, row, rsrp))
    
    if not uygun_satirlar:
        print("Hem konum hem sinyal bilgisi olan test verisi bulunamadı!")
        return None
    
    print(f"✅ Toplam {len(uygun_satirlar)} uygun test satırı bulundu")
    print(f"📊 Sinyal kalitesi dağılımı:")
    print(f"   🟢 Excellent (-85+ dBm): {len(excellent_quality)} satır")
    print(f"   🟡 Good (-85 to -95 dBm): {len(good_quality)} satır")
    print(f"   🟠 Fair (-95 to -105 dBm): {len(fair_quality)} satır")
    print(f"   🔴 Poor (-105- dBm): {len(poor_quality)} satır")
    
    # Çeşitlilik için farklı kalite seviyelerinden seçim yap
    # Öncelik sırası: Good > Fair > Excellent > Poor (orta kalite test için daha ideal)
    selection_pools = [
        ("Good", good_quality),
        ("Fair", fair_quality), 
        ("Excellent", excellent_quality),
        ("Poor", poor_quality)
    ]
    
    secilen_idx = None
    test_row = None
    quality_level = "Unknown"
    rsrp_value = -120
    
    # İlk dolu kategoriden rastgele seç
    for level, pool in selection_pools:
        if pool:
            secilen_idx, test_row, rsrp_value = random.choice(pool)
            quality_level = level
            print(f"🎯 {quality_level} kalitesinden rastgele seçim yapıldı")
            break
    
    # Hiçbir kategoride veri yoksa genel listeden seç
    if test_row is None:
        secilen_idx, test_row, rsrp_value = random.choice(uygun_satirlar)
        quality_level = analyze_signal_quality(rsrp_value)
        print(f"🎲 Genel listeden rastgele seçim yapıldı")
    
    actual_coords = [test_row['Latitude'], test_row['Longitude']]
    
    print(f"🎲 Seçilen test satırı: {secilen_idx} (Kalite: {quality_level})")
    print(f"✅ Test verisi gerçek konumu: {actual_coords[0]:.6f}, {actual_coords[1]:.6f}")
    print(f"📶 Sinyal gücü: {test_row['NR_UE_RSRP_0']:.1f} dBm ({quality_level})")
    print(f"📡 PCI: {test_row['NR_UE_PCI_0']}")
    
    # Ek test verisi bilgileri
    if 'NR_UE_SINR_0' in test_row and pd.notna(test_row['NR_UE_SINR_0']):
        print(f"📡 SINR: {test_row['NR_UE_SINR_0']:.1f} dB")
    if 'NR_UE_RSRQ_0' in test_row and pd.notna(test_row['NR_UE_RSRQ_0']):
        print(f"📡 RSRQ: {test_row['NR_UE_RSRQ_0']:.1f} dB")
    if 'NR_UE_Timing_Advance' in test_row and pd.notna(test_row['NR_UE_Timing_Advance']):
        print(f"⏱️ Timing Advance: {test_row['NR_UE_Timing_Advance']:.1f} μs")
    
    # Öznitelik tanımlamaları (main.py'deki ile aynı)
    main_features = [
        'NR_UE_PCI_0', 'NR_UE_RSRP_0', 'NR_UE_RSRQ_0', 'NR_UE_SINR_0',
        'NR_UE_Timing_Advance', 'NR_UE_Pathloss_DL_0',
        'NR_UE_Throughput_PDCP_DL', 'NR_UE_MCS_DL_0'
    ]
    nbr_pci = [f'NR_UE_Nbr_PCI_{i}' for i in range(5)]
    nbr_rsrp = [f'NR_UE_Nbr_RSRP_{i}' for i in range(5)]
    nbr_rsrq = [f'NR_UE_Nbr_RSRQ_{i}' for i in range(5)]
    
    # Tek satırlık DataFrame oluştur
    test_df = pd.DataFrame([test_row])
    
    # Veri temizleme ve öznitelik çıkarma
    all_features = main_features + nbr_pci + nbr_rsrp + nbr_rsrq
    available = [col for col in all_features if col in test_df.columns]
    X = test_df[available].fillna(-110)
    
    print(f"Kullanılan öznitelik sayısı: {len(available)}")
    
    # Hücre bilgileri ve coğrafi özellikler ekleme
    cellinfo['PCI'] = cellinfo['PCI '] if 'PCI ' in cellinfo.columns else cellinfo['PCI']
    
    # Baz istasyonu bilgilerini al ve öznitelikleri ekle
    def get_bs_coords(pci):
        match = cellinfo[cellinfo['PCI'] == pci]
        if not match.empty:
            return match.iloc[0]['Latitude'], match.iloc[0]['Longitude'], match.iloc[0]['Azimuth [°]']
        else:
            return np.nan, np.nan, np.nan
    
    # Özellik çıkarımı
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
        predicted_coords = predict_hybrid(X_np, feature_cols, scenario, cellinfo, X.iloc[0])
        pci = X.iloc[0]['NR_UE_PCI_0']
        bs_lat, bs_lon, _ = get_bs_coords(pci)
        calc_time = time.time() - start_time
        
        # Haritada göster - gerçek konum ve tahmin edilen konum
        map_file = visualize_prediction(
            predicted_coords, 
            pci, 
            bs_lat, 
            bs_lon, 
            scenario, 
            calc_time, 
            model_type="hybrid",
            actual_coords=actual_coords,
            input_data=test_row
        )
    else:
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
        
        # Haritada göster - gerçek konum ve tahmin edilen konum
        map_file = visualize_prediction(
            predicted_coords, 
            pci, 
            bs_lat, 
            bs_lon, 
            scenario, 
            calc_time, 
            model_type=model_type,
            actual_coords=actual_coords,
            input_data=test_row
        )
    
    # Sonuç özeti yazdır
    if actual_coords is not None:
        error_distance = geodesic(actual_coords, predicted_coords).meters
        print(f"\n📊 SONUÇ ÖZETİ:")
        print(f"🔴 Gerçek Konum: {actual_coords[0]:.6f}, {actual_coords[1]:.6f}")
        print(f"🟢 Tahmin Konum: {predicted_coords[0]:.6f}, {predicted_coords[1]:.6f}")
        print(f"📏 Hata Mesafesi: {error_distance:.2f} metre")
        print(f"⚡ Hesaplama Süresi: {calc_time:.3f} saniye")
    
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
    pci = row_data['NR_UE_PCI_0']
    bs_lat, bs_lon, _ = get_bs_coords(pci, cellinfo)
    
    # Hesaplama süresini ölç
    calc_time = time.time() - start_time
    print(f"Hibrit tahmin edilen konum: Lat: {predicted_coords[0]:.6f}, Lon: {predicted_coords[1]:.6f}")
    print(f"Hesaplama süresi: {calc_time:.2f} saniye")
    
    return predicted_coords

def get_bs_coords(pci, cellinfo):
    """Baz istasyonu koordinatlarını döndürür"""
    match = cellinfo[cellinfo['PCI'] == pci]
    if not match.empty:
        return match.iloc[0]['Latitude'], match.iloc[0]['Longitude'], match.iloc[0]['Azimuth [°]']
    else:
        return np.nan, np.nan, np.nan

def visualize_prediction(predicted_coords, pci, bs_lat, bs_lon, scenario, calc_time, model_type="standard", actual_coords=None, input_data=None):
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
            
            # Gerçek test verisi varsa kırmızı pin olarak ekle
            if actual_coords is not None:
                folium.Marker(
                    location=[actual_coords[0], actual_coords[1]],
                    popup=folium.Popup(f"""
                    <b>🔴 GERÇEK TEST KONUMİ</b><br>
                    <b>Koordinat:</b> {actual_coords[0]:.6f}, {actual_coords[1]:.6f}<br>
                    <b>PCI:</b> {pci}<br>
                    <b>Test Verisi</b>
                    """, max_width=400),
                    tooltip="Gerçek Test Konumu",
                    icon=folium.Icon(color='red', icon='map-pin', prefix='fa')
                ).add_to(harita)
            
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
                icon=folium.Icon(color='green', icon='crosshairs', prefix='fa')
            ).add_to(harita)
            
            # Hata hesaplama ve gösterme
            if actual_coords is not None:
                hata_mesafe = geodesic(actual_coords, predicted_coords).meters
                
                # Hata çizgisi çiz
                folium.PolyLine(
                    locations=[actual_coords, predicted_coords],
                    color='orange',
                    weight=3,
                    opacity=0.8,
                    popup=f"Tahmin Hatası: {hata_mesafe:.2f} m"
                ).add_to(harita)
                
                # Orta noktada hata bilgisi
                orta_lat = (actual_coords[0] + predicted_coords[0]) / 2
                orta_lon = (actual_coords[1] + predicted_coords[1]) / 2
                
                folium.Marker(
                    location=[orta_lat, orta_lon],
                    popup=f"Hata: {hata_mesafe:.2f} m",
                    icon=folium.DivIcon(html=f"""
                    <div style="font-size: 12px; color: orange; font-weight: bold; background: white; padding: 2px; border-radius: 3px; border: 1px solid orange;">
                    ⚠️ {hata_mesafe:.2f}m
                    </div>""")
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
                    popup=f"BS Mesafesi: {mesafe:.1f} m",
                    icon=folium.DivIcon(html=f"""
                    <div style="font-size: 12px; color: blue; font-weight: bold;">
                    📡 {mesafe:.1f}m
                    </div>""")
                ).add_to(harita)
            
            # Test verilerinde RSRP varsa, sinyal bilgilerini ekle
            if input_data is not None and 'NR_UE_RSRP_0' in input_data:
                rsrp = input_data['NR_UE_RSRP_0']
                sinr = input_data.get('NR_UE_SINR_0', 'N/A')
                rsrq = input_data.get('NR_UE_RSRQ_0', 'N/A')
                
                # Sinyal bilgisi kutusu
                folium.Marker(
                    location=[predicted_coords[0] + 0.0001, predicted_coords[1] + 0.0001],
                    popup=f"Sinyal: RSRP={rsrp} dBm, SINR={sinr} dB, RSRQ={rsrq} dB",
                    icon=folium.DivIcon(html=f"""
                    <div style="font-size: 10px; color: black; background: lightblue; padding: 3px; border-radius: 5px; border: 1px solid blue;">
                    📶 {rsrp}dBm
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
        
        # Gerçek konum varsa kırmızı pin
        if actual_coords is not None:
            folium.Marker(
                location=[actual_coords[0], actual_coords[1]],
                popup="Gerçek Test Konumu",
                icon=folium.Icon(color='red', icon='map-pin')
            ).add_to(m)
        
        # Tahmin edilen konumu göster
        folium.Marker(
            location=[predicted_coords[0], predicted_coords[1]],
            popup="Tahmin Edilen Konum",
            icon=folium.Icon(color='green', icon='crosshairs')
        ).add_to(m)
        
        # Baz istasyonu konumunu göster
        if not np.isnan(bs_lat) and not np.isnan(bs_lon):
            folium.Marker(
                location=[bs_lat, bs_lon],
                popup=f"Baz İstasyonu (PCI: {pci})",
                icon=folium.Icon(color='blue', icon='antenna')
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
    results_data = {
        'Predicted_Latitude': [predicted_coords[0]],
        'Predicted_Longitude': [predicted_coords[1]],
        'Base_Station_PCI': [pci],
        'BS_Latitude': [bs_lat],
        'BS_Longitude': [bs_lon],
        'Environment': [scenario['environment']],
        'LOS_Probability': [scenario['los_probability']],
        'Model_Type': [model_type],
        'Calculation_Time_Seconds': [calc_time]
    }
    
    # Gerçek konum varsa ekle
    if actual_coords is not None:
        results_data['Actual_Latitude'] = [actual_coords[0]]
        results_data['Actual_Longitude'] = [actual_coords[1]]
        
        # Hata hesapla
        hata_mesafe = geodesic(actual_coords, predicted_coords).meters
        results_data['Error_Distance_Meters'] = [hata_mesafe]
    
    results = pd.DataFrame(results_data)
    results.to_csv(f"outputs/tahmin_sonuclari_{timestamp}.csv", index=False)
    
    return map_file

def predict_realtime_advanced(input_file, cell_info_file='data/ITU5GHucreBilgileri.xlsx', model_type='hybrid'):
    """
    Gelişmiş gerçek zamanlı konum tahmini sistemi
    
    Args:
        input_file: Gelen veri dosyası (excel)
        cell_info_file: Baz istasyonu bilgileri dosyası
        model_type: Kullanılacak model tipi ('dl', 'ul', 'hybrid', 'advanced')
    
    Returns:
        predicted_coords: Tahmin edilen koordinatlar (lat, lon)
    """
    print(f"🚀 Gelişmiş gerçek zamanlı tahmin sistemi başlatılıyor...")
    print(f"📁 Dosya: {input_file}")
    print(f"🤖 Model Tipi: {model_type}")
    start_time = time.time()
    
    # Veri dosyalarını yükle
    try:
        input_sheets = pd.read_excel(input_file, sheet_name=None)
        cellinfo = pd.read_excel(cell_info_file)
        print(f"✅ Veriler yüklendi. Sheet sayısı: {len(input_sheets)}")
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        return None
    
    # Series sayfasını bul
    series_data = None
    for sheet_name, data in input_sheets.items():
        if 'Series' in sheet_name:
            series_data = data
            break
    
    if series_data is None:
        print("❌ Series verisi bulunamadı!")
        return None
    
    # Gelişmiş test noktası seçimi
    uygun_satirlar = []
    kalite_kategorileri = {'excellent': [], 'good': [], 'fair': [], 'poor': []}
    
    for idx, row in series_data.iterrows():
        if (pd.notna(row.get('NR_UE_PCI_0')) and 
            pd.notna(row.get('NR_UE_RSRP_0')) and 
            pd.notna(row.get('Latitude')) and 
            pd.notna(row.get('Longitude'))):
            
            rsrp = float(row.get('NR_UE_RSRP_0', -120))
            kalite = analyze_signal_quality(rsrp)
            
            if kalite == 'Excellent':
                kalite_kategorileri['excellent'].append((idx, row, rsrp))
            elif kalite == 'Good':
                kalite_kategorileri['good'].append((idx, row, rsrp))
            elif kalite == 'Fair':
                kalite_kategorileri['fair'].append((idx, row, rsrp))
            else:
                kalite_kategorileri['poor'].append((idx, row, rsrp))
            
            uygun_satirlar.append((idx, row, rsrp))
    
    if not uygun_satirlar:
        print("❌ Uygun test verisi bulunamadı!")
        return None
    
    # Akıllı test noktası seçimi
    secim_oncelik = ['good', 'fair', 'excellent', 'poor']
    test_row = None
    kalite_seviye = "unknown"
    
    for seviye in secim_oncelik:
        if kalite_kategorileri[seviye]:
            secilen_idx, test_row, rsrp_value = random.choice(kalite_kategorileri[seviye])
            kalite_seviye = seviye
            break
    
    if test_row is None:
        secilen_idx, test_row, rsrp_value = random.choice(uygun_satirlar)
    
    actual_coords = [test_row['Latitude'], test_row['Longitude']]
    
    print(f"\n📊 Test Verisi Analizi:")
    print(f"  🎯 Seçilen Satır: {secilen_idx}")
    print(f"  📍 Gerçek Konum: {actual_coords[0]:.6f}, {actual_coords[1]:.6f}")
    print(f"  📶 Sinyal Kalitesi: {kalite_seviye.title()} ({rsrp_value:.1f} dBm)")
    print(f"  📡 PCI: {test_row['NR_UE_PCI_0']}")
    
    # Özellik çıkarma
    test_df = pd.DataFrame([test_row])
    all_features = main_features + nbr_pci + nbr_rsrp + nbr_rsrq
    available = [col for col in all_features if col in test_df.columns]
    X = test_df[available].fillna(-110)
    
    # Coğrafi özellikler ekle
    cellinfo['PCI'] = cellinfo['PCI '] if 'PCI ' in cellinfo.columns else cellinfo['PCI']
    
    def get_bs_coords(pci):
        match = cellinfo[cellinfo['PCI'] == pci]
        if not match.empty:
            return match.iloc[0]['Latitude'], match.iloc[0]['Longitude'], match.iloc[0]['Azimuth [°]']
        else:
            return np.nan, np.nan, np.nan
    
    # Baz istasyonu bilgileri
    pci = X.iloc[0]['NR_UE_PCI_0']
    bs_lat, bs_lon, bs_azimuth = get_bs_coords(pci)
    
    # Ek özellikler
    if 'NR_UE_Timing_Advance' in X.columns and pd.notna(X.iloc[0]['NR_UE_Timing_Advance']):
        ta = X.iloc[0]['NR_UE_Timing_Advance']
        bs_distance = ClassicPositioning.calculate_toa(ta)
    else:
        bs_distance = np.nan
    
    X['bs_distance'] = bs_distance
    X['bs_azimuth_diff'] = 0
    X['rsrp_diff_0_1'] = X.get('NR_UE_RSRP_0', 0) - X.get('NR_UE_RSRP_1', 0)
    
    # Senaryo analizi
    scenario = analyze_scenario(X.iloc[0], cellinfo)
    print(f"  🌍 Çevre: {scenario['environment']}")
    print(f"  📡 LOS Olasılığı: {scenario['los_probability']:.2f}")
    
    # Model seçimi ve tahmin
    prediction_time = time.time()
    
    if model_type == 'advanced':
        # Gelişmiş ensemble model kullan
        try:
            # Mevcut advanced model'i yükle
            if os.path.exists('models/dl_advanced_model.pkl'):
                advanced_model = joblib.load('models/dl_advanced_model.pkl')
                
                # Gelişmiş özellik çıkarma
                if hasattr(advanced_model, 'advanced_feature_engineering'):
                    X_advanced = advanced_model.advanced_feature_engineering(X)
                    X_scaled = advanced_model.feature_scaler.transform(X_advanced.fillna(0))
                    predicted_coords = advanced_model.predict_advanced(X_scaled)[0]
                else:
                    # Fallback: basit tahmin
                    X_np = X.values
                    predicted_coords = advanced_model.predict(X_np)[0]
                
                model_info = "Advanced Ensemble"
            else:
                print("⚠️ Gelişmiş model bulunamadı, hibrit model kullanılıyor")
                predicted_coords = predict_hybrid_advanced(X, scenario, cellinfo)
                model_info = "Hybrid Fallback"
                
        except Exception as e:
            print(f"❌ Gelişmiş model hatası: {e}")
            predicted_coords = predict_hybrid_advanced(X, scenario, cellinfo)
            model_info = "Error Fallback"
    else:
        # Mevcut hibrit sistem kullan
        predicted_coords = predict_hybrid_advanced(X, scenario, cellinfo)
        model_info = "Hybrid"
    
    calc_time = time.time() - start_time
    pred_time = time.time() - prediction_time
    
    print(f"\n🎯 Tahmin Sonuçları:")
    print(f"  🤖 Kullanılan Model: {model_info}")
    print(f"  📍 Tahmin Konum: {predicted_coords[0]:.6f}, {predicted_coords[1]:.6f}")
    print(f"  ⏱️ Toplam Süre: {calc_time:.3f}s")
    print(f"  ⚡ Tahmin Süresi: {pred_time:.3f}s")
    
    # Hata hesaplama
    if actual_coords is not None:
        error_distance = geodesic(actual_coords, predicted_coords).meters
        print(f"  📏 Hata: {error_distance:.2f} metre")
        
        # Performans kategorisi
        if error_distance < 5:
            performance = "🎯 Mükemmel"
        elif error_distance < 10:
            performance = "🟢 Çok İyi"
        elif error_distance < 50:
            performance = "🟡 İyi"
        elif error_distance < 100:
            performance = "🟠 Orta"
        else:
            performance = "🔴 Zayıf"
        
        print(f"  🏆 Performans: {performance}")
    
    # Gelişmiş harita görselleştirmesi
    map_file = visualize_prediction_advanced(
        predicted_coords,
        actual_coords,
        pci,
        bs_lat,
        bs_lon,
        scenario,
        calc_time,
        model_info,
        test_row,
        error_distance if actual_coords else None
    )
    
    return predicted_coords

def predict_hybrid_advanced(X, scenario, cellinfo):
    """Gelişmiş hibrit tahmin sistemi"""
    # Mevcut modelleri yükle
    dl_path = "models/dl_advanced_model.pkl" if os.path.exists("models/dl_advanced_model.pkl") else "models/dl_model.pkl"
    ul_path = "models/ul_advanced_model.pkl" if os.path.exists("models/ul_advanced_model.pkl") else "models/ul_model.pkl"
    
    predictions = []
    weights = []
    
    # DL model tahmini
    if os.path.exists(dl_path):
        try:
            dl_model = joblib.load(dl_path)
            dl_pred = dl_model.predict(X.values)[0] if hasattr(dl_model, 'predict') else dl_model.predict_advanced(X.values)[0]
            predictions.append(dl_pred)
            weights.append(0.6)  # DL'e daha fazla ağırlık
        except Exception as e:
            print(f"⚠️ DL model hatası: {e}")
    
    # UL model tahmini
    if os.path.exists(ul_path):
        try:
            ul_model = joblib.load(ul_path)
            ul_pred = ul_model.predict(X.values)[0] if hasattr(ul_model, 'predict') else ul_model.predict_advanced(X.values)[0]
            predictions.append(ul_pred)
            weights.append(0.4)
        except Exception as e:
            print(f"⚠️ UL model hatası: {e}")
    
    if predictions:
        # Ağırlıklı ortalama
        final_pred = np.average(predictions, axis=0, weights=weights)
        return final_pred
    else:
        # Fallback: sabit koordinat
        return [41.1078, 29.0281]  # ITU kampüs merkezi

def visualize_prediction_advanced(predicted_coords, actual_coords, pci, bs_lat, bs_lon, 
                                scenario, calc_time, model_info, test_row, error_distance):
    """Gelişmiş tahmin haritası oluşturma"""
    timestamp = int(time.time())
    
    # Gelişmiş harita oluşturma
    if KAMPUS_HARITA_MEVCUT:
        try:
            from kampus_harita import KampusHaritasi
            kampus = KampusHaritasi()
            harita = kampus.harita_olustur()
            kampus.baz_istasyonlari_ekle(harita, "data/ITU5GHucreBilgileri.xlsx")
            
            # Gerçek konum (varsa)
            if actual_coords:
                folium.Marker(
                    location=actual_coords,
                    popup=folium.Popup(f"""
                    <b>🔴 GERÇEK KONUM</b><br>
                    📍 Koordinat: {actual_coords[0]:.6f}, {actual_coords[1]:.6f}<br>
                    📶 RSRP: {test_row['NR_UE_RSRP_0']:.1f} dBm<br>
                    📡 PCI: {pci}<br>
                    🌍 Çevre: {scenario['environment']}<br>
                    """, max_width=300),
                    icon=folium.Icon(color='red', icon='map-pin', prefix='fa')
                ).add_to(harita)
            
            # Tahmin edilen konum
            folium.Marker(
                location=predicted_coords,
                popup=folium.Popup(f"""
                <b>🎯 GELIŞMIŞ TAHMİN</b><br>
                🤖 Model: {model_info}<br>
                📍 Koordinat: {predicted_coords[0]:.6f}, {predicted_coords[1]:.6f}<br>
                ⏱️ Süre: {calc_time:.3f}s<br>
                📏 Hata: {error_distance:.2f}m<br>
                🌍 Senaryo: {scenario['environment']}
                """, max_width=350),
                icon=folium.Icon(color='green', icon='bullseye', prefix='fa')
            ).add_to(harita)
            
            # Hata çizgisi
            if actual_coords and error_distance:
                folium.PolyLine(
                    locations=[actual_coords, predicted_coords],
                    color='orange',
                    weight=4,
                    opacity=0.8,
                    popup=f"Hata: {error_distance:.2f}m"
                ).add_to(harita)
            
            # Baz istasyonu bağlantısı
            if not np.isnan(bs_lat) and not np.isnan(bs_lon):
                folium.PolyLine(
                    locations=[[bs_lat, bs_lon], predicted_coords],
                    color='blue',
                    weight=2,
                    opacity=0.6,
                    popup=f"BS Bağlantısı (PCI: {pci})"
                ).add_to(harita)
            
            map_file = f"outputs/advanced_prediction_map_{timestamp}.html"
            harita.save(map_file)
            print(f"🗺️ Gelişmiş harita kaydedildi: {map_file}")
            
        except Exception as e:
            print(f"❌ Gelişmiş harita hatası: {e}")
            map_file = None
    else:
        print("⚠️ Kampüs harita modülü yok")
        map_file = None
    
    # Sonuçları CSV olarak kaydet
    result_data = {
        'timestamp': [timestamp],
        'model_type': [model_info],
        'predicted_lat': [predicted_coords[0]],
        'predicted_lon': [predicted_coords[1]],
        'actual_lat': [actual_coords[0] if actual_coords else None],
        'actual_lon': [actual_coords[1] if actual_coords else None],
        'error_distance_m': [error_distance if error_distance else None],
        'pci': [pci],
        'bs_lat': [bs_lat],
        'bs_lon': [bs_lon],
        'environment': [scenario['environment']],
        'los_probability': [scenario['los_probability']],
        'calculation_time_s': [calc_time],
        'rsrp_dbm': [test_row['NR_UE_RSRP_0']]
    }
    
    results_df = pd.DataFrame(result_data)
    results_file = f"outputs/advanced_prediction_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"💾 Sonuçlar kaydedildi: {results_file}")
    
    return map_file

if __name__ == "__main__":
    main()
    
    # Gerçek zamanlı test için örnekler
    print("\n" + "="*60)
    print("🚀 GERÇEK ZAMANLI TEST ÖRNEKLERİ")
    print("="*60)
    
    # Test fonksiyonu
    def run_test_examples():
        """Gelişmiş test örneklerini çalıştır"""
        test_scenarios = [
            ('dl', 'DL Model Test', False),
            ('ul', 'UL Model Test', False), 
            ('hybrid', 'Hibrit Model Test', False),
            ('advanced', 'Gelişmiş Ensemble Model Test', True)
        ]
        
        print("🧪 Gelişmiş model testleri başlatılıyor...")
        results = []
        
        for model_type, description, use_advanced in test_scenarios:
            print(f"\n🧪 {description} başlatılıyor...")
            try:
                if use_advanced:
                    # Gelişmiş sistem testi
                    result = predict_realtime_advanced(
                        'data/Kopya5G_DL.xlsx', 
                        'data/ITU5GHucreBilgileri.xlsx', 
                        model_type
                    )
                else:
                    # Standart sistem testi
                    result = predict_realtime(
                        'data/Kopya5G_DL.xlsx', 
                        'data/ITU5GHucreBilgileri.xlsx', 
                        model_type
                    )
                
                if result is not None:
                    print(f"✅ {description} başarılı!")
                    results.append((model_type, "Başarılı", description))
                else:
                    print(f"❌ {description} başarısız!")
                    results.append((model_type, "Başarısız", description))
                    
            except Exception as e:
                print(f"❌ {description} hatası: {e}")
                results.append((model_type, f"Hata: {str(e)[:50]}...", description))
            
            time.sleep(1)  # Kısa bekleme
        
        # Test özeti
        print("\n" + "="*60)
        print("📋 TEST ÖZETİ")
        print("="*60)
        for model_type, status, description in results:
            status_icon = "✅" if status == "Başarılı" else "❌"
            print(f"{status_icon} {description}: {status}")
        
        # En iyi performansı göster
        if performance_analyzer.metrics_history:
            print(f"\n🏆 En son test sonuçları:")
            latest = performance_analyzer.metrics_history[-1]
            print(f"   Model: {latest['model_name']}")
            print(f"   RMSE: {latest['rmse']:.2f}m")
            print(f"   %5m altı: {latest['percent_under_5m']:.1f}%")
    
    # Kullanıcı seçimi
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        run_test_examples()
    elif len(sys.argv) > 1 and sys.argv[1] == '--advanced':
        print("🚀 Gelişmiş model tek test...")
        predict_realtime_advanced('data/Kopya5G_DL.xlsx', 'data/ITU5GHucreBilgileri.xlsx', 'advanced')
    else:
        print("\n🎯 Test seçenekleri:")
        print("python main.py --test      # Tüm modelleri test et")
        print("python main.py --advanced  # Sadece gelişmiş model test et")
        print("\nManuel gelişmiş test için:")
        print("predict_realtime_advanced('data/Kopya5G_DL.xlsx', 'data/ITU5GHucreBilgileri.xlsx', 'advanced')")