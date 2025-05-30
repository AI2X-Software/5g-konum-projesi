#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TEKNOFEST 2025 5G Konumlandırma Projesi
Kampüs Test Verileri Analiz ve Görselleştirme

Bu script, tüm test verilerini analiz edip kampüs haritası üzerinde görselleştirir.
"""

import pandas as pd
import numpy as np
from kampus_harita import KampusHaritasi
import folium
from folium.plugins import MarkerCluster, HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

class KampusAnalizi:
    """Kampüs test verilerini analiz eder"""
    
    def __init__(self, test_dosyasi, cellinfo_dosyasi):
        self.test_dosyasi = test_dosyasi
        self.cellinfo_dosyasi = cellinfo_dosyasi
        self.test_data = None
        self.cellinfo = None
        self._veri_yukle()
    
    def _veri_yukle(self):
        """Test verilerini yükler"""
        try:
            # Test verilerini yükle
            self.test_data = pd.read_excel(self.test_dosyasi, sheet_name='Series Formatted Data')
            self.cellinfo = pd.read_excel(self.cellinfo_dosyasi, sheet_name='Hücre tablosu')
            
            # Sadece geçerli konum verilerine sahip olanları al
            self.test_data = self.test_data[
                self.test_data['Latitude'].notna() & 
                self.test_data['Longitude'].notna()
            ].copy()
            
            print(f"✅ Test verisi yüklendi: {len(self.test_data)} geçerli konum noktası")
            print(f"✅ Baz istasyonu bilgisi yüklendi: {len(self.cellinfo)} BS")
            
        except Exception as e:
            print(f"❌ Veri yükleme hatası: {e}")
    
    def sinyal_analizi(self):
        """Sinyal kalitesi analizini yapar"""
        print("\n📊 Sinyal Kalitesi Analizi")
        print("=" * 50)
        
        # RSRP istatistikleri
        rsrp_stats = self.test_data['NR_UE_RSRP_0'].describe()
        print("RSRP İstatistikleri:")
        print(f"  Ortalama: {rsrp_stats['mean']:.2f} dBm")
        print(f"  Medyan: {rsrp_stats['50%']:.2f} dBm") 
        print(f"  Min: {rsrp_stats['min']:.2f} dBm")
        print(f"  Max: {rsrp_stats['max']:.2f} dBm")
        
        # Sinyal kalitesi kategorileri
        def sinyal_kategorisi(rsrp):
            if pd.isna(rsrp):
                return 'Bilinmiyor'
            elif rsrp > -80:
                return 'Mükemmel'
            elif rsrp > -90:
                return 'İyi'
            elif rsrp > -100:
                return 'Orta'
            elif rsrp > -110:
                return 'Zayıf'
            else:
                return 'Çok Zayıf'
        
        self.test_data['sinyal_kalitesi'] = self.test_data['NR_UE_RSRP_0'].apply(sinyal_kategorisi)
        
        # Sinyal kalitesi dağılımı
        print("\nSinyal Kalitesi Dağılımı:")
        for kategori, count in self.test_data['sinyal_kalitesi'].value_counts().items():
            print(f"  {kategori}: {count} nokta ({count/len(self.test_data)*100:.1f}%)")
        
        return rsrp_stats
    
    def konum_analizi(self):
        """Konum dağılımı analizini yapar"""
        print("\n📍 Konum Dağılımı Analizi")
        print("=" * 50)
        
        # Konum sınırları
        lat_min, lat_max = self.test_data['Latitude'].min(), self.test_data['Latitude'].max()
        lon_min, lon_max = self.test_data['Longitude'].min(), self.test_data['Longitude'].max()
        
        print(f"Enlem aralığı: {lat_min:.6f} - {lat_max:.6f}")
        print(f"Boylam aralığı: {lon_min:.6f} - {lon_max:.6f}")
        
        # Kampüs merkezi ile mesafeler
        kampus_merkezi = (41.1043, 29.0212)
        from geopy.distance import geodesic
        
        mesafeler = []
        for idx, row in self.test_data.iterrows():
            mesafe = geodesic(kampus_merkezi, (row['Latitude'], row['Longitude'])).meters
            mesafeler.append(mesafe)
        
        self.test_data['kampus_mesafe'] = mesafeler
        
        print(f"Kampüs merkezine ortalama mesafe: {np.mean(mesafeler):.1f} m")
        print(f"En yakın nokta: {np.min(mesafeler):.1f} m")
        print(f"En uzak nokta: {np.max(mesafeler):.1f} m")
        
        return mesafeler
    
    def heatmap_olustur(self):
        """Sinyal gücü heat map'i oluşturur"""
        print("\n🗺️ Heat Map oluşturuluyor...")
        
        # Kampüs harita nesnesi oluştur
        kampus = KampusHaritasi()
        harita = kampus.harita_olustur()
        
        # Heat map verisi hazırla
        heat_data = []
        for idx, row in self.test_data.iterrows():
            if pd.notna(row['NR_UE_RSRP_0']):
                # RSRP değerini 0-1 aralığına normalize et (-140 dBm ile -40 dBm arası)
                rsrp_norm = (row['NR_UE_RSRP_0'] + 140) / 100
                rsrp_norm = max(0, min(1, rsrp_norm))  # 0-1 aralığında tut
                
                heat_data.append([row['Latitude'], row['Longitude'], rsrp_norm])
        
        # Heat map katmanını ekle
        HeatMap(
            heat_data,
            min_opacity=0.2,
            max_zoom=18,
            radius=15,
            blur=15,
            gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'orange', 1.0: 'red'}
        ).add_to(harita)
        
        # Haritayı kaydet
        heat_file = "outputs/sinyal_gucü_heatmap.html"
        harita.save(heat_file)
        print(f"✅ Heat map kaydedildi: {heat_file}")
        
        return harita
    
    def pci_analizi(self):
        """PCI (Physical Cell ID) analizini yapar"""
        print("\n📡 PCI Analizi")
        print("=" * 50)
        
        # En sık kullanılan PCI'lar
        pci_counts = self.test_data['NR_UE_PCI_0'].value_counts().head(10)
        print("En sık kullanılan PCI'lar:")
        for pci, count in pci_counts.items():
            print(f"  PCI {pci}: {count} ölçüm ({count/len(self.test_data)*100:.1f}%)")
        
        # PCI başına ortalama sinyal gücü
        pci_rsrp = self.test_data.groupby('NR_UE_PCI_0')['NR_UE_RSRP_0'].agg(['mean', 'count', 'std'])
        pci_rsrp = pci_rsrp.sort_values('mean', ascending=False).head(10)
        
        print("\nEn iyi sinyal gücüne sahip PCI'lar:")
        for pci, row in pci_rsrp.iterrows():
            print(f"  PCI {pci}: {row['mean']:.1f} dBm (σ={row['std']:.1f}, n={row['count']})")
        
        return pci_counts, pci_rsrp
    
    def cluster_haritasi_olustur(self):
        """Test noktalarını sinyal gücüne göre gruplandırarak gösterir"""
        print("\n🗺️ Cluster haritası oluşturuluyor...")
        
        # Kampüs harita nesnesi oluştur
        kampus = KampusHaritasi()
        harita = kampus.harita_olustur()
        
        # Baz istasyonlarını ekle
        kampus.baz_istasyonlari_ekle(harita, self.cellinfo_dosyasi)
        
        # Sinyal kalitesine göre farklı renkli cluster'lar
        sinyal_gruplari = {
            'Mükemmel': {'color': 'green', 'data': []},
            'İyi': {'color': 'lightgreen', 'data': []},
            'Orta': {'color': 'orange', 'data': []},
            'Zayıf': {'color': 'red', 'data': []},
            'Çok Zayıf': {'color': 'darkred', 'data': []}
        }
        
        # Test verilerini gruplara ayır
        for idx, row in self.test_data.iterrows():
            kategori = row['sinyal_kalitesi']
            if kategori in sinyal_gruplari:
                sinyal_gruplari[kategori]['data'].append(row)
        
        # Her grup için ayrı cluster oluştur
        for kategori, grup in sinyal_gruplari.items():
            if grup['data']:
                feature_group = folium.FeatureGroup(name=f'Sinyal: {kategori} ({len(grup["data"])} nokta)')
                cluster = MarkerCluster(name=f'{kategori} Cluster').add_to(feature_group)
                
                for row in grup['data']:
                    popup_text = f"""
                    <b>Test Noktası</b><br>
                    <b>Sinyal Kalitesi:</b> {kategori}<br>
                    <b>RSRP:</b> {row['NR_UE_RSRP_0']:.1f} dBm<br>
                    <b>RSRQ:</b> {row['NR_UE_RSRQ_0']:.1f} dB<br>
                    <b>SINR:</b> {row['NR_UE_SINR_0']:.1f} dB<br>
                    <b>PCI:</b> {row['NR_UE_PCI_0']}<br>
                    <b>Konum:</b> {row['Latitude']:.6f}, {row['Longitude']:.6f}
                    """
                    
                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=folium.Popup(popup_text, max_width=300),
                        icon=folium.Icon(color=grup['color'], icon='signal', prefix='fa')
                    ).add_to(cluster)
                
                feature_group.add_to(harita)
        
        # Katman kontrolü ekle
        folium.LayerControl().add_to(harita)
        
        # Haritayı kaydet
        cluster_file = "outputs/test_noktalari_cluster.html"
        harita.save(cluster_file)
        print(f"✅ Cluster haritası kaydedildi: {cluster_file}")
        
        return harita
    
    def istatistik_grafikleri_olustur(self):
        """İstatistiksel grafikler oluşturur"""
        print("\n📈 İstatistik grafikleri oluşturuluyor...")
        
        Path("outputs/grafikler").mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. RSRP histogram
        plt.figure(figsize=(10, 6))
        plt.hist(self.test_data['NR_UE_RSRP_0'].dropna(), bins=50, alpha=0.7, color='blue')
        plt.xlabel('RSRP (dBm)')
        plt.ylabel('Frekans')
        plt.title('RSRP Dağılımı')
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs/grafikler/rsrp_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Sinyal kalitesi pie chart
        plt.figure(figsize=(8, 8))
        sinyal_counts = self.test_data['sinyal_kalitesi'].value_counts()
        colors = ['green', 'lightgreen', 'orange', 'red', 'darkred']
        plt.pie(sinyal_counts.values, labels=sinyal_counts.index, autopct='%1.1f%%', 
                colors=colors[:len(sinyal_counts)])
        plt.title('Sinyal Kalitesi Dağılımı')
        plt.savefig('outputs/grafikler/sinyal_kalitesi_pie.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. PCI kullanım dağılımı
        plt.figure(figsize=(12, 6))
        pci_counts = self.test_data['NR_UE_PCI_0'].value_counts().head(15)
        plt.bar(range(len(pci_counts)), pci_counts.values)
        plt.xlabel('PCI')
        plt.ylabel('Ölçüm Sayısı')
        plt.title('En Sık Kullanılan PCI\'lar')
        plt.xticks(range(len(pci_counts)), pci_counts.index, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs/grafikler/pci_dagilimi.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. RSRP vs SINR scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.test_data['NR_UE_RSRP_0'], self.test_data['NR_UE_SINR_0'], 
                   alpha=0.5, s=1)
        plt.xlabel('RSRP (dBm)')
        plt.ylabel('SINR (dB)')
        plt.title('RSRP vs SINR İlişkisi')
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs/grafikler/rsrp_vs_sinr.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Grafikler kaydedildi: outputs/grafikler/")
    
    def tam_analiz_yap(self):
        """Kapsamlı analiz yapar"""
        print("🔍 Kampüs Test Verileri Kapsamlı Analizi")
        print("=" * 60)
        
        # Analizleri yap
        self.sinyal_analizi()
        self.konum_analizi()
        self.pci_analizi()
        
        # Görselleştirmeleri oluştur
        self.heatmap_olustur()
        self.cluster_haritasi_olustur()
        self.istatistik_grafikleri_olustur()
        
        # Özet rapor
        print("\n📋 ÖZET RAPOR")
        print("=" * 50)
        print(f"📊 Toplam test noktası: {len(self.test_data):,}")
        print(f"📡 Kullanılan PCI sayısı: {self.test_data['NR_UE_PCI_0'].nunique()}")
        print(f"🗺️ Coğrafi alan: ~{(self.test_data['kampus_mesafe'].max() - self.test_data['kampus_mesafe'].min()):.0f}m çap")
        print(f"📶 Ortalama RSRP: {self.test_data['NR_UE_RSRP_0'].mean():.1f} dBm")
        print(f"🎯 İyi sinyal oranı: {(self.test_data['sinyal_kalitesi'].isin(['Mükemmel', 'İyi']).sum() / len(self.test_data) * 100):.1f}%")
        
        print("\n✅ Analiz tamamlandı! Çıktılar 'outputs/' klasöründe.")

def main():
    """Ana fonksiyon"""
    analiz = KampusAnalizi(
        test_dosyasi="data/Kopya5G_DL.xlsx",
        cellinfo_dosyasi="data/ITU5GHucreBilgileri.xlsx"
    )
    
    analiz.tam_analiz_yap()

if __name__ == "__main__":
    main() 