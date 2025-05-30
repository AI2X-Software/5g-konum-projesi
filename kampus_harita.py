#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TEKNOFEST 2025 5G Konumlandırma Projesi
İTÜ Kampüs Harita Görselleştirme Modülü

Bu modül, İTÜ kampüs shapefile verilerini kullanarak zengin harita görselleştirmeleri yapar.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import os
import time
from pathlib import Path

class KampusHaritasi:
    """İTÜ kampüs harita verilerini yönetir ve görselleştirir"""
    
    def __init__(self, harita_veri_yolu="data/İTÜ Kampüs Harita Verileri"):
        self.harita_veri_yolu = harita_veri_yolu
        self.harita_katmanlari = {}
        self.kampus_merkezi = [41.1043, 29.0212]  # İTÜ kampüs merkezi koordinatları
        
        # Shapefile dosyalarını yükle
        self._shapefile_yukle()
    
    def _shapefile_yukle(self):
        """Kampüs shapefile verilerini yükler"""
        shapefile_dosyalari = {
            'binalar_itu': 'ITU_3DBINA_EPSG4326.shp',
            'binalar_tcell': 'TCELL_3DBINA_EPSG4326.shp', 
            'yollar': 'ITU_ULASIMAGI_EPSG4326.shp',
            'su_kutlesi': 'ITU_SUKUTLESI_EPSG4326.shp',
            'sinir_duvar': 'ITU_SINIRDUVAR_EPSG4326.shp',
            'bitki_ortus': 'TCELL_3DVEGETATION_EPSG4326.shp'
        }
        
        for katman_adi, dosya_adi in shapefile_dosyalari.items():
            dosya_yolu = os.path.join(self.harita_veri_yolu, dosya_adi)
            try:
                if os.path.exists(dosya_yolu):
                    gdf = gpd.read_file(dosya_yolu)
                    # WGS84 koordinat sistemine dönüştür
                    if gdf.crs != 'EPSG:4326':
                        gdf = gdf.to_crs('EPSG:4326')
                    self.harita_katmanlari[katman_adi] = gdf
                    print(f"✅ {katman_adi} yüklendi: {len(gdf)} öğe")
                else:
                    print(f"⚠️ {dosya_adi} bulunamadı")
            except Exception as e:
                print(f"❌ {katman_adi} yüklenirken hata: {e}")
    
    def harita_olustur(self, zoom_start=16):
        """Temel kampüs haritasını oluşturur"""
        # Folium haritası oluştur
        m = folium.Map(
            location=self.kampus_merkezi,
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Alternatif harita katmanları ekle
        folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)
        
        # Kampüs katmanlarını ekle
        self._kampus_katmanlari_ekle(m)
        
        # Katman kontrolü ekle
        folium.LayerControl().add_to(m)
        
        return m
    
    def _kampus_katmanlari_ekle(self, harita):
        """Kampüs harita katmanlarını ekler"""
        # Binalar (İTÜ)
        if 'binalar_itu' in self.harita_katmanlari:
            self._bina_katmani_ekle(harita, self.harita_katmanlari['binalar_itu'], 
                                   'İTÜ Binaları', '#FF6B6B', 0.7)
        
        # Binalar (T-Cell)
        if 'binalar_tcell' in self.harita_katmanlari:
            self._bina_katmani_ekle(harita, self.harita_katmanlari['binalar_tcell'], 
                                   'T-Cell Binaları', '#4ECDC4', 0.7)
        
        # Yollar
        if 'yollar' in self.harita_katmanlari:
            self._yol_katmani_ekle(harita, self.harita_katmanlari['yollar'])
        
        # Su kütlesi
        if 'su_kutlesi' in self.harita_katmanlari:
            self._su_katmani_ekle(harita, self.harita_katmanlari['su_kutlesi'])
        
        # Sınır duvarları
        if 'sinir_duvar' in self.harita_katmanlari:
            self._sinir_katmani_ekle(harita, self.harita_katmanlari['sinir_duvar'])
        
        # Bitki örtüsü
        if 'bitki_ortus' in self.harita_katmanlari:
            self._bitki_katmani_ekle(harita, self.harita_katmanlari['bitki_ortus'])
    
    def _bina_katmani_ekle(self, harita, gdf, katman_adi, renk, opaklık):
        """Bina katmanını haritaya ekler"""
        feature_group = folium.FeatureGroup(name=katman_adi)
        
        for idx, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                # Polygon koordinatlarını al
                coords = []
                if hasattr(row.geometry.exterior, 'coords'):
                    coords = [[coord[1], coord[0]] for coord in row.geometry.exterior.coords]
                    
                    # Popup bilgisi oluştur
                    popup_text = f"<b>{katman_adi}</b><br>"
                    for col in gdf.columns:
                        if col not in ['geometry'] and pd.notna(row[col]):
                            popup_text += f"{col}: {row[col]}<br>"
                    
                    folium.Polygon(
                        locations=coords,
                        color=renk,
                        fillColor=renk,
                        fillOpacity=opaklık,
                        weight=2,
                        popup=folium.Popup(popup_text, max_width=300)
                    ).add_to(feature_group)
        
        feature_group.add_to(harita)
    
    def _yol_katmani_ekle(self, harita, gdf):
        """Yol katmanını haritaya ekler"""
        feature_group = folium.FeatureGroup(name='Yollar')
        
        for idx, row in gdf.iterrows():
            if row.geometry.geom_type in ['LineString', 'MultiLineString']:
                try:
                    if row.geometry.geom_type == 'LineString':
                        coords = [[coord[1], coord[0]] for coord in row.geometry.coords]
                        folium.PolyLine(
                            locations=coords,
                            color='#2C3E50',
                            weight=3,
                            opacity=0.8
                        ).add_to(feature_group)
                    elif row.geometry.geom_type == 'MultiLineString':
                        for line in row.geometry.geoms:
                            coords = [[coord[1], coord[0]] for coord in line.coords]
                            folium.PolyLine(
                                locations=coords,
                                color='#2C3E50',
                                weight=3,
                                opacity=0.8
                            ).add_to(feature_group)
                except Exception as e:
                    print(f"Yol çiziminde hata: {e}")
        
        feature_group.add_to(harita)
    
    def _su_katmani_ekle(self, harita, gdf):
        """Su kütlesi katmanını haritaya ekler"""
        feature_group = folium.FeatureGroup(name='Su Kütleleri')
        
        for idx, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                try:
                    coords = [[coord[1], coord[0]] for coord in row.geometry.exterior.coords]
                    folium.Polygon(
                        locations=coords,
                        color='#3498DB',
                        fillColor='#3498DB',
                        fillOpacity=0.6,
                        weight=2
                    ).add_to(feature_group)
                except Exception as e:
                    print(f"Su kütlesi çiziminde hata: {e}")
        
        feature_group.add_to(harita)
    
    def _sinir_katmani_ekle(self, harita, gdf):
        """Sınır duvar katmanını haritaya ekler"""
        feature_group = folium.FeatureGroup(name='Sınır Duvarları')
        
        for idx, row in gdf.iterrows():
            if row.geometry.geom_type in ['LineString', 'MultiLineString']:
                try:
                    if row.geometry.geom_type == 'LineString':
                        coords = [[coord[1], coord[0]] for coord in row.geometry.coords]
                        folium.PolyLine(
                            locations=coords,
                            color='#8B4513',
                            weight=4,
                            opacity=0.8
                        ).add_to(feature_group)
                except Exception as e:
                    print(f"Sınır duvar çiziminde hata: {e}")
        
        feature_group.add_to(harita)
    
    def _bitki_katmani_ekle(self, harita, gdf):
        """Bitki örtüsü katmanını haritaya ekler"""
        feature_group = folium.FeatureGroup(name='Bitki Örtüsü')
        
        for idx, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                try:
                    coords = [[coord[1], coord[0]] for coord in row.geometry.exterior.coords]
                    folium.Polygon(
                        locations=coords,
                        color='#27AE60',
                        fillColor='#27AE60',
                        fillOpacity=0.4,
                        weight=1
                    ).add_to(feature_group)
                except Exception as e:
                    print(f"Bitki örtüsü çiziminde hata: {e}")
        
        feature_group.add_to(harita)
    
    def test_verileri_ekle(self, harita, test_veri_dosyasi, cellinfo_dosyasi):
        """Test verilerini haritaya pin olarak ekler"""
        try:
            # Test verilerini yükle
            test_data = pd.read_excel(test_veri_dosyasi, sheet_name='Series Formatted Data')
            cellinfo = pd.read_excel(cellinfo_dosyasi, sheet_name='Hücre tablosu')
            
            print(f"Test verisi yüklendi: {len(test_data)} satır")
            
            # Test verilerini pin olarak ekle
            test_pins = folium.FeatureGroup(name='Test Verileri')
            marker_cluster = MarkerCluster(name='Test Noktaları').add_to(test_pins)
            
            pin_sayisi = 0
            for idx, row in test_data.iterrows():
                if pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')):
                    lat, lon = row['Latitude'], row['Longitude']
                    
                    # Pin bilgilerini oluştur
                    popup_text = self._pin_bilgisi_olustur(row, cellinfo)
                    
                    # Pin rengini sinyale göre belirle
                    pin_rengi = self._sinyal_gucune_gore_renk(row.get('NR_UE_RSRP_0', -120))
                    
                    folium.Marker(
                        location=[lat, lon],
                        popup=folium.Popup(popup_text, max_width=400),
                        tooltip=f"Test Noktası {idx+1}",
                        icon=folium.Icon(color=pin_rengi, icon='signal', prefix='fa')
                    ).add_to(marker_cluster)
                    
                    pin_sayisi += 1
            
            test_pins.add_to(harita)
            print(f"✅ {pin_sayisi} test noktası haritaya eklendi")
            
        except Exception as e:
            print(f"❌ Test verileri eklenirken hata: {e}")
    
    def baz_istasyonlari_ekle(self, harita, cellinfo_dosyasi):
        """Baz istasyonlarını haritaya ekler"""
        try:
            cellinfo = pd.read_excel(cellinfo_dosyasi, sheet_name='Hücre tablosu')
            
            bs_group = folium.FeatureGroup(name='Baz İstasyonları')
            
            for idx, row in cellinfo.iterrows():
                if pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')):
                    lat, lon = row['Latitude'], row['Longitude']
                    pci = row.get('PCI ', 'Bilinmiyor')
                    azimuth = row.get('Azimuth [°]', 'Bilinmiyor')
                    
                    popup_text = f"""
                    <b>Baz İstasyonu</b><br>
                    PCI: {pci}<br>
                    Azimuth: {azimuth}°<br>
                    Konum: {lat:.6f}, {lon:.6f}<br>
                    Yükseklik: {row.get('Height [m]', 'Bilinmiyor')} m<br>
                    Güç: {row.get('Power [W]', 'Bilinmiyor')} W
                    """
                    
                    folium.Marker(
                        location=[lat, lon],
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"BS-PCI:{pci}",
                        icon=folium.Icon(color='green', icon='tower-broadcast', prefix='fa')
                    ).add_to(bs_group)
            
            bs_group.add_to(harita)
            print(f"✅ {len(cellinfo)} baz istasyonu haritaya eklendi")
            
        except Exception as e:
            print(f"❌ Baz istasyonları eklenirken hata: {e}")
    
    def _pin_bilgisi_olustur(self, row, cellinfo):
        """Test noktası için popup bilgisi oluşturur"""
        popup_text = f"""
        <b>Test Noktası</b><br>
        <b>Konum:</b> {row.get('Latitude', 'N/A'):.6f}, {row.get('Longitude', 'N/A'):.6f}<br>
        <b>Zaman:</b> {row.get('Time', 'N/A')}<br>
        <hr>
        <b>Sinyal Bilgileri:</b><br>
        RSRP: {row.get('NR_UE_RSRP_0', 'N/A')} dBm<br>
        RSRQ: {row.get('NR_UE_RSRQ_0', 'N/A')} dB<br>
        SINR: {row.get('NR_UE_SINR_0', 'N/A')} dB<br>
        PCI: {row.get('NR_UE_PCI_0', 'N/A')}<br>
        """
        
        # Timing Advance varsa ekle
        if pd.notna(row.get('NR_UE_Timing_Advance')):
            popup_text += f"Timing Advance: {row.get('NR_UE_Timing_Advance')} μs<br>"
        
        # Throughput varsa ekle
        if pd.notna(row.get('NR_UE_Throughput_PDCP_DL')):
            popup_text += f"DL Throughput: {row.get('NR_UE_Throughput_PDCP_DL'):.2f} kbps<br>"
        
        return popup_text
    
    def _sinyal_gucune_gore_renk(self, rsrp):
        """RSRP değerine göre pin rengi belirler"""
        if pd.isna(rsrp):
            return 'gray'
        elif rsrp > -80:
            return 'green'  # Çok iyi sinyal
        elif rsrp > -90:
            return 'lightgreen'  # İyi sinyal
        elif rsrp > -100:
            return 'orange'  # Orta sinyal
        elif rsrp > -110:
            return 'red'  # Zayıf sinyal
        else:
            return 'darkred'  # Çok zayıf sinyal

def kampus_haritasi_olustur(test_veri_dosyasi=None, cellinfo_dosyasi=None, 
                           cikti_dosyasi="outputs/kampus_haritasi.html"):
    """Kapsamlı kampüs haritası oluşturur ve kaydeder"""
    print("🗺️ Kampüs haritası oluşturuluyor...")
    
    # Kampüs harita nesnesi oluştur
    kampus = KampusHaritasi()
    
    # Temel haritayı oluştur
    harita = kampus.harita_olustur()
    
    # Baz istasyonlarını ekle
    if cellinfo_dosyasi and os.path.exists(cellinfo_dosyasi):
        kampus.baz_istasyonlari_ekle(harita, cellinfo_dosyasi)
    
    # Test verilerini ekle
    if test_veri_dosyasi and os.path.exists(test_veri_dosyasi):
        kampus.test_verileri_ekle(harita, test_veri_dosyasi, cellinfo_dosyasi)
    
    # Haritayı kaydet
    Path("outputs").mkdir(exist_ok=True)
    harita.save(cikti_dosyasi)
    print(f"✅ Kampüs haritası kaydedildi: {cikti_dosyasi}")
    
    return harita

if __name__ == "__main__":
    # Test için
    kampus_haritasi_olustur(
        test_veri_dosyasi="data/Kopya5G_DL.xlsx",
        cellinfo_dosyasi="data/ITU5GHucreBilgileri.xlsx"
    ) 