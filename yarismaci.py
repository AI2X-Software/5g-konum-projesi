#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TEKNOFEST 2025 5G Konumlandırma Yarışması
Gerçek Zamanlı Konum Tahmini

Bu script yarışma sırasında kullanılacak basit komut satırı arayüzüdür.
"""

import argparse
import sys
import os
from pathlib import Path

# main.py'deki fonksiyonları import et
try:
    from main import predict_realtime
    print("✅ Main modül başarıyla yüklendi")
except ImportError as e:
    print(f"❌ Main modül yüklenemedi: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="TEKNOFEST 2025 5G Konumlandırma Yarışması - Gerçek Zamanlı Tahmin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek Kullanım:
  python yarismaci.py data/test_input.xlsx
  python yarismaci.py data/test_input.xlsx --cellinfo=data/ITU5GHucreBilgileri.xlsx --model=hybrid
  python yarismaci.py data/test_input.xlsx --model=dl --output=sonuclar/
        """
    )
    
    parser.add_argument('input_file', 
                       help='Giriş veri dosyası (Excel formatı)')
    
    parser.add_argument('--cellinfo', 
                       default='data/ITU5GHucreBilgileri.xlsx',
                       help='Baz istasyonu bilgileri dosyası (default: data/ITU5GHucreBilgileri.xlsx)')
    
    parser.add_argument('--model', 
                       choices=['dl', 'ul', 'hybrid'],
                       default='dl',
                       help='Kullanılacak model tipi (default: dl)')
    
    parser.add_argument('--output',
                       default='outputs',
                       help='Çıktı dizini (default: outputs)')
    
    args = parser.parse_args()
    
    # Giriş dosyası kontrolü
    if not os.path.exists(args.input_file):
        print(f"❌ HATA: Giriş dosyası bulunamadı: {args.input_file}")
        return 1
    
    # Cell info dosyası kontrolü
    if not os.path.exists(args.cellinfo):
        print(f"❌ HATA: Baz istasyonu bilgi dosyası bulunamadı: {args.cellinfo}")
        return 1
    
    # Çıktı dizini oluştur
    Path(args.output).mkdir(exist_ok=True)
    
    # Başlık yazdır
    print("=" * 60)
    print("  TEKNOFEST 2025 5G Konumlandırma Yarışması - Konum Tahmini")
    print("=" * 60)
    print(f"Giriş dosyası:       {args.input_file}")
    print(f"Baz istasyonu bilgi: {args.cellinfo}")
    print(f"Model tipi:          {args.model}")
    print(f"Çıktı dizini:        {args.output}")
    print("-" * 60)
    
    try:
        # Konum tahmini yap
        predicted_coords = predict_realtime(
            input_file=args.input_file,
            cell_info_file=args.cellinfo, 
            model_type=args.model
        )
        
        if predicted_coords is not None:
            print(f"\n✅ İşlem başarıyla tamamlandı!")
            print(f"   Tahmini Konum: {predicted_coords[0]:.6f}, {predicted_coords[1]:.6f}")
            print(f"   Sonuç dosyaları '{args.output}' dizinine kaydedildi.")
            return 0
        else:
            print(f"\n❌ Konum tahmini başarısız!")
            return 1
            
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 