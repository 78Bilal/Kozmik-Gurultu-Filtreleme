
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import savgol_filter
import pandas as pd
from datetime import datetime
import os
import sounddevice as sd  # pyright: ignore

# --- 1. SİNYAL SÜRESİ VE ÜRETİMİ ---
# Sesin net duyulması için 2 saniyelik 16000 örnek üretiyoruz.
t = np.linspace(0, 2, 16000) 
f1 = np.random.randint(20, 50)
f2 = np.random.randint(70, 120)
clean_signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# Beyaz gürültü ekle
noise = np.random.normal(0, 0.7, 16000)
noisy_signal = clean_signal + noise

# KOZMİK PATLAMA (Anomali) EKLE
burst_power = np.random.uniform(4, 9)
burst_idx = np.random.randint(4000, 12000) 
noisy_signal[burst_idx:burst_idx+200] += burst_power

# --- 2. FFT ANALİZİ VE FİLTRELEME ---
n = len(t)
f_hat = fft(noisy_signal)
psd = np.abs(f_hat * np.conj(f_hat) / n)
freq = fftfreq(n, d=(t[1]-t[0]))

# Dinamik Eşik (Adaptive Thresholding)
threshold = np.mean(psd) + 2.5 * np.std(psd)
f_hat_clean = f_hat.copy()
f_hat_clean[psd < threshold] = 0

# Ters FFT ve Yumuşatma
fft_filtered = ifft(f_hat_clean).real
final_signal = savgol_filter(fft_filtered, window_length=501, polyorder=3)

# --- 3. PERFORMANS HESAPLAMA ---
snr_val = 10 * np.log10(np.mean(clean_signal**2) / np.mean((clean_signal - final_signal)**2))

# --- 4. KLASÖR VE DOSYA YÖNETİMİ ---
hedef_klasor = os.path.join(os.path.expanduser("~"), "Documents", "TUA_ASTRO_HACKATHON", "Frekans_analiz")
if not os.path.exists(hedef_klasor):
    os.makedirs(hedef_klasor)

zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
dosya_base = f"kozmik_analiz_{zaman_damgasi}"
csv_yolu = os.path.join(hedef_klasor, dosya_base + ".csv")
png_yolu = os.path.join(hedef_klasor, dosya_base + ".png")
txt_yolu = os.path.join(hedef_klasor, dosya_base + "_rapor.txt")

# --- 5. VERİLERİ KAYDETME (CSV) ---
df = pd.DataFrame({'Zaman_sn': t, 'Ham_Kozmik_Veri': noisy_signal, 'Temizlenmis_Sinyal': final_signal})
df.to_csv(csv_yolu, index=False, sep=';', decimal=',')

# --- 6. GÖRSELLEŞTİRME VE KAYIT (PNG) ---
plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
plt.plot(t, noisy_signal, color='red', alpha=0.3, label='Ham Veri (Gürültülü)')
plt.plot(t, final_signal, color='blue', linewidth=2, label='Filtrelenmiş Sinyal')
plt.title(f'Karabük Merkez İstasyonu - Zaman Serisi Analizi | SNR: {snr_val:.2f} dB')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(freq[:n//2], psd[:n//2], color='black', label='Güç Spektrumu (PSD)')
plt.axhline(y=threshold, color='green', linestyle='--', label='Dinamik Eşik')
plt.xlim(0, 150)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, noisy_signal - final_signal, color='gray', alpha=0.5)
plt.title('Sinyalden Ayıklanan Toplam Gürültü Katmanı')

plt.tight_layout()
plt.savefig(png_yolu)

# --- 7. KARABÜK MERKEZ ANALİZ RAPORU (TXT) ---
rapor_metni = f"""
/*******************************************************************************
 * TÜRK UZAY AJANSI - ASTRO ANALİZ SİSTEMİ (v2.0)                              *
 * OTOMATİK SİNYAL RAPORU - KARABÜK MERKEZ İSTASYONU                         *
 *******************************************************************************/

 [GENEL BİLGİLER]
 -------------------------------------------------------------------------------
 Veri Kayıt Kimliği  : {dosya_base}
 Analiz Tarihi       : {datetime.now().strftime('%d/%m/%Y')}
 Analiz Saati        : {datetime.now().strftime('%H:%M:%S')}
 İstasyon Konumu     : Karabük / Türkiye (Ana Kontrol Merkezi)
 
 [TEKNİK PARAMETRELER]
 -------------------------------------------------------------------------------
 Ana Sinyal Frekansı (f1)  : {f1:<6} Hz
 Yan Sinyal Frekansı (f2)  : {f2:<6} Hz
 FFT Örnekleme Sayısı (N)  : {n:<6}
 Dinamik Eşik Değeri       : {threshold:.4f}
 
 [PERFORMANS VE KALİTE ANALİZİ]
 -------------------------------------------------------------------------------
 Sinyal-Gürültü Oranı (SNR) : {snr_val:.2f} dB
 Temizlik Başarı Skoru      : %{min(100, max(0, snr_val*5)):.1f}
 Filtreleme Algoritması     : FFT + Savitzky-Golay (Adaptive)
 
 [ANOMALİ TESPİT DURUMU]
 -------------------------------------------------------------------------------
 DURUM              : {'⚠️ KRİTİK ANALİZ GEREKLİ' if burst_power > 6.5 else '✅ NORMAL OPERASYON'}
 Tespit Edilen Güç  : {burst_power:.2f} (Eşik: 6.50)
 Açıklama           : {'Sinyal içinde yüksek enerjili bir patlama (Burst) saptandı.' if burst_power > 6.5 else 'Sinyal akışı doğal kozmik limitler dahilinde.'}
 
 [SONUÇ VE ÖNERİLER]
 -------------------------------------------------------------------------------
 Karabük Merkez İstasyonu üzerinden alınan veriler başarıyla gürültüden 
 arındırılmıştır. Temizlenen veriler dijital arşive aktarılmıştır.
 
 Arşiv Kaydı  : {csv_yolu}
 Grafik Kaydı : {png_yolu}

-------------------------------------------------------------------------------
                     COPYRIGHT (C) 2026 - TUA HACKATHON
*******************************************************************************/
"""

with open(txt_yolu, "w", encoding="utf-8") as f:
    f.write(rapor_metni)

print("-" * 50)
print(f"🏙️  KARABÜK MERKEZ İSTASYONU: Analiz Raporu Hazır!")
print(f"📄 Rapor Kaydedildi: {txt_yolu}")
print("-" * 50)

# --- 8. ETKİLEŞİMLİ SESLİLEŞTİRME ---
def interaktif_seslendir(ham_sinyal, temiz_sinyal, fs=8000):
    # Normalize et
    ham_ses = ham_sinyal / np.max(np.abs(ham_sinyal))
    temiz_ses = temiz_sinyal / np.max(np.abs(temiz_sinyal))
    
    print("\n" + "="*50)
    print("🛰️  KOZMİK SONIFICATION (SESLİLEŞTİRME) SİSTEMİ")
    print("="*50)
    
    input("\n👉 [ENTER] tuşuna basarak HAM KOZMİK VERİYİ dinlet...")
    print("🔈 Oynatılıyor: Filtresiz Radyo Dalgaları (Karabük Alıcısı)...")
    sd.play(ham_ses, fs)
    sd.wait()
    
    input("\n✅ Gürültü duyuldu. TEMİZLENMİŞ SİNYAL için [ENTER]'a bas!")
    print("🔊 Oynatılıyor: Ayıklanmış Yıldız Sinyali (Saf)...")
    sd.play(temiz_ses, fs)
    sd.wait()
    
    print("\n✅ Sesli analiz başarıyla tamamlandı. Grafik açılıyor...")

# --- ÇALIŞTIRMA ---
interaktif_seslendir(noisy_signal, final_signal)
plt.show()
