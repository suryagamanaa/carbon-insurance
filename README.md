# carbon-insurance
# KafalaCarbon

# Parametric Takaful for Carbon Credit Insurance

Kerangka **takaful parametrik** untuk melindungi pendapatan kredit karbon dari risiko kebakaran hutan di **Kalimantan Timur, Indonesia**.

Repositori ini mereproduksi paper:

> **KafalaCarbon: A Parametric Takaful Framework for Wildfire-Induced Carbon Credit Reversal Risk in Indonesia**
---
Notes:
Repositori ini dikelola oleh co-writer. Sesuai kesepakatan tim penulis, 
hanya abstrak yang dipublikasikan. Full paper, data mentah, dan kode 
lengkap tidak disertakan karena hak publikasi ada pada penulis utama.

## 📄 Abstrak

Indonesia memiliki 125,9 juta hektar hutan tropis dengan kapasitas serapan karbon melebihi 113 miliar ton CO₂, namun pasar karbon sukarela nasional justru stagnan: volume perdagangan di IDXCarbon per Mei 2026 hanya mencapai **0,00064%** dari total pasokan. Akar masalahnya adalah **reversal risk** — pembatalan kredit karbon akibat kebakaran hutan — yang belum memiliki produk asuransi. Penelitian ini memperkenalkan **KafalaCarbon**, sebuah kerangka **takaful parametrik** yang secara otomatis mencairkan klaim ketika kepadatan titik api satelit NASA VIIRS melampaui ambang batas spasial yang telah ditentukan, sehingga menghilangkan *gharar* melalui verifikasi objektif. Dengan menggunakan data kebakaran 13 tahun dari Kalimantan Timur (2012–2024), frekuensi kebakaran dimodelkan dengan distribusi **Negative Binomial** (λ_normal = 1.177; λ_ENSO = 3.110), severity dengan **Lognormal** (E[X] = IDR 666.742; CV = 5,02), dan penetapan harga spasial diturunkan pada grid **0,1° × 0,1°**. Premi bruto berkisar **0,54% – 1,20%** dari pendapatan karbon bersih pada tujuh segmen pasar — jauh di bawah benchmark konvensional — dengan **coverage ratio 83×** untuk titik masuk koperasi 2.500 ha yang direkomendasikan. Produk disusun di bawah kontrak **Wakalah bil Ujrah** dan **Tabarru'** sesuai Fatwa DSN-MUI No. 52 dan 53 (2006).

**Kata Kunci:** Takaful parametrik; asuransi kredit karbon; risiko kebakaran hutan; penetapan harga aktuaria; kepatuhan syariah

---

## 📋 Model Overview

Kerangka KafalaCarbon terdiri dari **enam komponen berurutan**:

| # | Komponen | Metode |
|---|---|---|
| 1 | Pengumpulan data & delineasi wilayah | VIIRS SNPP + GFW + IDXCarbon + NOAA ONI |
| 2 | Konstruksi event kebakaran | DBSCAN spasiotemporal (48 jam, 5 km) |
| 3 | Estimasi area terbakar | Convex hull + buffer 500 m + kalibrasi GFW |
| 4 | Pemodelan severity | Lognormal / Gamma / Weibull (dipilih via AIC) |
| 5 | Pemodelan frekuensi | Negative Binomial dual-regime (ENSO vs non-ENSO) |
| 6 | Penetapan harga berbasis grid | Grid 0,1° × 0,1° + eksponen diversifikasi δ |

### Parameter Kunci

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| λ_normal | 1.177,2 events/yr | Frekuensi non-ENSO |
| λ_ENSO | 3.110,2 events/yr | Frekuensi ENSO (multiplier 2,64×) |
| E[X] | IDR 666.742 | Expected loss per event |
| CV | 5,02 | Coefficient of variation |
| δ | 0,50 | Spatial diversification exponent |
| θ | 0,35 | Safety loading |
| w | 0,25 | Wakalah fee |
| φ | 0,70 | Basis risk factor |
| η | 0,25 | Net revenue factor |

### Hasil Pricing

| Segmen | Area (ha) | Rate (% net revenue) |
|--------|-----------|----------------------|
| Small Cooperative | 500 | 0,99% |
| Large Cooperative (entry) | 2.500 | **1,20%** |
| Large Corporate | 50.000 | 0,54% |

**Premium range:** 0,54% – 1,20% dari pendapatan karbon bersih
**Coverage ratio:** hingga **83×** di titik masuk 2.500 ha

---

## 📊 Data

Folder `data/` berisi hasil pengolahan dari data mentah:

| File | Deskripsi | Ukuran |
|------|-----------|--------|
| `hotspots_kaltim.parquet` | Hotspot VIIRS SNPP Kaltim (2012–2024) | ~5,5 MB |
| `gfw_parameters.json` | Parameter GFW (A_REGIONAL, R_SEQ, dll) | < 1 KB |
| `fire_events.parquet` | Event kebakaran + loss + grid | ~272 KB |
| `metadata.json` | Ringkasan & log fallback | ~1 KB |

**Data mentah tidak disertakan** karena ukurannya besar (~272 MB). Untuk mereplikasi dari awal, unduh:
- **VIIRS SNPP** (375 m): https://firms.modaps.eosdis.nasa.gov/
- **GFW Subnational Data**: https://www.globalforestwatch.org/

### Ringkasan Statistik

| Metrik | Nilai |
|--------|-------|
| Luas hutan Kaltim (A_REGIONAL) | 11.718.040 ha |
| Total hotspot Kaltim | 356.581 |
| Total fire events | 3.297 |
| Total burned area (terkalibrasi) | 524.514 ha |
| R_SEQ | 2,3844 tCO2/ha/yr |
| Calibration factor | 0,0062 |
| Mean loss per event | IDR 8.402.806 |

> **Catatan:** Hasil ini menggunakan pendekatan DBSCAN per temporal window. Paper referensi menggunakan 23.036 events. Perbedaan ini didokumentasikan sebagai analisis sensitivitas.

---

## 🕌 Syariah Compliance

Struktur kontrak **Wakalah bil Ujrah + Tabarru'** sesuai Fatwa DSN-MUI:

- **DSN-MUI Fatwa No. 52/2006** — Wakalah bil Ujrah
- **DSN-MUI Fatwa No. 53/2006** — Tabarru'
- **Haq Mali framework** — pendapatan karbon sebagai objek takaful yang sah

Empat kondisi yang terpenuhi:

1. **Bebas riba** — Dana Tabarru' terpisah dari modal operator
2. **Bebas maysir** — Pendapatan operator dari fee tetap, bukan residual premium
3. **Bebas gharar** — Pemicu satelit objektif, bukan penilaian subjektif
4. **Haq Mali** — Objek asuransi (pendapatan karbon) sah secara fiqh

### Kontribusi Yurisprudensi

Penetapan bahwa pendapatan dari kredit karbon terverifikasi merupakan **Haq Mali** menyelesaikan keberatan intangibilitas yang selama ini menghalangi asuransi Islam meluas ke aset karbon, dan membuka jalur untuk pengembangan produk Takaful di pasar lingkungan intangible lainnya.

---

## 🔧 Instalasi

```bash
git clone https://github.com/suryagamanaa/carbon-insurance.git
cd carbon-insurance
pip install -r requirements.txt
