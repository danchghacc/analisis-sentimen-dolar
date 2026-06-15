# 📊 Analisis Sentimen: Dolar AS Tembus Rp18.000
**Perbandingan Narasi Media Massa vs Opini Publik Instagram**

---

## 📁 File dalam Proyek

| File | Keterangan |
|------|------------|
| `main_pipeline.py` | Kode utama scraping + analisis sentimen |
| `requirements.txt` | Daftar library yang dibutuhkan |
| `.env` | Kredensial (tidak diupload, buat sendiri) |

---

## ⚙️ Cara Menjalankan

### 1. Install library
```bash
pip install -r requirements.txt
```

### 2. Buat file `.env`
Buat file baru bernama `.env` di folder yang sama, isi dengan:
```
IG_USERNAME=username_instagram_kamu
IG_PASSWORD=password_instagram_kamu
HF_TOKEN=token_huggingface_kamu
```

> **Cara dapat HF_TOKEN:** Daftar di [huggingface.co](https://huggingface.co) → Settings → Access Tokens → New Token

### 3. Jalankan
```bash
python main_pipeline.py
```

---

## 📄 Output CSV yang Dihasilkan

| File CSV | Isi |
|----------|-----|
| `sentimen_artikel.csv` | Hasil sentimen artikel Detik.com (1 baris) |
| `sentimen_artikel_per_paragraf.csv` | Sentimen tiap paragraf artikel |
| `artikel_kata_kunci.csv` | Top 10 kata kunci artikel |
| `comments_id_sentiment.csv` | Komentar Instagram + hasil sentimen |
| `ig_kata_kunci.csv` | Top 10 kata kunci komentar Instagram |

---

## 🤖 Model NLP yang Digunakan
- **IndoBERT** — `crypter70/IndoBERT-Sentiment-Analysis`
- Label: `positive` / `negative`
- Artikel dianalisis per paragraf lalu diagregasi menjadi 1 sentimen final

---

## 🔗 Sumber Data
- **Artikel:** Detik.com — *"Dolar AS Tembus Rp18.000, Ini Penjelasan Bos BI"*
- **Komentar:** Instagram (postingan terkait topik dolar Rp18.000)
