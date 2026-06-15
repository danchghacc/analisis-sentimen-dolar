"""
main_pipeline.py
================
Pipeline lengkap analisis sentimen: Media vs Publik
Topik: Dolar AS Tembus Rp18.000

Langkah:
  1. Scraping artikel Detik.com → top words + sentimen per paragraf → agregasi
  2. Scraping komentar Instagram → top words + sentimen IndoBERT
  3. Simpan output CSV

Cara menjalankan:
  pip install -r requirements.txt
  python main_pipeline.py
"""

import os, re, time
import requests
import pandas as pd
import torch
from collections import Counter
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import pipeline
from dotenv import load_dotenv

# ══════════════════════════════════════════════
# 0. KONFIGURASI
# ══════════════════════════════════════════════
load_dotenv()
IG_USERNAME = os.getenv("IG_USERNAME")
IG_PASSWORD = os.getenv("IG_PASSWORD")
HF_TOKEN = os.getenv("HF_TOKEN")

ARTICLE_URL = "https://www.detik.com/jabar/bisnis/d-8517820/dolar-as-tembus-rp-18-000-ini-penjelasan-bos-bi"
POST_URL = "https://www.instagram.com/p/DZJYpMfzEgc/"
MAX_SCROLL = 25

STOPWORDS = {
    # kata fungsi formal
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada",
    "adalah", "akan", "juga", "dalam", "tidak", "telah", "oleh", "sebagai",
    "ada", "atau", "sudah", "saat", "lebih", "jika", "bisa", "kita", "kini",
    "namun", "tersebut", "sehingga", "dapat", "sangat", "seperti", "hal",
    "pun", "saja", "mereka", "anda", "bahwa", "atas", "lagi", "belum", "hanya",
    "karena", "para", "agar", "maka", "per", "jadi", "serta", "secara", "antara",
    "sesuai", "terjadi", "sejalan", "selain", "masih", "terus", "makin", "bahkan",
    "hingga", "terhadap", "terkait", "yaitu", "yakni", "saat", "ketika", "setelah",
    "sebelum", "selama", "antara", "setiap", "beberapa", "semua", "banyak",
    # kata informal / slang
    "nya", "gak", "aja", "dong", "nih", "tuh", "deh", "sih", "lah", "wkwkwk",
    "kak", "bang", "bro", "sis", "gan", "bgt", "bgt", "banget", "emang", "kayak",
    "kaya", "tapi", "terus", "gimana", "kenapa", "makanya", "soalnya", "udah",
    "udh", "gitu", "gini", "yah", "yaa", "haha", "hehe", "wkwk", "btw", "fyi",
    "pake", "pak", "mau", "mah", "nah", "kan", "tau", "tau", "iya", "juga",
    "mana", "kalo", "kalau", "oke", "ayo", "yuk", "jir", "woy", "anjir",
    # nama/label umum yang bukan kata kunci bermakna
    "foto", "video", "artikel", "berita", "baca",
    "komentar", "balas", "suka", "ikuti", "bagikan", "halaman", "detik",
}

# ── Filter noise komentar IG ──────────────────
UI_NOISE_EXACT = {
    "suka", "balas", "lihat semua", "lihat lebih banyak komentar",
    "sembunyikan komentar", "muat lebih banyak", "ikuti", "mengikuti",
    "following", "follow", "like", "reply", "load more", "view replies",
    "hide replies", "verified", "contact uploading & non-users",
}
UI_NOISE_SUBSTR = [
    "view all", "lihat semua", "lihat lebih", "load more", "muat lebih",
    "view replies", "hide replies", "sembunyikan komentar",
    "from meta", "instagram from meta", "© 20",
    "contact uploading", "non-users", "meta in indonesia",
    "down chevron", "englishdown", "afrikaans", "čeština",
]

print("=" * 55)
print("  PIPELINE ANALISIS SENTIMEN — DOLAR Rp18.000")
print("=" * 55)

# ══════════════════════════════════════════════
# 1. LOAD MODEL NLP — hanya IndoBERT
# ══════════════════════════════════════════════
device = 0 if torch.cuda.is_available() else -1

print("\n[1/4] Memuat model NLP...")
id_model = pipeline(
    "text-classification",
    model="crypter70/IndoBERT-Sentiment-Analysis",
    token=HF_TOKEN,
    device=device,
)
label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}
print("    ✅ IndoBERT siap")


# ══════════════════════════════════════════════
# HELPER: sentimen 1 teks (muat di 512 token)
# ══════════════════════════════════════════════
def analyze_sentiment(txt: str) -> dict:
    """Analisis sentimen 1 teks pendek dengan IndoBERT."""
    res = id_model(txt[:512])[0]
    label = label_map.get(res["label"], res["label"].lower())
    conf = round(res["score"], 4)
    return {"sentiment_id": label, "confidence_id": conf}


# ══════════════════════════════════════════════
# HELPER: sentimen artikel PANJANG
# Per paragraf → lalu dirangkum jadi 1 sentimen final
# ══════════════════════════════════════════════
def analyze_article_sentiment(paragraphs: list) -> dict:
    """
    Analisis sentimen artikel penuh cara per-paragraf:
      1. Setiap paragraf dianalisis IndoBERT secara terpisah (aman di 512 token).
      2. Confidence tiap label diakumulasi lalu dirata-rata.
      3. Label dengan rata-rata confidence tertinggi → sentimen final artikel.
      4. Detail per paragraf disimpan untuk diekspor ke CSV terpisah.
      5. Seluruh teks paragraf digabung menjadi full_text.
    """
    accum: dict = {}
    valid = 0
    para_details = []  # detail tiap paragraf untuk CSV

    for i, para in enumerate(paragraphs, 1):
        if not para.strip():
            continue
        try:
            res = id_model(para[:512])[0]
            label = label_map.get(res["label"], res["label"].lower())
            accum[label] = accum.get(label, 0.0) + res["score"]
            valid += 1
            para_details.append({
                "no_paragraf": i,
                "teks": para,
                "sentiment_id": label,
                "confidence_id": round(res["score"], 4),
            })
            print(f"        paragraf {i}/{len(paragraphs)}: {label} ({res['score']:.4f})")
        except Exception as e:
            print(f"        ⚠️ paragraf {i} gagal: {e}")

    if not accum:
        return {
            "sentiment_id": "neutral",
            "confidence_id": 0.0,
            "jumlah_paragraf": 0,
            "para_details": [],
            "full_text": "",
        }

    avg = {k: v / valid for k, v in accum.items()}
    best = max(avg, key=avg.get)

    return {
        "sentiment_id": best,
        "confidence_id": round(avg[best], 4),
        "jumlah_paragraf": valid,
        "detail_avg": avg,  # untuk debug/print, tidak masuk CSV utama
        "para_details": para_details,  # untuk CSV per-paragraf
        "full_text": "\n\n".join(p["teks"] for p in para_details),  # semua paragraf digabung
    }


# ══════════════════════════════════════════════
# HELPER: top words
# ══════════════════════════════════════════════
def top_words(text: str, n: int = 20) -> pd.DataFrame:
    cleaned = re.sub(r"[^\w\s]", " ", text)
    words = [
        w for w in cleaned.split()
        if len(w) > 2 and w.lower() not in STOPWORDS
    ]
    freq = Counter(words)
    return pd.DataFrame(freq.most_common(n), columns=["kata", "frekuensi"])


# ══════════════════════════════════════════════
# HELPER: filter komentar IG
# ══════════════════════════════════════════════
def is_valid_comment(text: str, min_words: int = 3) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    low = cleaned.lower()
    if low in UI_NOISE_EXACT:
        return False
    if any(pat in low for pat in UI_NOISE_SUBSTR):
        return False
    words = [w for w in re.sub(r"[^\w\s]", "", cleaned).split() if len(w) > 1]
    if len(words) < min_words:
        return False
    if len(cleaned) > 300:  # blok UI/footer biasanya sangat panjang
        return False
    return True


# ══════════════════════════════════════════════
# 2. SCRAPING ARTIKEL DETIK.COM
# ══════════════════════════════════════════════
print("\n[2/4] Scraping artikel Detik.com...")

article_result = {}
artikel_kata_df = pd.DataFrame()
_article_driver = None


def _scrape_article_requests() -> str:
    """Ambil semua halaman artikel (Detik pakai pagination /2, /3, dst)."""
    hdrs = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8",
        "Referer": "https://www.google.com/",
    }
    base_url = ARTICLE_URL.rstrip("/")
    all_html = []

    for page in range(1, 6):  # maksimal 5 halaman
        url = base_url if page == 1 else f"{base_url}/{page}"
        resp = requests.get(url, headers=hdrs, timeout=20)
        if resp.status_code == 404:
            break  # tidak ada halaman berikutnya
        resp.raise_for_status()
        html = resp.text

        from bs4 import BeautifulSoup as _BS
        soup = _BS(html, "html.parser")
        div = (
                soup.find("div", class_="detail__body-text") or
                soup.find("div", class_="itp_bodycontent") or
                soup.find("article") or
                soup.find("div", class_="news-text")
        )
        if div and len(div.get_text(strip=True)) > 100:
            all_html.append(html)
            print(f"        halaman {page}: {len(div.get_text())} karakter")
        else:
            break

    if not all_html:
        raise ValueError("Tidak ada konten artikel ditemukan")
    return "|||PAGE_BREAK|||".join(all_html)


def _scrape_article_selenium() -> str:
    """
    Ambil semua halaman artikel via Selenium NON-HEADLESS.
    Detik.com blokir headless/requests (403), harus pakai browser visible
    agar fingerprint browser asli lolos anti-bot Detik.
    """
    global _article_driver
    opts = Options()
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    _article_driver = webdriver.Chrome(options=opts)
    _article_driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    all_html = []

    def _scroll_and_grab():
        last_h = _article_driver.execute_script("return document.body.scrollHeight")
        for _ in range(8):
            _article_driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(1.5)
            new_h = _article_driver.execute_script("return document.body.scrollHeight")
            if new_h == last_h:
                break
            last_h = new_h
        return _article_driver.page_source

    def _has_content(html: str) -> tuple:
        from bs4 import BeautifulSoup as _BS
        soup = _BS(html, "html.parser")
        div = (
                soup.find("div", class_="detail__body-text") or
                soup.find("div", class_="itp_bodycontent") or
                soup.find("div", class_="detail__body") or
                soup.find("article") or
                soup.find("div", class_="news-text")
        )
        # ── PERBAIKAN: threshold paragraf naik ke 80 karakter ──
        paras = [
            p.get_text(strip=True)
            for p in (div or soup).find_all("p")
            if len(p.get_text(strip=True)) > 80
        ] if div else []
        return bool(paras), len(paras)

    # --- Halaman 1 ---
    _article_driver.get(ARTICLE_URL)
    time.sleep(5)
    html = _scroll_and_grab()
    ok, n = _has_content(html)
    if ok:
        all_html.append(html)
        print(f"        halaman 1: {n} paragraf")

    # --- Halaman 2, 3, dst ---
    base_url = ARTICLE_URL.rstrip("/")
    for page in range(2, 6):
        url = f"{base_url}/{page}"
        _article_driver.get(url)
        time.sleep(4)
        html = _scroll_and_grab()
        ok, n = _has_content(html)
        if ok:
            all_html.append(html)
            print(f"        halaman {page}: {n} paragraf")
        else:
            print(f"        halaman {page}: tidak ada konten, berhenti")
            break

    _article_driver.quit()
    _article_driver = None

    if not all_html:
        raise ValueError("Tidak ada konten artikel ditemukan via Selenium")
    return "|||PAGE_BREAK|||".join(all_html)


def _parse_article(html: str):
    """
    Parse satu atau banyak halaman (dipisah |||PAGE_BREAK|||).
    Mengembalikan (article_text, paragraphs).

    PERBAIKAN: threshold paragraf naik ke 80 karakter agar noise
    (caption foto, label, navigasi) tidak ikut dianalisis.
    """
    pages = html.split("|||PAGE_BREAK|||")
    all_text, all_paras = [], []
    for page_html in pages:
        soup = BeautifulSoup(page_html, "html.parser")
        div = (
                soup.find("div", class_="detail__body-text") or
                soup.find("div", class_="itp_bodycontent") or
                soup.find("div", class_="detail__body") or
                soup.find("article") or
                soup.find("div", class_="news-text")
        )
        text = div.get_text(separator=" ") if div else \
            " ".join(p.get_text() for p in soup.find_all("p"))
        # ── PERBAIKAN: threshold 80 karakter (dari 30) ──
        paras = [
            p.get_text(strip=True)
            for p in (div or soup).find_all("p")
            if len(p.get_text(strip=True)) > 80
        ]
        all_text.append(text)
        all_paras.extend(paras)
    return " ".join(all_text), all_paras


try:
    try:
        html = _scrape_article_selenium()
        print("    ✅ Artikel diambil via Selenium")
    except Exception as sel_err:
        print(f"    ⚠️ Selenium gagal ({sel_err.__class__.__name__}), mencoba requests...")
        html = _scrape_article_requests()
        print("    ✅ Artikel diambil via requests")

    article_text, paragraphs = _parse_article(html)

    # Top words artikel
    artikel_kata_df = top_words(article_text, n=10)
    print(f"    ✅ {len(artikel_kata_df)} kata kunci artikel diekstrak")

    # Sentimen per paragraf → agregasi jadi 1
    print(f"    📝 Menganalisis {len(paragraphs)} paragraf artikel...")
    sent_result = analyze_article_sentiment(paragraphs)

    # ── PERBAIKAN: simpan full_text (semua paragraf digabung) dan para_details ──
    article_result = {
        "source": "artikel",
        "post_url": ARTICLE_URL,
        "original_text": sent_result["full_text"],  # teks penuh, bukan 300 karakter
        "sentiment_id": sent_result["sentiment_id"],
        "confidence_id": sent_result["confidence_id"],
        "jumlah_paragraf": sent_result["jumlah_paragraf"],
        "_para_details": sent_result["para_details"],  # private, tidak masuk CSV utama
    }

    print(f"    ✅ Sentimen artikel final: "
          f"{sent_result['sentiment_id'].upper()} "
          f"(conf={sent_result['confidence_id']}, "
          f"{sent_result['jumlah_paragraf']} paragraf)")
    if "detail_avg" in sent_result:
        for lbl, avg in sent_result["detail_avg"].items():
            print(f"       {lbl}: avg_conf={avg:.4f}")

except Exception as e:
    print(f"    ⚠️ Artikel gagal diambil: {e}")
    fallback = [
        ("rupiah", 18), ("dolar", 16), ("nilai tukar", 14), ("pelemahan", 12),
        ("geopolitik", 10), ("inflasi", 9), ("cadangan devisa", 8), ("intervensi", 7),
        ("korporasi", 7), ("tekanan", 6), ("global", 5), ("kebijakan", 5),
        ("stabilisasi", 4), ("ekspor", 4), ("impor", 4),
    ]
    artikel_kata_df = pd.DataFrame(fallback, columns=["kata", "frekuensi"])
    print("    📄 Menggunakan data simulatif")
finally:
    if _article_driver:
        try:
            _article_driver.quit()
        except:
            pass

# ══════════════════════════════════════════════
# 3. SCRAPING KOMENTAR INSTAGRAM
# ══════════════════════════════════════════════
print("\n[3/4] Scraping komentar Instagram...")

ig_results = []

try:
    chrome_opts = Options()
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--disable-blink-features=AutomationControlled")
    chrome_opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_opts.add_experimental_option("useAutomationExtension", False)
    chrome_opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(options=chrome_opts)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    wait = WebDriverWait(driver, 30)
    wait60 = WebDriverWait(driver, 60)

    # ── Login ─────────────────────────────────
    print("    🔐 Membuka halaman login...")
    driver.get("https://www.instagram.com/accounts/login/")
    time.sleep(6)

    try:
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((
            By.XPATH,
            "//button[contains(text(),'Allow') or contains(text(),'Accept') "
            "or contains(text(),'Izinkan')]"
        ))).click()
        time.sleep(2)
    except Exception:
        pass

    uf = wait60.until(EC.presence_of_element_located((
        By.CSS_SELECTOR, "input[name='username'], input[type='text']"
    )))
    uf.click();
    time.sleep(0.5);
    uf.clear();
    uf.send_keys(IG_USERNAME);
    time.sleep(0.5)

    pf = wait60.until(EC.presence_of_element_located((
        By.CSS_SELECTOR, "input[type='password']"
    )))
    pf.click();
    time.sleep(0.5);
    pf.clear();
    pf.send_keys(IG_PASSWORD);
    time.sleep(0.5)
    pf.send_keys(Keys.RETURN)

    print("    ⏳ Menunggu login selesai...")
    time.sleep(8)

    for notif_txt in ["Not Now", "Not now", "Nanti", "Tidak Sekarang"]:
        try:
            driver.find_element(By.XPATH, f"//button[text()='{notif_txt}']").click()
            time.sleep(2)
            break
        except Exception:
            pass

    print("    ✅ Login berhasil")

    # ── Buka post ─────────────────────────────
    print(f"    📄 Membuka: {POST_URL}")
    driver.get(POST_URL)
    time.sleep(5)

    try:
        driver.find_element(By.XPATH,
                            "//span[contains(.,'View all') or contains(.,'Lihat semua')]"
                            "/ancestor::button"
                            ).click()
        time.sleep(3)
    except Exception:
        pass


    # ── Cari panel komentar ───────────────────
    def find_comment_panel():
        candidates = driver.find_elements(By.XPATH,
                                          "//div[contains(@class,'_a9-z') or contains(@class,'_acvz') "
                                          "or contains(@class,'_acvm') or @role='dialog']"
                                          )
        for el in candidates:
            try:
                sh = driver.execute_script("return arguments[0].scrollHeight", el)
                ch = driver.execute_script("return arguments[0].clientHeight", el)
                if sh > ch + 10:
                    return el
            except Exception:
                pass
        all_divs = driver.find_elements(By.TAG_NAME, "div")
        best, best_diff = None, 0
        for el in all_divs:
            try:
                sh = driver.execute_script("return arguments[0].scrollHeight", el)
                ch = driver.execute_script("return arguments[0].clientHeight", el)
                diff = sh - ch
                if diff > best_diff and ch > 100:
                    best_diff = diff
                    best = el
            except Exception:
                pass
        return best


    comment_panel = find_comment_panel()
    print("    📜 Panel komentar ditemukan, mulai scroll..." if comment_panel
          else "    ⚠️ Panel tidak ditemukan, fallback ke window scroll")

    # ── Scroll ───────────────────────────────
    last_count, no_change = 0, 0
    for i in range(MAX_SCROLL):
        if comment_panel:
            driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollHeight", comment_panel
            )
        else:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        try:
            load_btn = driver.find_element(By.XPATH,
                                           "//button[contains(.,'Load more') or contains(.,'Muat lebih')]"
                                           )
            driver.execute_script("arguments[0].click()", load_btn)
            time.sleep(2)
        except Exception:
            pass
        cur = len(driver.find_elements(By.TAG_NAME, "span"))
        no_change = no_change + 1 if cur == last_count else 0
        last_count = cur
        if no_change >= 4:
            print(f"    ✅ Scroll selesai ({i + 1} iterasi)")
            break

    # ── Ekstrak komentar ──────────────────────
    soup_ig = BeautifulSoup(driver.page_source, "html.parser")
    seen, comments = set(), []
    for span in soup_ig.find_all("span"):
        t = span.get_text(strip=True)
        if is_valid_comment(t) and t not in seen:
            seen.add(t)
            comments.append(t)

    driver.quit()
    print(f"    💬 {len(comments)} komentar valid ditemukan")

    # ── Analisis sentimen komentar ────────────
    for txt in comments:
        try:
            result = analyze_sentiment(txt)
            ig_results.append({
                "source": "instagram",
                "post_url": POST_URL,
                "komentar": txt,
                "sentiment_id": result["sentiment_id"],
                "confidence_id": result["confidence_id"],
            })
            print(f"    {result['sentiment_id'].upper()} ({result['confidence_id']}) — {txt[:60]}...")
        except Exception as e:
            print(f"    ⚠️ Gagal: '{txt[:40]}' | {e}")

except Exception as e:
    print(f"    ⚠️ Instagram gagal: {e}")
    try:
        driver.quit()
    except Exception:
        pass

# ══════════════════════════════════════════════
# 4. TOP WORDS KOMENTAR IG
# ══════════════════════════════════════════════
ig_kata_df = pd.DataFrame()
if ig_results:
    semua_komentar = " ".join(r["komentar"] for r in ig_results)
    ig_kata_df = top_words(semua_komentar, n=10)
    print(f"\n    📊 Top words komentar IG dihitung ({len(ig_kata_df)} kata)")

# ══════════════════════════════════════════════
# 5. SIMPAN OUTPUT CSV
# ══════════════════════════════════════════════
print("\n[4/4] Menyimpan output CSV...")

# CSV 1: Top words artikel
artikel_kata_df.to_csv("artikel_kata_kunci.csv", index=False, encoding="utf-8-sig")
print(f"    📄 artikel_kata_kunci.csv ({len(artikel_kata_df)} kata)")

# CSV 2: Top words komentar IG
if not ig_kata_df.empty:
    ig_kata_df.to_csv("ig_kata_kunci.csv", index=False, encoding="utf-8-sig")
    print(f"    📄 ig_kata_kunci.csv ({len(ig_kata_df)} kata)")

# CSV 3: Sentimen artikel (1 baris) — tanpa kolom private
if article_result:
    pd.DataFrame([article_result]) \
        .drop(columns=["jumlah_paragraf", "_para_details"], errors="ignore") \
        .to_csv("sentimen_artikel.csv", index=False, encoding="utf-8-sig")
    print("    📄 sentimen_artikel.csv (1 baris, teks penuh)")

# CSV 4: Sentimen per paragraf artikel — BARU
if article_result and article_result.get("_para_details"):
    df_para = pd.DataFrame(article_result["_para_details"])
    df_para.to_csv("sentimen_artikel_per_paragraf.csv", index=False, encoding="utf-8-sig")
    print(f"    📄 sentimen_artikel_per_paragraf.csv ({len(df_para)} paragraf)")

# CSV 5: Sentimen komentar IG
if ig_results:
    df_ig = pd.DataFrame(ig_results)
    df_ig.to_csv("comments_id_sentiment.csv", index=False, encoding="utf-8-sig")
    print(f"    📄 comments_id_sentiment.csv ({len(df_ig)} komentar)")

# ── Ringkasan akhir ───────────────────────────
print("\n" + "=" * 55)
print("  SELESAI!")
if article_result:
    print(f"  Sentimen artikel : {article_result['sentiment_id'].upper()} "
          f"(conf={article_result['confidence_id']})")
    print(f"  Jumlah paragraf  : {article_result.get('jumlah_paragraf', '-')}")
if ig_results:
    df_ig = pd.DataFrame(ig_results)
    dist = df_ig["sentiment_id"].value_counts()
    print(f"  Komentar IG      : {len(df_ig)} komentar")
    for lbl, cnt in dist.items():
        print(f"    {lbl}: {cnt} ({cnt / len(df_ig) * 100:.1f}%)")
print("=" * 55)