import io
import zipfile
import os
import tempfile
import pickle
from pathlib import Path
from itertools import count
import time

import numpy as np
import fitz  # PyMuPDF
import cv2
from PIL import Image
import streamlit as st
from pptx import Presentation
from pptx.util import Inches, Pt

# ========================= Sabit Ayarlar =========================
DPI = 240
ROW_RATIO_THRESH = 0.45
CNT_MIN_W_RATIO = 0.65
MIN_BAND_H = 32
MAX_BAND_H_FRAC = 1 / 10
PAD_TOP = 6
PAD_BOTTOM = 12


# ========================= Yardımcılar =========================

def pdf_to_images(file_bytes: bytes, dpi: int = DPI):
    """PDF dosyasını (bytes) sayfa görsellerine çevirir (BGR numpy)."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    imgs = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgs.append(img)
    doc.close()
    return imgs


def red_mask(bgr_img, lower1, upper1, lower2, upper2):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
    m2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
    mask = cv2.bitwise_or(m1, m2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def headers_by_row(mask, ratio_thresh, min_h, max_h, gap_tol):
    """Satır projeksiyonu ile geniş kırmızı şeritleri bul."""
    H, W = mask.shape[:2]
    red_ratio = mask.sum(axis=1) / (255.0 * W)
    win = max(7, H // 200)
    kernel1d = np.ones(win, dtype=np.float32) / win
    smooth = np.convolve(red_ratio, kernel1d, mode='same')

    bands, in_run, run_start, last_y = [], False, 0, -10 ** 9
    for y, val in enumerate(smooth):
        if val >= ratio_thresh:
            if not in_run:
                if bands and (y - bands[-1][1] <= gap_tol):
                    in_run, run_start = True, bands[-1][0]
                    bands.pop()
                else:
                    in_run, run_start = True, y
            last_y = y
        else:
            if in_run:
                in_run = False
                if last_y - run_start + 1 >= min_h:
                    bands.append((run_start, last_y))
    if in_run and (last_y - run_start + 1 >= min_h):
        bands.append((run_start, last_y))

    cleaned = []
    for s, e in bands:
        h = e - s + 1
        if h > max_h:
            mid = (s + e) // 2
            half = max_h // 2
            cleaned.append((max(0, mid - half), min(H - 1, mid + half)))
        else:
            cleaned.append((s, e))
    return cleaned


def headers_by_contours(mask, w_ratio, min_h, max_h):
    """Kontur tabanlı: tam genişliğe yakın yatay kırmızı şeritler."""
    H, W = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w >= w_ratio * W and (min_h <= h <= max_h):
            out.append((y, y + h - 1))
    return out


def merge_bands(bands, gap_tol):
    if not bands:
        return []
    bands = sorted(bands, key=lambda t: t[0])
    merged = []
    cs, ce = bands[0]
    for s, e in bands[1:]:
        if s <= ce + gap_tol:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged


def find_header_bands(img,
                      row_ratio_thresh=ROW_RATIO_THRESH,
                      min_band_h=MIN_BAND_H,
                      max_band_h_frac=MAX_BAND_H_FRAC,
                      gap_frac=1 / 300,
                      cnt_min_w_ratio=CNT_MIN_W_RATIO,
                      cnt_min_h=28,
                      cnt_max_h=260,
                      hsv1_low=(0, 80, 70),
                      hsv1_up=(12, 255, 255),
                      hsv2_low=(170, 80, 70),
                      hsv2_up=(180, 255, 255)):
    """Şeritleri iki yöntemle bul, birleştir."""
    H, W = img.shape[:2]
    gap_tol = max(6, int(H * gap_frac))
    max_band_h = max(int(H * max_band_h_frac), min_band_h + 10)

    mask = red_mask(img, hsv1_low, hsv1_up, hsv2_low, hsv2_up)
    b1 = headers_by_row(mask, row_ratio_thresh, min_band_h, max_band_h, gap_tol)
    b2 = headers_by_contours(mask, cnt_min_w_ratio, cnt_min_h, cnt_max_h)
    bands = merge_bands(b1 + b2, gap_tol)
    return bands, mask


def slice_by_headers(img, header_bands, pad_top=PAD_TOP, pad_bottom=PAD_BOTTOM):
    H, W = img.shape[:2]
    if not header_bands:
        return [(0, 0, W, H)]
    header_bands = sorted(header_bands, key=lambda t: t[0])
    crops = []
    for i in range(len(header_bands)):
        y0 = max(0, header_bands[i][0] - pad_top)
        if i < len(header_bands) - 1:
            y1 = header_bands[i + 1][0] - 1
        else:
            y1 = H
        y1 = min(H, y1 + pad_bottom)
        crops.append((0, y0, W, y1))
    return crops


def bgr_to_pil(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def create_powerpoint(images_bytes_list):
    """Görselleri PowerPoint sunumuna dönüştür (16:9 optimize edilmiş)."""
    prs = Presentation()

    # 16:9 oran ayarla (standart slide boyutu)
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)

    for img_bytes in images_bytes_list:
        blank_slide_layout = prs.slide_layouts[6]  # Blank layout
        slide = prs.slides.add_slide(blank_slide_layout)

        img = Image.open(io.BytesIO(img_bytes))

        img_width, img_height = img.size
        img_ratio = img_width / img_height

        slide_width = prs.slide_width
        slide_height = prs.slide_height
        slide_ratio = slide_width / slide_height

        if img_ratio > slide_ratio:
            pic_width = slide_width
            pic_height = int(slide_width / img_ratio)
        else:
            pic_height = slide_height
            pic_width = int(slide_height * img_ratio)

        left = (slide_width - pic_width) // 2
        top = (slide_height - pic_height) // 2

        img_buffer = io.BytesIO(img_bytes)
        slide.shapes.add_picture(img_buffer, left, top, width=pic_width, height=pic_height)

    pptx_buffer = io.BytesIO()
    prs.save(pptx_buffer)
    pptx_buffer.seek(0)
    return pptx_buffer


# ========================= Arayüz =========================

st.set_page_config(
    page_title="PDF Slide Kesici",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Minimal session state - sadece gerekli olanlar
def init_minimal_state():
    """Minimal session state başlatma"""
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "current_file" not in st.session_state:
        st.session_state.current_file = None


init_minimal_state()

# ---- Modern CSS Stilleri ----
st.markdown(
    """
    <style>
    h1 { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem !important; font-weight: 800 !important; margin-bottom: 0.5rem !important; }
    .upload-card { background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border: 2px dashed #667eea; border-radius: 20px; padding: 3rem 2rem; text-align: center; transition: all 0.3s ease; margin: 2rem 0; }
    .upload-card:hover { border-color: #764ba2; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2); transform: translateY(-2px); }
    .stDownloadButton button { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important; color: white !important; border: none !important; border-radius: 12px !important; font-weight: 700 !important; padding: 0.7rem 2rem !important; box-shadow: 0 6px 20px rgba(245, 87, 108, 0.35) !important; transition: all 0.3s ease !important; width: 100% !important; }
    .stDownloadButton button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(245, 87, 108, 0.45) !important; }
    .stButton button[kind="secondary"] { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important; color: white !important; border: none !important; border-radius: 12px !important; font-weight: 700 !important; padding: 0.7rem 2rem !important; box-shadow: 0 6px 20px rgba(79, 172, 254, 0.35) !important; transition: all 0.3s ease !important; width: 100% !important; }
    .stButton button[kind="secondary"]:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(79, 172, 254, 0.45) !important; }
    .stButton button[kind="primary"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; border: none !important; border-radius: 12px !important; font-weight: 700 !important; padding: 0.7rem 2rem !important; box-shadow: 0 6px 20px rgba(102, 126, 234, 0.35) !important; transition: all 0.3s ease !important; width: 100% !important; }
    .stButton button[kind="primary"]:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.45) !important; }
    .stat-card { background: white; border-radius: 16px; padding: 1.5rem; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08); text-align: center; border-left: 4px solid; transition: all 0.3s ease; }
    .stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12); }
    .stat-number { font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0; }
    .stat-label { color: #666; font-size: 0.95rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    [data-testid="stImage"] { border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); transition: all 0.3s ease; }
    [data-testid="stImage"]:hover { transform: scale(1.02); box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); }
    .stAlert { border-radius: 12px; border-left: 4px solid #667eea; }
    .success-message { background: linear-gradient(135deg, #84fab015 0%, #8fd3f415 100%); border: 2px solid #84fab0; border-radius: 16px; padding: 1.5rem; margin: 2rem 0; text-align: center; }
    [data-testid="stFileUploader"] { border-radius: 16px; }
    [data-testid="stFileUploader"] > div { border-radius: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Başlık ----
st.markdown("<h1>✂️ PDF Slide Kesici</h1>", unsafe_allow_html=True)
st.markdown("### 🎯 Kırmızı başlık şeritlerine göre otomatik kesim")
st.markdown("---")

# ---- Dosya Yükleme Bölümü ----
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 📁 PDF Dosyanızı Yükleyin")
    uploaded = st.file_uploader(
        "PDF dosyasını sürükleyin veya seçin",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

    if uploaded is not None:
        file_size = len(uploaded.getvalue()) / (1024 * 1024)
        st.success(f"✅ **{uploaded.name}** yüklendi ({file_size:.1f} MB)")

        st.markdown("<br>", unsafe_allow_html=True)

        # İşle butonu - tam genişlik
        if st.button("🚀 İşlemeyi Başlat", type="primary", key="process_btn"):
            st.session_state.processing = True
            st.session_state.current_file = uploaded.name

            try:
                with st.spinner("📄 PDF işleniyor..."):
                    pdf_bytes = uploaded.getvalue()
                    pages = pdf_to_images(pdf_bytes, dpi=DPI)

                    crops_pngs = []
                    total_bands = 0

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, img in enumerate(pages):
                        status_text.text(f"🔍 Sayfa {idx + 1}/{len(pages)} işleniyor...")
                        bands, _ = find_header_bands(img)
                        total_bands += len(bands)
                        boxes = slice_by_headers(img, bands)
                        for box in boxes:
                            x0, y0, x1, y1 = box
                            crop = img[y0:y1, x0:x1]
                            pil_im = bgr_to_pil(crop)
                            b = io.BytesIO()
                            pil_im.save(b, format="PNG")
                            crops_pngs.append(b.getvalue())

                        progress_bar.progress((idx + 1) / len(pages))

                    # Geçici dosyalara kaydet (session state yerine)
                    import tempfile
                    import pickle

                    # Geçici dosya oluştur
                    temp_dir = tempfile.mkdtemp()
                    results_file = os.path.join(temp_dir, "results.pkl")

                    results = {
                        'crops_pngs': crops_pngs,
                        'pages_count': len(pages),
                        'bands_count': total_bands,
                        'last_count': len(crops_pngs)
                    }

                    with open(results_file, 'wb') as f:
                        pickle.dump(results, f)

                    st.session_state.results_file = results_file
                    st.session_state.processing = False

                st.success(f"✅ İşlem tamamlandı! {len(crops_pngs)} slide oluşturuldu.")
                st.rerun()

            except Exception as e:
                st.session_state.processing = False
                st.error(f"❌ Hata oluştu: {str(e)}")
                st.error("Lütfen PDF dosyanızı kontrol edin ve tekrar deneyin.")

st.markdown("---")

# ---- İstatistikler ve İndirme ----
# Sonuçları geçici dosyadan yükle
results = None
if hasattr(st.session_state, 'results_file') and st.session_state.results_file:
    try:
        with open(st.session_state.results_file, 'rb') as f:
            results = pickle.load(f)
    except:
        results = None

if results and results['crops_pngs']:
    st.markdown("### 📊 İşlem Sonuçları")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card" style="border-left-color: #667eea;">
                <div class="stat-label">📄 Sayfa</div>
                <div class="stat-number" style="color: #667eea;">{results['pages_count']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card" style="border-left-color: #f5576c;">
                <div class="stat-label">🎯 Başlık</div>
                <div class="stat-number" style="color: #f5576c;">{results['bands_count']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="stat-card" style="border-left-color: #4facfe;">
                <div class="stat-label">✂️ Kesim</div>
                <div class="stat-number" style="color: #4facfe;">{results['last_count']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        # ZIP oluşturma - sadece indirme anında
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, data in enumerate(results['crops_pngs'], start=1):
                zf.writestr(f"slide_{i:04d}.png", data)
        zip_buf.seek(0)

        st.download_button(
            label=f"📦 ZIP İndir ({results['last_count']} görsel)",
            data=zip_buf,
            file_name="slides.zip",
            mime="application/zip",
            key="download_zip_btn"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # PowerPoint oluşturma bölümü
    col_ppt1, col_ppt2, col_ppt3 = st.columns([1, 2, 1])

    with col_ppt2:
        if st.button("🎯 Slide'lara Aktar (PowerPoint Oluştur)", type="secondary", key="create_ppt_btn"):
            with st.spinner("📊 PowerPoint sunumu oluşturuluyor..."):
                pptx_buffer = create_powerpoint(results['crops_pngs'])
                st.download_button(
                    label="📥 PowerPoint'i İndir (.pptx)",
                    data=pptx_buffer,
                    file_name="sunum.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    type="primary",
                    key="download_ppt_btn"
                )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # Önizleme bölümü
    st.markdown("### 🖼️ Kesilen Görseller")
    st.info("ℹ️ Yeni bir PDF yüklediğinizde önizleme otomatik silinir", icon="ℹ️")
    st.caption("💡 Görsellerin üzerine gelerek büyütebilirsiniz")

    # Görselleri göster - 3 sütunlu grid
    for i in range(0, len(results['crops_pngs']), 3):
        cols = st.columns(3)
        for j in range(3):
            idx = i + j
            if idx < len(results['crops_pngs']):
                data = results['crops_pngs'][idx]
                try:
                    im = Image.open(io.BytesIO(data))
                    # RGB'ye çevir
                    if im.mode != 'RGB':
                        im = im.convert('RGB')
                    # Streamlit'e göster
                    cols[j].image(im, caption=f"Slide {idx + 1:04d}")
                except Exception as e:
                    cols[j].error(f"❌ Görsel yüklenemedi: {str(e)}")

else:
    st.info("👆 PDF dosyanızı yükleyin ve işleme başlatın", icon="ℹ️")

    with st.expander("❓ Nasıl Çalışır?"):
        st.markdown(
            """
        **Bu araç neler yapar?**

        1. 📤 PDF dosyanızı yüklersiniz
        2. 🔍 Sistem kırmızı başlık şeritlerini otomatik tespit eder
        3. ✂️ Her başlık arasını ayrı görsel olarak keser
        4. 📦 Tüm görselleri ZIP dosyası olarak indirebilirsiniz
        5. 🎯 PowerPoint sunumu oluşturabilirsiniz

        **Önemli notlar:**
        - Yeni bir PDF yüklediğinizde önceki veriler otomatik silinir
        - Sekmeyi kapatıp açtığınızda önizleme kaybolur
        - Her işlem için tekrar PDF yüklemeniz gerekir
        """
        )

    with st.expander("⚙️ Özellikler"):
        st.markdown(
            """
        - ✨ Otomatik kırmızı renk algılama
        - 🎯 Akıllı şerit birleştirme
        - 📏 Otomatik padding ayarı
        - 🖼️ Yüksek kaliteli çıktı (DPI: 240)
        - 📦 Toplu ZIP indirme desteği
        - 🎯 PowerPoint sunumu oluşturma (16:9)
        - 📐 Otomatik slide optimizasyonu
        - 👁️ Canlı önizleme
        - 🔄 Otomatik veri temizleme (yeni PDF yüklenince)
        - 🧹 Sekme kapanınca otomatik temizlik
        """
        )

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #999; font-size: 0.9rem;'>Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True,
)
