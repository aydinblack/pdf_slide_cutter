import streamlit as st
import io
import zipfile
import numpy as np
import fitz  # PyMuPDF
import cv2
from PIL import Image
from pptx import Presentation
from pptx.util import Inches

# ========================= Sabit Ayarlar =========================
DPI = 240
ROW_RATIO_THRESH = 0.45
CNT_MIN_W_RATIO = 0.65
MIN_BAND_H = 32
MAX_BAND_H_FRAC = 1 / 10
PAD_TOP = 6
PAD_BOTTOM = 12


# ========================= Yardımcı Fonksiyonlar =========================

def pdf_to_images(file_bytes: bytes, dpi: int = DPI):
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


def find_header_bands(img):
    H, W = img.shape[:2]
    gap_tol = max(6, int(H / 300))
    max_band_h = max(int(H * MAX_BAND_H_FRAC), MIN_BAND_H + 10)

    mask = red_mask(img, (0, 80, 70), (12, 255, 255), (170, 80, 70), (180, 255, 255))
    b1 = headers_by_row(mask, ROW_RATIO_THRESH, MIN_BAND_H, max_band_h, gap_tol)
    b2 = headers_by_contours(mask, CNT_MIN_W_RATIO, 28, 260)
    return merge_bands(b1 + b2, gap_tol)


def slice_by_headers(img, header_bands):
    H, W = img.shape[:2]
    if not header_bands:
        return [(0, 0, W, H)]
    header_bands = sorted(header_bands, key=lambda t: t[0])
    crops = []
    for i in range(len(header_bands)):
        y0 = max(0, header_bands[i][0] - PAD_TOP)
        if i < len(header_bands) - 1:
            y1 = header_bands[i + 1][0] - 1
        else:
            y1 = H
        y1 = min(H, y1 + PAD_BOTTOM)
        crops.append((0, y0, W, y1))
    return crops


def bgr_to_pil(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def create_powerpoint(images_bytes_list):
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)

    for img_bytes in images_bytes_list:
        blank_slide_layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(blank_slide_layout)
        img = Image.open(io.BytesIO(img_bytes))

        img_width, img_height = img.size
        img_ratio = img_width / img_height
        slide_ratio = prs.slide_width / prs.slide_height

        if img_ratio > slide_ratio:
            pic_width = prs.slide_width
            pic_height = int(prs.slide_width / img_ratio)
        else:
            pic_height = prs.slide_height
            pic_width = int(prs.slide_height * img_ratio)

        left = (prs.slide_width - pic_width) // 2
        top = (prs.slide_height - pic_height) // 2

        img_buffer = io.BytesIO(img_bytes)
        slide.shapes.add_picture(img_buffer, left, top, width=pic_width, height=pic_height)

    pptx_buffer = io.BytesIO()
    prs.save(pptx_buffer)
    pptx_buffer.seek(0)
    return pptx_buffer


# ========================= Streamlit Config =========================

st.set_page_config(
    page_title="PDF Slide Kesici",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Session State Başlatma
if "crops" not in st.session_state:
    st.session_state.crops = []
if "stats" not in st.session_state:
    st.session_state.stats = {"pages": 0, "bands": 0, "total": 0}
if "pptx_data" not in st.session_state:
    st.session_state.pptx_data = None
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# CSS
st.markdown("""
<style>
h1 { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    font-size: 2.5rem !important; 
    font-weight: 800 !important; 
}
.stButton button[kind="primary"] { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
    color: white !important;
    border: none !important; 
    border-radius: 12px !important; 
    font-weight: 700 !important; 
    padding: 0.7rem 2rem !important; 
}
.stButton button[kind="secondary"] { 
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important; 
    color: white !important;
    border: none !important; 
    border-radius: 12px !important; 
    font-weight: 700 !important; 
}
.stDownloadButton button { 
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important; 
    color: white !important; 
    border: none !important; 
    border-radius: 12px !important; 
    font-weight: 700 !important; 
}
</style>
""", unsafe_allow_html=True)

# Başlık
st.markdown("<h1>✂️ PDF Slide Kesici</h1>", unsafe_allow_html=True)
st.markdown("### 🎯 Kırmızı başlık şeritlerine göre otomatik kesim")
st.markdown("---")

# Dosya Yükleme
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 📁 PDF Dosyanızı Yükleyin")
    uploaded = st.file_uploader(
        "PDF dosyasını seçin",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

    if uploaded:
        file_size = len(uploaded.getvalue()) / (1024 * 1024)
        st.success(f"✅ **{uploaded.name}** ({file_size:.1f} MB)")

        if st.button("🚀 İşlemeyi Başlat", type="primary", key="process_btn"):
            try:
                with st.spinner("📄 PDF işleniyor..."):
                    pdf_bytes = uploaded.getvalue()
                    pages = pdf_to_images(pdf_bytes, dpi=DPI)

                    crops_list = []
                    total_bands = 0

                    progress = st.progress(0)

                    for idx, img in enumerate(pages):
                        try:
                            bands = find_header_bands(img)
                            total_bands += len(bands)
                            boxes = slice_by_headers(img, bands)

                            for box in boxes:
                                x0, y0, x1, y1 = box
                                crop = img[y0:y1, x0:x1]
                                pil_im = bgr_to_pil(crop)
                                buf = io.BytesIO()
                                pil_im.save(buf, format="PNG")
                                crops_list.append(buf.getvalue())
                        except:
                            pass

                        progress.progress((idx + 1) / len(pages))

                    progress.empty()

                    # Session state'e kaydet
                    st.session_state.crops = crops_list
                    st.session_state.stats = {
                        "pages": len(pages),
                        "bands": total_bands,
                        "total": len(crops_list)
                    }
                    st.session_state.show_results = True
                    st.session_state.pptx_data = None

                st.success(f"✅ {len(crops_list)} slide oluşturuldu!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Hata: {str(e)}")

st.markdown("---")

# Sonuçlar
if st.session_state.show_results and st.session_state.crops:
    stats = st.session_state.stats

    st.markdown("### 📊 İşlem Sonuçları")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📄 Sayfa", stats["pages"])
    with col2:
        st.metric("🎯 Başlık", stats["bands"])
    with col3:
        st.metric("✂️ Kesim", stats["total"])

    st.markdown("<br>", unsafe_allow_html=True)

    # İndirme Butonları
    col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])

    with col_dl1:
        # ZIP oluştur
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, data in enumerate(st.session_state.crops, start=1):
                zf.writestr(f"slide_{i:04d}.png", data)
        zip_buf.seek(0)

        st.download_button(
            label=f"📦 ZIP İndir ({stats['total']} görsel)",
            data=zip_buf,
            file_name="slides.zip",
            mime="application/zip",
            key="zip_btn"
        )

    with col_dl2:
        if st.button("🎯 PowerPoint Oluştur", type="secondary", key="ppt_btn"):
            with st.spinner("📊 PowerPoint hazırlanıyor..."):
                try:
                    pptx_buf = create_powerpoint(st.session_state.crops)
                    st.session_state.pptx_data = pptx_buf.getvalue()
                    st.success("✅ PowerPoint hazır!")
                except Exception as e:
                    st.error(f"❌ {str(e)}")

    with col_dl3:
        if st.session_state.pptx_data:
            st.download_button(
                label="📥 PowerPoint İndir",
                data=st.session_state.pptx_data,
                file_name="sunum.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                type="primary",
                key="ppt_dl_btn"
            )

    st.markdown("---")

    # Önizleme
    st.markdown("### 🖼️ Önizleme")

    # Temizle butonu
    if st.button("🗑️ Önizlemeyi Kapat", key="clear_btn"):
        st.session_state.show_results = False
        st.session_state.crops = []
        st.session_state.pptx_data = None
        st.rerun()

    # Görseller
    for i in range(0, len(st.session_state.crops), 3):
        cols = st.columns(3)
        for j in range(3):
            idx = i + j
            if idx < len(st.session_state.crops):
                try:
                    img = Image.open(io.BytesIO(st.session_state.crops[idx]))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    cols[j].image(img, caption=f"Slide {idx + 1:04d}")
                except:
                    cols[j].error("❌")

else:
    st.info("👆 PDF dosyanızı yükleyin ve işlemeyi başlatın")

    with st.expander("❓ Nasıl Çalışır?"):
        st.markdown("""
        1. 📤 PDF dosyanızı yükleyin
        2. 🚀 İşleme başlatın
        3. 📦 ZIP veya PowerPoint olarak indirin

        **Özellikler:**
        - Otomatik kırmızı başlık algılama
        - 16:9 PowerPoint slaytları
        - Yüksek kalite (240 DPI)
        """)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #999;'>Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)