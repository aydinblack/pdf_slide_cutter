import streamlit as st
import io
import zipfile
import time

import numpy as np
import fitz  # PyMuPDF
import cv2
from PIL import Image
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
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)

    for img_bytes in images_bytes_list:
        blank_slide_layout = prs.slide_layouts[6]
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

# CSS
st.markdown("""
<style>
h1 { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    font-size: 2.5rem !important; 
    font-weight: 800 !important; 
    margin-bottom: 0.5rem !important; 
}
.stButton button[kind="primary"] { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
    border: none !important; 
    border-radius: 12px !important; 
    font-weight: 700 !important; 
    padding: 0.7rem 2rem !important; 
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.35) !important; 
    transition: all 0.3s ease !important; 
    width: 100% !important; 
}
.stDownloadButton button { 
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important; 
    color: white !important; 
    border: none !important; 
    border-radius: 12px !important; 
    font-weight: 700 !important; 
    padding: 0.7rem 2rem !important; 
    box-shadow: 0 6px 20px rgba(245, 87, 108, 0.35) !important; 
    transition: all 0.3s ease !important; 
    width: 100% !important; 
}
</style>
""", unsafe_allow_html=True)

# Başlık
st.markdown("<h1>✂️ PDF Slide Kesici</h1>", unsafe_allow_html=True)
st.markdown("### 🎯 Kırmızı başlık şeritlerine göre otomatik kesim")
st.markdown("---")

# Dosya yükleme
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

    if st.button("🚀 İşlemeyi Başlat", type="primary"):
        try:
            # PDF işleme
            st.info("📄 PDF işleniyor...")
            pdf_bytes = uploaded.getvalue()
            pages = pdf_to_images(pdf_bytes, dpi=DPI)

            st.info(f"🔍 {len(pages)} sayfa tespit edildi, işleniyor...")

            crops_pngs = []
            total_bands = 0

            progress_bar = st.progress(0)

            for idx, img in enumerate(pages):
                try:
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

                except Exception as e:
                    st.warning(f"⚠️ Sayfa {idx + 1} işlenirken hata: {str(e)}")
                    continue

                progress_bar.progress((idx + 1) / len(pages))

            progress_bar.empty()

            if crops_pngs:
                st.success(f"✅ İşlem tamamlandı! {len(crops_pngs)} slide oluşturuldu.")

                # İstatistikler
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Sayfa", len(pages))
                with col2:
                    st.metric("🎯 Başlık", total_bands)
                with col3:
                    st.metric("✂️ Kesim", len(crops_pngs))

                # ZIP indirme
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for i, data in enumerate(crops_pngs, start=1):
                        zf.writestr(f"slide_{i:04d}.png", data)
                zip_buf.seek(0)

                st.download_button(
                    label=f"📦 ZIP İndir ({len(crops_pngs)} görsel)",
                    data=zip_buf,
                    file_name="slides.zip",
                    mime="application/zip"
                )

                # PowerPoint
                try:
                    st.info("📊 PowerPoint oluşturuluyor...")
                    pptx_buffer = create_powerpoint(crops_pngs)

                    st.download_button(
                        label="📥 PowerPoint İndir (.pptx)",
                        data=pptx_buffer,
                        file_name="sunum.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        type="primary"
                    )
                except Exception as e:
                    st.warning(f"⚠️ PowerPoint oluşturulamadı: {str(e)}")

                # Önizleme
                st.markdown("### 🖼️ Kesilen Görseller")
                for i in range(0, len(crops_pngs), 3):
                    cols = st.columns(3)
                    for j in range(3):
                        idx = i + j
                        if idx < len(crops_pngs):
                            data = crops_pngs[idx]
                            try:
                                im = Image.open(io.BytesIO(data))
                                if im.mode != 'RGB':
                                    im = im.convert('RGB')
                                cols[j].image(im, caption=f"Slide {idx + 1:04d}")
                            except Exception as e:
                                cols[j].error(f"❌ Görsel yüklenemedi: {str(e)}")
            else:
                st.error("❌ Hiç slide oluşturulamadı. PDF'de kırmızı başlık şeritleri bulunamadı.")

        except Exception as e:
            st.error(f"❌ Hata oluştu: {str(e)}")
            st.error("Lütfen PDF dosyanızı kontrol edin ve tekrar deneyin.")

else:
    st.info("👆 PDF dosyanızı yükleyin ve işleme başlatın", icon="ℹ️")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #999; font-size: 0.9rem;'>Made with ❤️ using Streamlit</p>",
            unsafe_allow_html=True)