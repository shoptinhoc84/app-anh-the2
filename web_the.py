import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc
import streamlit.components.v1 as components

# --- BẢO VỆ CHỐNG SẬP KHI THIẾU THƯ VIỆN ---
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_detection as mp_face_detection
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# --- 1. CẤU HÌNH TRANG VÀ CSS HIỆN ĐẠI (PREMIUM UI) ---
st.set_page_config(
    page_title="Hệ Sinh Thái Ảnh Thẻ Cao Cấp - SHOPTINHOC", 
    layout="wide", 
    page_icon="📸",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    .brand-container {
        text-align: center;
        padding: 20px 10px;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    .main-title {
        font-size: 2rem;
        color: #ffffff;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 0.9rem;
        color: #bfdbfe;
        margin-top: 5px;
        font-weight: 400;
    }
    
    .image-container {
        border: none;
        padding: 20px;
        border-radius: 20px;
        background-color: #ffffff;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.05), 0 10px 10px -5px rgba(0,0,0,0.02);
        border: 1px solid #f1f5f9;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre;
        background-color: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1e40af;
        background-color: #ffffff;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff;
        color: #0f172a;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    div[data-testid="stButton"] > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    
    .premium-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rembg_session():
    return new_session("u2netp")

# --- 2. LOGIC HÀM XỬ LÝ ---
def reset_beauty_params():
    st.session_state.val_smooth = 0
    st.session_state.val_makeup = 0
    st.session_state.val_exposure = 1.0
    st.session_state.val_contrast = 1.0
    st.session_state.val_temp = 0
    st.session_state.val_sharp_amount = 0 
    st.session_state.val_clarity = 0
    st.session_state.val_denoise = 0      
    st.session_state.val_blacks = 0       
    st.session_state.val_whites = 0       
    st.session_state.val_dehaze = 0
    st.session_state.val_zoom = 1.0
    st.session_state.val_move_x = 0
    st.session_state.val_move_y = 0
    st.session_state.val_edge_soft = 0
    st.session_state.auto_level = 0

def apply_gender_preset():
    if 'gender_radio' in st.session_state:
        style = st.session_state.gender_radio
        if style == "Nam":
            st.session_state.val_smooth = 5
            st.session_state.val_makeup = 2
            st.session_state.val_exposure = 1.05
            st.session_state.val_contrast = 1.15
            st.session_state.val_sharp_amount = 20
            st.session_state.val_clarity = 15
            st.session_state.val_denoise = 5
            st.session_state.val_blacks = 10
            st.session_state.val_whites = 5
            st.toast("👨 Đã áp dụng mẫu Nam chuyên nghiệp")
        else:
            st.session_state.val_smooth = 25
            st.session_state.val_makeup = 20
            st.session_state.val_exposure = 1.10
            st.session_state.val_contrast = 1.05
            st.session_state.val_sharp_amount = 10
            st.session_state.val_clarity = 5
            st.session_state.val_denoise = 10
            st.session_state.val_whites = 15
            st.toast("👩 Đã áp dụng mẫu Nữ tự nhiên")

def set_auto_beauty():
    if 'auto_level' not in st.session_state:
        st.session_state.auto_level = 0
    current_level = st.session_state.auto_level
    next_level = (current_level + 1) % 3
    st.session_state.auto_level = next_level

    if next_level == 1:
        st.toast("✨ Tối ưu Level 1: Nhẹ nhàng")
        st.session_state.val_smooth = 5
        st.session_state.val_makeup = 2
        st.session_state.val_exposure = 1.05
        st.session_state.val_whites = 6
        st.session_state.val_blacks = 4
        st.session_state.val_sharp_amount = 2
        st.session_state.val_edge_soft = 2
    elif next_level == 2:
        st.toast("✨✨ Tối ưu Level 2: Sắc nét thương mại")
        st.session_state.val_smooth = 10
        st.session_state.val_makeup = 4
        st.session_state.val_exposure = 1.10
        st.session_state.val_whites = 12
        st.session_state.val_blacks = 8
        st.session_state.val_sharp_amount = 4
        st.session_state.val_edge_soft = 4
    else:
        st.toast("🔄 Đã đặt lại thông số gốc")
        reset_beauty_params()
        return

    st.session_state.val_contrast = 1.0
    st.session_state.val_temp = 0
    st.session_state.val_clarity = 0
    st.session_state.val_denoise = 0
    st.session_state.val_dehaze = 0

def resize_image_input(image, max_height=1200):
    w, h = image.size
    if h > max_height:
        ratio = max_height / h
        new_w = int(w * ratio)
        return image.resize((new_w, max_height), Image.Resampling.LANCZOS)
    return image

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def get_face_angle(gray_img, face_rect):
    (x, y, w, h) = face_rect
    roi_gray = gray_img[y:y+h, x:x+w]
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        p1 = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
        p2 = (eyes[-1][0] + eyes[-1][2]//2, eyes[-1][1] + eyes[-1][3]//2)
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        if delta_x < w/5: return 0.0
        return np.degrees(np.arctan2(delta_y, delta_x))
    return 0.0

def detect_face_mediapipe(img_bgra):
    if not HAS_MEDIAPIPE: return None, 0.0
    img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
    h, w, _ = img_rgb.shape
    
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img_rgb)
        if not results.detections: return None, 0.0
            
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        
        x = int(bboxC.xmin * w)
        y = int(bboxC.ymin * h)
        width = int(bboxC.width * w)
        height = int(bboxC.height * h)
        face_rect = (x, y, width, height)
        
        right_eye = detection.location_data.relative_keypoints[0]
        left_eye = detection.location_data.relative_keypoints[1]
        p1 = (int(right_eye.x * w), int(right_eye.y * h))
        p2 = (int(left_eye.x * w), int(left_eye.y * h))
        
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        angle = 0.0 if delta_x == 0 else np.degrees(np.arctan2(delta_y, delta_x))
        return face_rect, angle

def process_raw_to_nobg(file_input):
    image = Image.open(file_input)
    image = resize_image_input(image, max_height=1200)
    session = get_rembg_session()
    
    no_bg_pil = remove(
        image, 
        session=session, 
        alpha_matting=True, 
        alpha_matting_foreground_threshold=240, 
        alpha_matting_background_threshold=10, 
        alpha_matting_erode_size=2 
    )
    
    no_bg_cv = cv2.cvtColor(np.array(no_bg_pil), cv2.COLOR_RGBA2BGRA)
    b, g, r, alpha = cv2.split(no_bg_cv)
    _, alpha_sharp = cv2.threshold(alpha, 200, 255, cv2.THRESH_BINARY)
    no_bg_cv = cv2.merge([b, g, r, alpha_sharp])
    return no_bg_cv

def crop_final_image(no_bg_img, manual_angle, target_ratio, detector_type="MediaPipe"):
    try:
        img_working = no_bg_img.copy()
        if detector_type == "MediaPipe" and HAS_MEDIAPIPE:
            result = detect_face_mediapipe(img_working)
            if result[0] is None: return None, "Không tìm thấy khuôn mặt (MediaPipe)", 0
            face_rect, auto_angle = result
            (x, y, w, h) = face_rect
        else:
            gray = cv2.cvtColor(img_working, cv2.COLOR_BGRA2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) == 0: return None, "Không tìm thấy khuôn mặt (Haarcascade)", 0
            face_rect = max(faces, key=lambda f: f[2] * f[3])
            auto_angle = get_face_angle(gray, face_rect)
            (x, y, w, h) = face_rect

        if abs(auto_angle) < 1.0 or abs(auto_angle) > 20.0: auto_angle = 0.0 
        total_angle = auto_angle + manual_angle
        img_rotated = rotate_image(img_working, total_angle) if abs(total_angle) > 0.1 else img_working

        if detector_type == "MediaPipe" and HAS_MEDIAPIPE:
            result_new = detect_face_mediapipe(img_rotated)
            if result_new[0] is not None: (x, y, w, h) = result_new[0]
        else:
            gray_new = cv2.cvtColor(img_rotated, cv2.COLOR_BGRA2GRAY)
            faces_new = face_cascade.detectMultiScale(gray_new, 1.1, 5)
            if len(faces_new) > 0: (x, y, w, h) = max(faces_new, key=lambda f: f[2] * f[3])

        if target_ratio == 1.0: 
            zoom_factor = 1.8  
            top_offset = 0.55 
        elif 0.77 <= target_ratio <= 0.78: 
            zoom_factor = 1.7  
            top_offset = 0.50 
        elif 0.68 <= target_ratio <= 0.69: 
            zoom_factor = 1.75 
            top_offset = 0.50  
        elif target_ratio < 0.7: 
            zoom_factor = 2.0  
            top_offset = 0.45   
        else: 
            zoom_factor = 2.2
            top_offset = 0.5

        crop_h = int(h * zoom_factor) 
        crop_w = int(crop_h * target_ratio)
        face_center_x = x + w // 2
        top_y = int(y - (h * top_offset)) 
        left_x = int(face_center_x - crop_w // 2)

        img_pil = Image.fromarray(cv2.cvtColor(img_rotated, cv2.COLOR_BGRA2RGBA))
        canvas = Image.new("RGBA", (crop_w, crop_h), (0,0,0,0))
        canvas.paste(img_pil, (-left_x, -top_y), img_pil)
        return canvas, f"Góc Auto ({detector_type}): {auto_angle:.1f}°", total_angle
    except Exception as e:
        return None, str(e), 0

def apply_transform(image, zoom=1.0, move_x=0, move_y=0):
    if zoom == 1.0 and move_x == 0 and move_y == 0: return image
    w, h = image.size
    new_w = int(w * zoom)
    new_h = int(h * zoom)
    img_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    center_x = (w - new_w) // 2
    center_y = (h - new_h) // 2
    canvas.paste(img_resized, (center_x + move_x, center_y + move_y), img_resized)
    return canvas

def apply_edge_softness(image_rgba, strength=0):
    if strength == 0: return image_rgba
    img = np.array(image_rgba)
    alpha = img[:, :, 3]
    k_size = int(strength) * 2 + 1 
    alpha_blurred = cv2.GaussianBlur(alpha, (k_size, k_size), 0)
    img[:, :, 3] = alpha_blurred
    return Image.fromarray(img)

def adjust_levels(image, blacks=0, whites=0):
    if blacks == 0 and whites == 0: return image
    in_black = blacks
    in_white = 255 - whites
    if in_black >= in_white: in_black = in_white - 1
    lut = np.arange(256, dtype=np.float32)
    scale = 255.0 / (in_white - in_black)
    lut = (lut - in_black) * scale
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(image, lut)

def apply_super_sharpen(image, amount=0):
    if amount == 0: return image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return cv2.addWeighted(image, 1.0 - (amount/40.0), sharpened, (amount/40.0), 0)

def apply_clarity(image_bgr, amount=0):
    if amount == 0: return image_bgr
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=(amount / 10.0) + 1.0, tileGridSize=(8, 8))
    l_new = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_new, a, b)), cv2.COLOR_LAB2BGR)

def apply_advanced_effects(base_img, params):
    img_transformed = apply_transform(base_img, params['zoom'], params['move_x'], params['move_y'])
    if params['edge_soft'] > 0:
        img_transformed = apply_edge_softness(img_transformed, params['edge_soft'])

    img_bgra = cv2.cvtColor(np.array(img_transformed), cv2.COLOR_RGBA2BGRA)
    b, g, r, a = cv2.split(img_bgra)
    img_bgr = cv2.merge([b, g, r])
    
    if params['denoise'] > 0:
        h_val = params['denoise']
        img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, h_val, h_val, 7, 21)
    if params['smooth'] > 0:
        sigma = int(params['smooth'] * 2) + 10
        img_bgr = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=sigma, sigmaSpace=sigma)
    if params['dehaze'] > 0:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_c, a_c, b_c = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0 + (params['dehaze']/10.0), tileGridSize=(8,8))
        l_c = clahe.apply(l_c)
        img_bgr = cv2.cvtColor(cv2.merge((l_c, a_c, b_c)), cv2.COLOR_LAB2BGR)
    if params['temp'] != 0:
        temp = int(params['temp'])
        b_c, g_c, r_c = cv2.split(img_bgr)
        r_c = cv2.add(r_c, temp)
        b_c = cv2.subtract(b_c, temp)
        img_bgr = cv2.merge([b_c, g_c, r_c])
    if params['makeup'] > 0:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_c, s_c, v_c = cv2.split(hsv)
        s_c = cv2.add(s_c, int(params['makeup'] * 1.5))
        v_c = cv2.add(v_c, int(params['makeup'] * 0.5))
        img_bgr = cv2.cvtColor(cv2.merge([h_c, s_c, v_c]), cv2.COLOR_HSV2BGR)
    if params['blacks'] > 0 or params['whites'] > 0:
        img_bgr = adjust_levels(img_bgr, params['blacks'], params['whites'])
    if params['clarity'] > 0:
        img_bgr = apply_clarity(img_bgr, params['clarity'])
    if params['sharp_amount'] > 0:
        img_bgr = apply_super_sharpen(img_bgr, params['sharp_amount'])

    final_bgra = cv2.merge([img_bgr[:,:,0], img_bgr[:,:,1], img_bgr[:,:,2], a])
    img_pil = Image.fromarray(cv2.cvtColor(final_bgra, cv2.COLOR_BGRA2RGBA))
    if params['exposure'] != 1.0:
        img_pil = ImageEnhance.Brightness(img_pil).enhance(params['exposure'])
    if params['contrast'] != 1.0:
        img_pil = ImageEnhance.Contrast(img_pil).enhance(params['contrast'])
    return img_pil

def create_pdf(img_person, size_type):
    if not HAS_FPDF: return None
    pdf = FPDF(orientation='P', unit='mm', format='A4' if "4x6" in size_type else (105, 148))
    pdf.add_page()
    temp_img_path = "temp_print.jpg"
    img_person.save(temp_img_path, quality=100, dpi=(300, 300))
    
    if "5x5" in size_type:
        w_mm, h_mm, cols, rows, margin_x, margin_y = 50, 50, 2, 2, 2, 5
    elif "3.5x4.5" in size_type: 
        w_mm, h_mm, cols, rows, margin_x, margin_y = 35, 45, 2, 3, 17, 6
    elif "3.3x4.8" in size_type: 
        w_mm, h_mm, cols, rows, margin_x, margin_y = 33, 48, 2, 2, 19, 20
    elif "4x6" in size_type:
        w_mm, h_mm, cols, rows, margin_x, margin_y = 40, 60, 2, 4, 62, 25
    else: 
        w_mm, h_mm, cols, rows, margin_x, margin_y = 29.0, 39.0, 7, 7, 1.0, 2.0

    for r in range(rows):
        for c in range(cols):
            gap = 4 if "4x6" in size_type else 0.2
            pdf.image(temp_img_path, x=margin_x + c * (w_mm + gap), y=margin_y + r * (h_mm + gap), w=w_mm, h=h_mm)
    return bytes(pdf.output())

def create_print_layout_preview(img_person, size_type):
    PAPER_W_PX, PAPER_H_PX = (2480, 3508) if "4x6" in size_type else (1240, 1748)
    bg_paper = Image.new("RGB", (PAPER_W_PX, PAPER_H_PX), (255, 255, 255))
    
    if "5x5" in size_type: 
        target_w, target_h, rows, cols, start_x, start_y, gap = 590, 590, 2, 2, 30, 200, 30
    elif "3.5x4.5" in size_type:
        target_w, target_h, rows, cols, start_x, start_y, gap = 413, 531, 3, 2, 190, 80, 40
    elif "3.3x4.8" in size_type: 
        target_w, target_h, rows, cols, start_x, start_y, gap = 390, 567, 2, 2, 200, 250, 40
    elif "4x6" in size_type:
        target_w, target_h, rows, cols, start_x, start_y, gap = 472, 708, 4, 2, 743, 263, 50
    else: 
        target_w, target_h, rows, cols, start_x, start_y, gap = 342, 460, 7, 7, 10, 20, 2

    img_resized = img_person.resize((target_w, target_h), Image.Resampling.LANCZOS)
    for r in range(rows):
        for c in range(cols):
            bg_paper.paste(img_resized, (start_x + c * (target_w + gap), start_y + r * (target_h + gap)))
    return bg_paper

# --- 3. GIAO DIỆN DESIGN THƯƠNG MẠI ---
st.markdown("""
<div class="brand-container">
    <div class="main-title">🌟 SMART ID STUDIO PRO</div>
    <div class="sub-title">Hệ thống xử lý và tối ưu hóa ảnh thẻ thông minh chuyên nghiệp dành cho doanh nghiệp</div>
</div>
""", unsafe_allow_html=True)

if not HAS_FPDF:
    st.warning("⚠️ Hệ thống in ấn PDF (fpdf) chưa được đồng bộ hoàn toàn.")

# --- SIDEBAR MENU TẬP TRUNG KHÁCH HÀNG ---
with st.sidebar:
    st.markdown("### 🛠️ KHÔNG GIAN LÀM VIỆC")
    app_mode = st.radio("Chọn chế độ vận hành:", [
        "📸 Studio Xử Lý (Cá nhân)", 
        "👥 Ghép In Hàng Loạt (Số lượng lớn)"
    ])
    st.markdown("---")

# ==============================================================================
# CHẾ ĐỘ SỐ LƯỢNG LỚN (10 NGƯỜI - XẾP 7 ẢNH/HÀNG CHUẨN KÍCH THƯỚC)
# ==============================================================================
if app_mode == "👥 Ghép In Hàng Loạt (Số lượng lớn)":
    st.info("⚙️ Chế độ in hàng loạt chuẩn: Kích thước 29.0mm x 39.0mm giúp xếp chuẩn 7 ảnh 3x4 trên 1 hàng A4")
    
    html_code = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%); 
            display: flex; justify-content: center; align-items: flex-start; 
            min-height: auto; margin: 0; padding: 10px 0;
        }
        .container { 
            background: #ffffff; padding: 25px 35px; border-radius: 20px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.08); max-width: 850px; width: 100%; text-align: center;
            margin-top: 0;
        }
        h2 { color: #2c3e50; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px; margin-top: 0;}
        .upload-group { display: flex; justify-content: space-between; gap: 20px; margin-bottom: 15px;}
        .person-box { 
            flex: 1; border: 2px dashed #b8c2cc; padding: 12px 10px; border-radius: 14px; 
            background: #fafafa; transition: all 0.3s ease; position: relative; text-align: center;
        }
        .person-box:hover { border-color: #007bff; background: #f0f7ff;}
        .person-box h4 { margin: 0 0 8px 0; color: #0056b3; font-size: 15px; font-weight: 700;}
        .name-input {
            width: 85%; padding: 6px 10px; margin: 6px auto; border: 1px solid #ced4da;
            border-radius: 6px; font-size: 13px; outline: none; text-align: center; display: block;
        }
        .name-input:focus { border-color: #007bff; box-shadow: 0 0 4px rgba(0,123,255,0.2); }
        .qty-area {
            margin-top: 8px; background: #eee; padding: 8px; border-radius: 8px;
            display: flex; flex-direction: column; gap: 6px;
        }
        .qty-row { display: flex; justify-content: space-between; align-items: center; font-size: 13px; font-weight: bold; color: #444;}
        .qty-row input { width: 50px; text-align: center; padding: 4px; border-radius: 4px; border: 1px solid #ccc; font-weight: bold;}
        .badge { color: white; padding: 3px 6px; border-radius: 4px; font-size: 11px;}
        .bg-3x4 { background: #007bff; }
        .bg-4x6 { background: #28a745; }
        input[type="file"] { display: none; }
        .custom-file-upload { 
            display: inline-block; padding: 8px 12px; cursor: pointer; background-color: #edf2f7; 
            color: #4a5568; border-radius: 8px; font-weight: 600; font-size: 12px; 
            border: 1px solid #e2e8f0; width: 85%; margin: 0 auto;
        }
        .custom-file-upload:hover { background-color: #e2e8f0; }
        .img-wrapper { position: relative; display: inline-block; margin-top: 8px; }
        .preview { 
            max-width: 80px; max-height: 100px; border-radius: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); 
            border: 2px solid #fff; display: none; object-fit: cover;
        }
        .clear-btn { 
            position: absolute; top: -8px; right: -8px; background: #ff4757; color: white; 
            border: none; border-radius: 50%; width: 20px; height: 20px; font-size: 10px; 
            font-weight: bold; cursor: pointer; display: none; align-items: center; justify-content: center;
        }
        .btn-group { display: flex; gap: 12px; justify-content: center; margin-top: 25px;}
        .btn { 
            border-radius: 50px; padding: 14px 20px; font-size: 14px; font-weight: 700; 
            text-transform: uppercase; letter-spacing: 0.5px; cursor: pointer; color: white; border: none; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1); transition: all 0.3s ease; flex: 1; 
        }
        #previewBtn { background: linear-gradient(135deg, #36D1DC 0%, #5B86E5 100%); }
        #downloadBtn { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); display: none; }
        #directPrintBtn { background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%); display: none; }
        #previewContainer { display: none; margin-top: 30px; border-top: 2px dashed #e2e8f0; padding-top: 20px; }
        #previewContainer h4 { color: #4a5568; margin-bottom: 20px; font-weight: 700;}
        .a4-page-preview {
            position: relative; width: 100%; max-width: 480px; background: white; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin: 0 auto 30px auto; 
            border: 1px solid #ccc; overflow: hidden; border-radius: 4px;
            box-sizing: border-box;
        }
        .label-text-style {
            position: absolute; width: 100%; text-align: center; color: #333;
            font-family: Arial, sans-serif; font-weight: bold; overflow: hidden; white-space: nowrap;
        }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
</head>
<body>
    <div class="container">
        <h2>HỆ THỐNG XẾP IN HỒ SƠ CAO CẤP SMART STUDIO (10 NGƯỜI)</h2>
        
        <!-- Hàng 1 đến 5 -->
        <script>
            let htmlInputs = '';
            for(let i = 1; i <= 10; i+=2) {
                htmlInputs += `
                <div class="upload-group">
                    <div class="person-box">
                        <h4>👤 Người thứ ${i}</h4>
                        <input type="text" id="name${i}" class="name-input" placeholder="Nhập tên học viên...">
                        <label for="imgInput${i}" class="custom-file-upload" id="labelInput${i}">📁 Chọn Ảnh...</label>
                        <input type="file" id="imgInput${i}" accept="image/png, image/jpeg, image/jpg">
                        <center><div class="img-wrapper"><img id="preview${i}" class="preview" alt="Preview ${i}"><button id="clearBtn${i}" class="clear-btn">✖</button></div></center>
                        <div class="qty-area">
                            <div class="qty-row"><span><span class="badge bg-3x4">3x4</span> SL:</span><input type="number" id="qty3x4_${i}" value="${i <= 2 ? 7 : 0}" min="0" max="28"></div>
                            <div class="qty-row"><span><span class="badge bg-4x6">4x6</span> SL:</span><input type="number" id="qty4x6_${i}" value="0" min="0" max="28"></div>
                        </div>
                    </div>
                    <div class="person-box">
                        <h4>👤 Người thứ ${i+1}</h4>
                        <input type="text" id="name${i+1}" class="name-input" placeholder="Nhập tên học viên...">
                        <label for="imgInput${i+1}" class="custom-file-upload" id="labelInput${i+1}">📁 Chọn Ảnh...</label>
                        <input type="file" id="imgInput${i+1}" accept="image/png, image/jpeg, image/jpg">
                        <center><div class="img-wrapper"><img id="preview${i+1}" class="preview" alt="Preview ${i+1}"><button id="clearBtn${i+1}" class="clear-btn">✖</button></div></center>
                        <div class="qty-area">
                            <div class="qty-row"><span><span class="badge bg-3x4">3x4</span> SL:</span><input type="number" id="qty3x4_${i+1}" value="${i <= 2 ? 7 : 0}" min="0" max="28"></div>
                            <div class="qty-row"><span><span class="badge bg-4x6">4x6</span> SL:</span><input type="number" id="qty4x6_${i+1}" value="0" min="0" max="28"></div>
                        </div>
                    </div>
                </div>`;
            }
            document.write(htmlInputs);
        </script>

        <div class="btn-group">
            <button id="previewBtn" class="btn">👁️ Xem Trước Bản Xếp</button>
            <button id="downloadBtn" class="btn">⬇️ Tải Xuống PDF</button>
            <button id="directPrintBtn" class="btn">🖨️ Tiến Hành In</button>
        </div>
        <div id="previewContainer">
            <h4>📄 MÔ PHỎNG TRANG IN CHUẨN A4</h4>
            <div id="pdfIframeContainer"></div>
        </div>
    </div>
    <script>
        let dataStore = Array(11).fill(null);
        let typeStore = Array(11).fill('JPEG');

        function handleImageUpload(inputId, previewId, clearBtnId, labelId, personNum) {
            document.getElementById(inputId).addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    typeStore[personNum] = (file.type === 'image/png') ? 'PNG' : 'JPEG';

                    const reader = new FileReader();
                    reader.onload = function(event) {
                        dataStore[personNum] = event.target.result;

                        const imgElement = document.getElementById(previewId);
                        imgElement.src = event.target.result;
                        imgElement.style.display = 'block';
                        document.getElementById(clearBtnId).style.display = 'flex';
                        document.getElementById(labelId).innerHTML = '🔄 Đổi Ảnh';
                    }
                    reader.readAsDataURL(file);
                }
            });
            document.getElementById(clearBtnId).addEventListener('click', function() {
                document.getElementById(inputId).value = "";
                document.getElementById(previewId).style.display = 'none';
                document.getElementById(previewId).src = "";
                this.style.display = 'none';
                document.getElementById(labelId).innerHTML = '📁 Chọn Ảnh...';
                dataStore[personNum] = null;

                document.getElementById('previewContainer').style.display = 'none';
                document.getElementById('downloadBtn').style.display = 'none';
                document.getElementById('directPrintBtn').style.display = 'none';
            });
        }

        for(let i = 1; i <= 10; i++) {
            handleImageUpload(`imgInput${i}`, `preview${i}`, `clearBtn${i}`, `labelInput${i}`, i);
        }

        function getPersonsData() {
            let list = [];
            for(let i = 1; i <= 10; i++) {
                let q3x4 = parseInt(document.getElementById(`qty3x4_${i}`).value) || 0;
                let q4x6 = parseInt(document.getElementById(`qty4x6_${i}`).value) || 0;
                let pName = document.getElementById(`name${i}`).value.trim();
                if (dataStore[i] && (q3x4 > 0 || q4x6 > 0)) {
                    list.push({ data: dataStore[i], type: typeStore[i], qty3x4: q3x4, qty4x6: q4x6, name: pName });
                }
            }
            return list;
        }

        // TỐI ƯU TOÁN TỬ VÀ KÍCH THƯỚC TRÁNH BỊ TRỎI HÀNG
        function buildLayoutData(persons) {
            const a4W = 210, a4H = 297;
            // 7 ảnh * 29.0mm = 203mm, lề marginX = 1mm, gapX = 0.2mm => Tổng = 206.2mm <= 210mm
            let gapX = 0.2, gapY = 0.5, marginX = 1.0, marginY = 1.5;
            let pages = [], currentPage = [], curX = marginX, curY = marginY;
            let maxRowHeight = 0;

            let allItems = [];
            persons.forEach((person) => {
                for (let i = 0; i < person.qty3x4; i++) {
                    allItems.push({ data: person.data, type: person.type, w: 29.0, h: 39.0, name: person.name });
                }
                for (let i = 0; i < person.qty4x6; i++) {
                    allItems.push({ data: person.data, type: person.type, w: 39.0, h: 59.0, name: person.name });
                }
            });

            allItems.forEach((item) => {
                // Sử dụng toán tử so sánh chính xác để tránh rớt dòng
                if (Math.round((curX + item.w) * 10) > Math.round((a4W - marginX) * 10)) {
                    curX = marginX;
                    curY += maxRowHeight + gapY;
                    maxRowHeight = 0;
                }

                if (Math.round((curY + item.h) * 10) > Math.round((a4H - marginY) * 10)) {
                    pages.push(currentPage);
                    currentPage = [];
                    curX = marginX;
                    curY = marginY;
                    maxRowHeight = 0;
                }

                currentPage.push({ data: item.data, type: item.type, x: curX, y: curY, w: item.w, h: item.h, name: item.name });

                if (item.h > maxRowHeight) {
                    maxRowHeight = item.h;
                }

                curX += item.w + gapX;
            });

            if (currentPage.length > 0) pages.push(currentPage);
            return pages;
        }

        document.getElementById('previewBtn').addEventListener('click', function() {
            let persons = getPersonsData();
            if (persons.length === 0) return alert("Vui lòng tải ảnh lên và nhập số lượng!");
            let pages = buildLayoutData(persons);
            let pagesHtml = '';

            pages.forEach(page => {
                pagesHtml += `<div class="a4-page-preview" style="aspect-ratio: 210/297; border: 1px solid #777; background:#fff; margin-bottom:20px; position:relative;">`;
                page.forEach(img => {
                    let pLeft = (img.x / 210) * 100 + '%';
                    let pTop = (img.y / 297) * 100 + '%';
                    let pWidth = (img.w / 210) * 100 + '%';
                    let pHeight = (img.h / 297) * 100 + '%';
                    pagesHtml += `<img src="${img.data}" style="position: absolute; left: ${pLeft}; top: ${pTop}; width: ${pWidth}; height: ${pHeight}; object-fit: cover; box-sizing: border-box;">`;
                    if (img.name) {
                        let labelTop = ((img.y + img.h - 3.2) / 297) * 100 + '%';
                        let labelFontSize = (img.w < 35) ? '8px' : '9px';
                        pagesHtml += `<div class="label-text-style" style="left: ${pLeft}; top: ${labelTop}; font-size: ${labelFontSize}; background: rgba(255,255,255,0.8); height:13px; line-height:13px;">${img.name}</div>`;
                    }
                });
                pagesHtml += `</div>`;
            });
            document.getElementById('pdfIframeContainer').innerHTML = pagesHtml;
            document.getElementById('previewContainer').style.display = 'block';
            document.getElementById('downloadBtn').style.display = 'inline-block';
            document.getElementById('directPrintBtn').style.display = 'inline-block';
        });

        function generateJsPDFObject() {
            let persons = getPersonsData();
            const { jsPDF } = window.jspdf;
            let doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
            let pages = buildLayoutData(persons);
            pages.forEach((page, pageIdx) => {
                if (pageIdx > 0) doc.addPage();
                page.forEach(img => {
                    doc.addImage(img.data, img.type, img.x, img.y, img.w, img.h);
                    doc.setDrawColor(225, 225, 225); doc.setLineWidth(0.08); doc.rect(img.x, img.y, img.w, img.h, 'S');
                    if (img.name) {
                        doc.setFillColor(255, 255, 255); doc.rect(img.x + 0.2, img.y + img.h - 3.0, img.w - 0.4, 2.8, 'F');
                        doc.setTextColor(60, 60, 60); let fSize = (img.w < 35) ? 5.5 : 6.5;
                        doc.setFontSize(fSize); doc.setFont("Helvetica", "bold");
                        doc.text(img.name, img.x + (img.w / 2), img.y + img.h - 0.8, { align: 'center' });
                    }
                });
            });
            return doc;
        }

        document.getElementById('downloadBtn').addEventListener('click', function() { generateJsPDFObject().save('SmartStudio_Print_Layout.pdf'); });
        document.getElementById('directPrintBtn').addEventListener('click', function() {
            let doc = generateJsPDFObject(); const blobUrl = doc.output('bloburl'); const printWindow = window.open(blobUrl, '_blank');
            if (printWindow) { printWindow.onload = function() { printWindow.focus(); printWindow.print(); }; }
            else { alert("Vui lòng cho phép Pop-up trên trình duyệt để in trực tiếp!"); }
        });
    </script>
</body>
</html>"""
    components.html(html_code, height=1850, scrolling=True)
    st.stop()

# ==============================================================================
# HOẠT ĐỘNG CHẾ ĐỘ STUDIO XỬ LÝ (TỐI ƯU GIAO DIỆN PREMIUM)
# ==============================================================================
with st.sidebar:
    st.markdown("### 📥 NGUỒN DỮ LIỆU ĐẦU VÀO")
    input_method = st.radio("Phương thức:", ["📁 Tải ảnh từ máy", "📷 Sử dụng Camera"], horizontal=True)
    
    input_file = None
    if input_method == "📁 Tải ảnh từ máy":
        input_file = st.file_uploader("Kéo thả ảnh vào đây (.JPG, .PNG)", type=['jpg', 'png', 'jpeg'])
    else:
        input_file = st.camera_input("Chụp ảnh trực tiếp")

    st.markdown("---")
    st.markdown("### 🧠 ĐỘNG CƠ TRÍ TUỆ NHÂN TẠO")
    if HAS_MEDIAPIPE:
        detector_option = st.radio("Bộ máy quét khuôn mặt:", ["MediaPipe (Độ chính xác cao)", "Haarcascade (Mô hình dự phòng)"], horizontal=True)
        detector_type = "MediaPipe" if "MediaPipe" in detector_option else "Haarcascade"
    else:
        st.warning("⚠️ Đang chạy chế độ Haarcascade mặc định.")
        detector_type = "Haarcascade"

    st.markdown("---")
    st.markdown("### 📐 TIÊU CHUẨN ĐẦU RA")
    size_option = st.selectbox("Chọn kích thước ảnh cần xuất:", [
        "4x6 cm (Hộ chiếu Quốc tế)", 
        "3.5x4.5 cm (Visa Úc / Hàn / Âu / Đài Loan)",
        "5x5 cm (Visa Mỹ / Hộ chiếu Mỹ)",
        "3.3x4.8 cm (Visa Trung Quốc)", 
        "3x4 cm (Hồ sơ học tập & GPLX)"
    ])
    
    if "Visa Mỹ" in size_option: target_ratio = 1.0 
    elif "3.5x4.5" in size_option: target_ratio = 3.5/4.5
    elif "Visa Trung Quốc" in size_option: target_ratio = 3.3/4.8
    elif "3x4" in size_option: target_ratio = 3/4
    else: target_ratio = 4/6
    
    bg_name = st.radio("Màu phông nền mong muốn:", ["Trắng tinh khôi", "Xanh chuẩn quốc tế", "Xanh nhạt pastel", "Xanh GPLX chuẩn Bộ GTVT"], horizontal=False)
    bg_map = {
        "Trắng tinh khôi": (255, 255, 255, 255), 
        "Xanh chuẩn quốc tế": (66, 135, 245, 255), 
        "Xanh nhạt pastel": (135, 206, 250, 255),
        "Xanh GPLX chuẩn Bộ GTVT": (37, 133, 197, 255)
    }
    bg_val = bg_map.get(bg_name)

# --- XỬ LÝ ẢNH CHUYÊN NGHIỆP ---
if input_file:
    current_file_key = f"{input_file.name}_{input_file.size}"
    if 'current_file_key' in st.session_state and st.session_state.current_file_key != current_file_key:
        if 'raw_nobg' in st.session_state: del st.session_state.raw_nobg
        if 'base' in st.session_state: del st.session_state.base
        gc.collect()

    if 'current_file_key' not in st.session_state or st.session_state.current_file_key != current_file_key:
        with st.spinner('⏳ Đang tiến hành bóc tách nền bằng thuật toán AI...'):
            try:
                st.session_state.raw_nobg = process_raw_to_nobg(input_file)
                st.session_state.current_file_key = current_file_key
            except Exception as e: 
                st.error(f"Không thể xử lý tệp tin đầu vào: {e}")

col_btn1, col_btn2, col_space = st.columns([1.5, 1, 3])
with col_btn1:
    current_lvl = st.session_state.get('auto_level', 0)
    label_auto = f"✨ CHẾ ĐỘ AUTO: ĐANG BẬT LV {current_lvl}" if current_lvl > 0 else "✨ CLICK AUTO LÀM ĐẸP NGAY"
    st.button(label_auto, on_click=set_auto_beauty, type="primary", use_container_width=True)
with col_btn2:
    st.button("🔄 Đặt lại từ đầu", on_click=reset_beauty_params, use_container_width=True)

st.divider()

col_tools, col_result = st.columns([1.1, 1.2], gap="large")

with col_tools:
    st.markdown("### 🎛️ BẢNG ĐIỀU CHỈNH CHUYÊN SÂU")
    
    manual_rot = st.slider("🔄 Đồng bộ trục thẳng (Xoay đầu):", -15.0, 15.0, 0.0, 0.5)
    if 'raw_nobg' in st.session_state:
        final_crop, debug_info, _ = crop_final_image(st.session_state.raw_nobg, manual_rot, target_ratio, detector_type)
        if final_crop: st.session_state.base = final_crop
        else: st.error(f"Thông báo hệ thống: {debug_info}")

    tab1, tab2, tab3 = st.tabs(["🎨 Ánh Sáng & Sắc Độ", "👩 Thẩm Mỹ Khuôn Mặt", "📐 Bố Cục & Chi Tiết"])
    
    with tab1:
        p_exposure = st.slider("Độ sáng cân bằng", 0.5, 1.5, st.session_state.get('val_exposure', 1.0), 0.05, key="val_exposure")
        p_contrast = st.slider("Độ tương phản ảnh", 0.5, 1.5, st.session_state.get('val_contrast', 1.0), 0.05, key="val_contrast")
        p_temp = st.slider("Nhiệt độ màu (Ấm/Lạnh)", -50, 50, st.session_state.get('val_temp', 0), key="val_temp")
        col_b, col_w = st.columns(2)
        with col_b: p_blacks = st.slider("Màu Đen (Blacks)", 0, 50, st.session_state.get('val_blacks', 0), key="val_blacks")
        with col_w: p_whites = st.slider("Màu Trắng (Whites)", 0, 50, st.session_state.get('val_whites', 0), key="val_whites")

    with tab2:
        p_smooth = st.slider("Mịn da kỹ thuật số", 0, 30, st.session_state.get('val_smooth', 0), key="val_smooth")
        p_makeup = st.slider("Hồng hào / Makeup tươi tắn", 0, 50, st.session_state.get('val_makeup', 0), key="val_makeup")
        st.markdown("---")
        ai_enabled = st.checkbox("🎯 Kích hoạt Preset thương mại nhanh", key='ai_enabled')
        if ai_enabled:
            st.radio("Chọn giới tính để tự động căn chỉnh da:", ["Nam", "Nữ"], horizontal=True, key="gender_radio", on_change=apply_gender_preset)

    with tab3:
        p_zoom = st.slider("Phóng to / Thu nhỏ tỷ lệ khuôn mặt", 0.5, 1.5, st.session_state.get('val_zoom', 1.0), 0.05, key="val_zoom")
        col_m1, col_m2 = st.columns(2)
        with col_m1: p_move_x = st.number_input("Dịch chuyển ngang (Pixel)", -100, 100, st.session_state.get('val_move_x', 0), key="val_move_x")
        with col_m2: p_move_y = st.number_input("Dịch chuyển dọc (Pixel)", -100, 100, st.session_state.get('val_move_y', 0), key="val_move_y")
        st.markdown("---")
        p_sharp_amount = st.slider("Độ sắc nét chi tiết", 0, 50, st.session_state.get('val_sharp_amount', 0), key="val_sharp_amount")
        p_clarity = st.slider("Độ rõ nét khối (Clarity)", 0, 50, st.session_state.get('val_clarity', 0), key="val_clarity")
        p_denoise = st.slider("Khử hạt hạt nhiễu (Denoise)", 0, 20, st.session_state.get('val_denoise', 0), key="val_denoise")
        p_dehaze = st.slider("Khử sương mờ", 0, 30, st.session_state.get('val_dehaze', 0), key="val_dehaze")
        p_edge_soft = st.slider("Làm mềm biên đúng chuẩn", 0, 10, st.session_state.get('val_edge_soft', 0), key="val_edge_soft")

    params = {
        'smooth': p_smooth, 'makeup': p_makeup, 'exposure': p_exposure, 'contrast': p_contrast, 'temp': p_temp,
        'sharp_amount': p_sharp_amount, 'clarity': p_clarity, 'dehaze': p_dehaze, 'blacks': p_blacks, 'whites': p_whites, 
        'denoise': p_denoise, 'zoom': p_zoom, 'move_x': p_move_x, 'move_y': p_move_y, 'edge_soft': p_edge_soft
    }

with col_result:
    st.markdown("### 🖼️ BẢN XEM TRƯỚC SẢN PHẨM KHÁCH HÀNG")
    if 'base' in st.session_state and st.session_state.base:
        with st.spinner("🚀 Đang hoàn thiện ảnh chất lượng cao..."):
            final_person = apply_advanced_effects(st.session_state.base, params)
        
        w, h = final_person.size
        final_img = Image.new("RGBA", (w, h), bg_val)
        final_img.paste(final_person, (0, 0), final_person)
        final_rgb = final_img.convert("RGB")

        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(final_rgb, caption=f"Hình ảnh đầu ra đạt chuẩn: {size_option}", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📥 MODULE XUẤT FILE THƯƠNG MẠI")
        d_tab1, d_tab2 = st.tabs(["💾 Lưu File Ảnh Đơn (JPG)", "🖨️ Xuất Bản Khổ Để In (PDF)"])
        
        with d_tab1:
            buf = io.BytesIO()
            final_rgb.save(buf, format="JPEG", quality=98, dpi=(300, 300))
            st.download_button(
                label="⬇️ TẢI ẢNH JPG ĐƠN (CHẤT LƯỢNG CAO)", 
                data=buf.getvalue(), 
                file_name=f"smart_id_photo.jpg", 
                mime="image/jpeg", 
                type="primary", 
                use_container_width=True
            )

        with d_tab2:
            st.image(create_print_layout_preview(final_rgb, size_option), caption="Mô phỏng vị trí trên giấy in ảnh tiêu chuẩn", use_container_width=True)
            if HAS_FPDF:
                pdf_data = create_pdf(final_rgb, size_option)
                if pdf_data:
                    st.download_button(
                        label="📄 TẢI FILE PDF ĐÃ XẾP KHỔ (SẴN SÀNG IN)", 
                        data=pdf_data, 
                        file_name="smart_id_print_ready.pdf", 
                        mime="application/pdf", 
                        use_container_width=True
                    )
            else:
                st.error("Lỗi: Hệ thống không tìm thấy lõi kết xuất PDF.")
        
        with st.expander("👁️ Khảo sát so sánh Trước / Sau chỉnh sửa"):
            c_before, c_after = st.columns(2)
            with c_before: st.image(st.session_state.base, caption="Ảnh gốc ban đầu")
            with c_after: st.image(final_rgb, caption="Sản phẩm hoàn thiện")
    else:
        st.info("👈 Vui lòng lựa chọn hoặc chụp ảnh từ bảng điều khiển bên trái để bắt đầu.")
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
