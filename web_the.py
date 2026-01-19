import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc
import sys

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Studio Ảnh Thẻ Pro AI", layout="wide", page_icon="📸")

# --- 2. IMPORT AN TOÀN (SAFE IMPORT) ---
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

HAS_MEDIAPIPE = False
mp_face_detection = None
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
        mp_face_detection = mp.solutions.face_detection
        HAS_MEDIAPIPE = True
    else:
        HAS_MEDIAPIPE = False
except ImportError:
    HAS_MEDIAPIPE = False
except Exception as e:
    HAS_MEDIAPIPE = False
    print(f"MediaPipe Error: {e}")

# --- 3. CSS GIAO DIỆN ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #B22222;
        text-align: center;
        font-weight: 800;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    div[data-testid="stButton"] > button:first-child {
        border-radius: 8px;
        font-weight: bold;
    }
    .image-container {
        border: 3px solid #B22222;
        padding: 10px;
        border-radius: 10px;
        background-color: #f8f9fa;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rembg_session():
    return new_session("u2netp")

# --- 4. CÁC HÀM XỬ LÝ (CORE) ---

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

def set_auto_beauty():
    if 'auto_level' not in st.session_state: st.session_state.auto_level = 0
    current_level = st.session_state.auto_level
    next_level = (current_level + 1) % 3
    st.session_state.auto_level = next_level

    if next_level == 1:
        st.toast("✨ Auto Level 1: Tự nhiên")
        st.session_state.val_smooth = 8
        st.session_state.val_makeup = 5
        st.session_state.val_exposure = 1.05
        st.session_state.val_whites = 8
        st.session_state.val_blacks = 5
        st.session_state.val_sharp_amount = 5
        st.session_state.val_edge_soft = 2
    elif next_level == 2:
        st.toast("✨✨ Auto Level 2: Sáng đẹp")
        st.session_state.val_smooth = 15
        st.session_state.val_makeup = 10
        st.session_state.val_exposure = 1.10
        st.session_state.val_whites = 15
        st.session_state.val_blacks = 10
        st.session_state.val_sharp_amount = 10
        st.session_state.val_edge_soft = 4
    else:
        st.toast("🔄 Đã tắt Auto")
        reset_beauty_params()
        return
    
    st.session_state.val_contrast = 1.0
    st.session_state.val_temp = 0
    st.session_state.val_clarity = 0
    st.session_state.val_denoise = 0
    st.session_state.val_dehaze = 0

def resize_image_input(image, max_height=1000):
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

def process_raw_to_nobg(file_input):
    image = Image.open(file_input)
    image = resize_image_input(image, max_height=1000)
    session = get_rembg_session()
    no_bg_pil = remove(image, session=session, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10, alpha_matting_erode_size=5)
    no_bg_cv = cv2.cvtColor(np.array(no_bg_pil), cv2.COLOR_RGBA2BGRA)
    return no_bg_cv

# --- AI DETECTION (MEDIAPIPE) ---
def detect_face_mediapipe(image_cv, manual_angle=0):
    if not HAS_MEDIAPIPE:
        return None, 0, "Lỗi thư viện MediaPipe"

    img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2RGB)
    h, w = img_rgb.shape[:2]

    try:
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(img_rgb)
            if not results.detections:
                return None, 0, "Không tìm thấy mặt"

            detection = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            
            kp_right_eye = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
            kp_left_eye = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
            re_x, re_y = kp_right_eye.x * w, kp_right_eye.y * h
            le_x, le_y = kp_left_eye.x * w, kp_left_eye.y * h
            
            delta_y = re_y - le_y
            delta_x = re_x - le_x
            angle = np.degrees(np.arctan2(delta_y, delta_x))
            total_angle = angle + manual_angle
            return (x, y, bw, bh), total_angle, None
    except Exception as e:
        return None, 0, f"Lỗi xử lý AI: {str(e)}"

# --- CẮT ẢNH THÔNG MINH (V3) ---
def crop_final_image_v3(no_bg_img, manual_angle, target_ratio):
    try:
        if not HAS_MEDIAPIPE:
            h, w = no_bg_img.shape[:2]
            rotated = rotate_image(no_bg_img, manual_angle) if manual_angle != 0 else no_bg_img
            crop_h = int(h * 0.8)
            crop_w = int(crop_h * target_ratio)
            cy, cx = h // 2, w // 2
            x1 = max(0, cx - crop_w // 2)
            y1 = max(0, cy - crop_h // 2)
            img_pil = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGRA2RGBA))
            return img_pil.crop((x1, y1, x1+crop_w, y1+crop_h)), "⚠️ Chế độ thủ công (Không có AI)", manual_angle

        img_working = no_bg_img.copy()
        face_rect, auto_angle, err = detect_face_mediapipe(img_working, manual_angle)
        
        if err: return None, err, 0
        
        if abs(auto_angle) > 0.5:
            img_rotated = rotate_image(img_working, auto_angle)
        else:
            img_rotated = img_working

        face_rect_new, _, err_new = detect_face_mediapipe(img_rotated, 0)
        (x, y, w, h) = face_rect if err_new else face_rect_new

        # --- LOGIC CẮT ẢNH (ĐÃ THÊM ĐÀI LOAN) ---
        if target_ratio == 1.0: # 5x5 Visa Mỹ
            zoom_factor = 2.0; top_offset = 0.6 
        elif 0.77 <= target_ratio <= 0.78: # 3.5x4.5 (Úc, Hàn, Đài Loan, Châu Âu)
            zoom_factor = 1.8; top_offset = 0.55 
        elif 0.68 <= target_ratio <= 0.69: # 3.3x4.8 (Trung Quốc)
            zoom_factor = 1.9; top_offset = 0.55  
        elif target_ratio < 0.7: # 4x6 Thường
            zoom_factor = 2.2; top_offset = 0.5   
        else: # 3x4
            zoom_factor = 2.4; top_offset = 0.55

        crop_h = int(h * zoom_factor) 
        crop_w = int(crop_h * target_ratio)
        face_center_x = x + w // 2
        top_y = int(y - (h * (top_offset - 0.1))) 
        left_x = int(face_center_x - crop_w // 2)

        img_pil = Image.fromarray(cv2.cvtColor(img_rotated, cv2.COLOR_BGRA2RGBA))
        canvas = Image.new("RGBA", (crop_w, crop_h), (0,0,0,0))
        canvas.paste(img_pil, (-left_x, -top_y), img_pil)
        return canvas, f"Góc xoay AI: {auto_angle:.1f}°", auto_angle

    except Exception as e:
        return None, str(e), 0

# --- HẬU KỲ ---
def apply_transform(image, zoom=1.0, move_x=0, move_y=0):
    if zoom == 1.0 and move_x == 0 and move_y == 0: return image
    w, h = image.size
    new_w = int(w * zoom)
    new_h = int(h * zoom)
    img_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    center_x = (w - new_w) // 2
    center_y = (h - new_h) // 2
    paste_x = center_x + move_x
    paste_y = center_y + move_y
    canvas.paste(img_resized, (paste_x, paste_y), img_resized)
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
    lab_new = cv2.merge((l_new, a, b))
    return cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)

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
        d = 5; sigma = int(params['smooth'] * 2) + 10
        img_bgr = cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma, sigmaSpace=sigma)
    if params['dehaze'] > 0:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_c, a_c, b_c = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0 + (params['dehaze']/10.0), tileGridSize=(8,8))
        l_c = clahe.apply(l_c)
        img_bgr = cv2.cvtColor(cv2.merge((l_c, a_c, b_c)), cv2.COLOR_LAB2BGR)
    if params['temp'] != 0:
        temp = int(params['temp']); b_c, g_c, r_c = cv2.split(img_bgr)
        if temp > 0: r_c = cv2.add(r_c, temp); b_c = cv2.subtract(b_c, temp)
        else: r_c = cv2.add(r_c, temp); b_c = cv2.subtract(b_c, temp)
        img_bgr = cv2.merge([b_c, g_c, r_c])
    if params['makeup'] > 0:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_c, s_c, v_c = cv2.split(hsv)
        s_c = cv2.add(s_c, int(params['makeup'] * 1.5))
        v_c = cv2.add(v_c, int(params['makeup'] * 0.5))
        img_bgr = cv2.cvtColor(cv2.merge([h_c, s_c, v_c]), cv2.COLOR_HSV2BGR)
    if params['blacks'] > 0 or params['whites'] > 0:
        img_bgr = adjust_levels(img_bgr, params['blacks'], params['whites'])
    if params['clarity'] > 0: img_bgr = apply_clarity(img_bgr, params['clarity'])
    if params['sharp_amount'] > 0: img_bgr = apply_super_sharpen(img_bgr, params['sharp_amount'])

    final_bgra = cv2.merge([img_bgr[:,:,0], img_bgr[:,:,1], img_bgr[:,:,2], a])
    img_pil = Image.fromarray(cv2.cvtColor(final_bgra, cv2.COLOR_BGRA2RGBA))
    if params['exposure'] != 1.0: img_pil = ImageEnhance.Brightness(img_pil).enhance(params['exposure'])
    if params['contrast'] != 1.0: img_pil = ImageEnhance.Contrast(img_pil).enhance(params['contrast'])
    return img_pil

# --- IN ẤN ---
def create_pdf(img_person, size_type):
    if not HAS_FPDF: return None
    pdf = FPDF(orientation='P', unit='mm', format=(105, 148))
    pdf.add_page()
    temp_img_path = "temp_print.jpg"
    img_person.save(temp_img_path, quality=100, dpi=(300, 300))
    
    if "5x5" in size_type:
        w_mm, h_mm = 50, 50; cols, rows = 2, 2; margin_x, margin_y = 2, 5
    elif "3.5x4.5" in size_type: # Nhóm Visa 3.5x4.5
        w_mm, h_mm = 35, 45; cols, rows = 2, 3; margin_x, margin_y = 17, 6 
    elif "3.3x4.8" in size_type:
        w_mm, h_mm = 33, 48; cols, rows = 2, 2; margin_x, margin_y = 19, 20
    elif "4x6" in size_type:
        w_mm, h_mm = 40, 60; cols, rows = 2, 2; margin_x, margin_y = 10, 10
    else:
        w_mm, h_mm = 30, 40; cols, rows = 3, 3; margin_x, margin_y = 5, 10

    for r in range(rows):
        for c in range(cols):
            x = margin_x + c * (w_mm + 2); y = margin_y + r * (h_mm + 2)
            pdf.image(temp_img_path, x=x, y=y, w=w_mm, h=h_mm)
    return pdf.output(dest='S').encode('latin-1')

def create_print_layout_preview(img_person, size_type):
    PAPER_W_PX, PAPER_H_PX = 1240, 1748
    bg_paper = Image.new("RGB", (PAPER_W_PX, PAPER_H_PX), (255, 255, 255))
    
    if "5x5" in size_type: 
        target_w, target_h = 590, 590; rows, cols = 2, 2; start_x, start_y = 30, 200; gap = 30
    elif "3.5x4.5" in size_type: # Nhóm Visa 3.5x4.5
        target_w, target_h = 413, 531; rows, cols = 3, 2; start_x, start_y = 190, 80; gap = 40
    elif "3.3x4.8" in size_type: 
        target_w, target_h = 390, 567; rows, cols = 2, 2; start_x, start_y = 200, 250; gap = 40
    elif "4x6" in size_type:
        target_w, target_h = 472, 708; rows, cols = 2, 2; start_x, start_y = 120, 150; gap = 50
    else:
        target_w, target_h = 354, 472; rows, cols = 3, 3; start_x, start_y = 80, 120; gap = 40

    img_resized = img_person.resize((target_w, target_h), Image.Resampling.LANCZOS)
    for r in range(rows):
        for c in range(cols):
            x = start_x + c * (target_w + gap); y = start_y + r * (target_h + gap)
            bg_paper.paste(img_resized, (x, y))
    return bg_paper

# --- GIAO DIỆN CHÍNH ---
st.markdown('<div class="main-title">📸 STUDIO ẢNH THẺ PRO AI</div>', unsafe_allow_html=True)

if not HAS_MEDIAPIPE:
    st.error("⚠️ AI NHẬN DIỆN (MediaPipe) CHƯA ĐƯỢC KÍCH HOẠT. Chế độ Manual đang bật.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Thiết lập")
    input_method = st.radio("Nguồn ảnh:", ["📁 Tải ảnh lên", "📷 Chụp ảnh"], horizontal=True)
    input_file = st.file_uploader("Chọn file", type=['jpg','png','jpeg']) if input_method == "📁 Tải ảnh lên" else st.camera_input("Chụp ảnh ngay")

    st.markdown("---")
    # --- CẬP NHẬT MENU: THÊM ĐÀI LOAN VÀO NHÓM 3.5x4.5 ---
    size_option = st.radio("Chọn cỡ ảnh:", 
                         ["5x5 cm (Visa Mỹ)", 
                          "3.5x4.5 cm (Visa Úc/Hàn/Đài Loan/Âu)", # <-- Đã thêm Đài Loan
                          "3.3x4.8 cm (Visa Trung Quốc)", 
                          "4x6 cm (Hộ chiếu)", 
                          "3x4 cm (Giấy tờ)"])
    
    if "Visa Mỹ" in size_option: target_ratio = 1.0 
    elif "3.5x4.5" in size_option: target_ratio = 3.5/4.5 # Check theo kích thước cho chuẩn
    elif "Visa Trung Quốc" in size_option: target_ratio = 3.3/4.8
    elif "3x4" in size_option: target_ratio = 3/4
    else: target_ratio = 4/6
    
    bg_name = st.radio("Màu nền:", ["Trắng", "Xanh Chuẩn", "Xanh Nhạt"])
    bg_val = {"Trắng": (255,255,255,255), "Xanh Chuẩn": (66,135,245,255), "Xanh Nhạt": (135,206,250,255)}.get(bg_name)

# --- XỬ LÝ ---
if input_file:
    current_file_key = f"{input_file.name}_{input_file.size}"
    if 'current_file_key' not in st.session_state or st.session_state.current_file_key != current_file_key:
        if 'raw_nobg' in st.session_state: del st.session_state.raw_nobg
        if 'base' in st.session_state: del st.session_state.base
        gc.collect()
        with st.spinner('⏳ Đang tách nền AI...'):
            try:
                st.session_state.raw_nobg = process_raw_to_nobg(input_file)
                st.session_state.current_file_key = current_file_key
            except Exception as e: st.error(f"Lỗi tải ảnh: {e}")

# --- MAIN UI ---
col_btn1, col_btn2, col_space = st.columns([1.5, 1, 3])
with col_btn1:
    current_lvl = st.session_state.get('auto_level', 0)
    label_auto = f"✨ AUTO ĐẸP (Level {current_lvl})" if current_lvl > 0 else "✨ AUTO ĐẸP NGAY"
    st.button(label_auto, on_click=set_auto_beauty, type="primary", use_container_width=True)
with col_btn2:
    st.button("🔄 Làm lại", on_click=reset_beauty_params, use_container_width=True)

st.divider()
col_tools, col_result = st.columns([1, 1.2])

with col_tools:
    st.subheader("🎛️ Bảng điều khiển")
    manual_rot = st.slider("Góc nghiêng:", -15.0, 15.0, 0.0, 0.5)
    
    if 'raw_nobg' in st.session_state:
        final_crop, debug_info, _ = crop_final_image_v3(st.session_state.raw_nobg, manual_rot, target_ratio)
        if final_crop: st.session_state.base = final_crop; st.caption(f"ℹ️ {debug_info}")
        else: st.error(f"Lỗi crop: {debug_info}")

    tab1, tab2, tab3 = st.tabs(["🎨 Ánh sáng", "👩 Làm đẹp", "📐 Bố cục"])
    with tab1:
        p_exposure = st.slider("Độ sáng", 0.5, 1.5, st.session_state.get('val_exposure', 1.0), 0.05, key="val_exposure")
        p_contrast = st.slider("Tương phản", 0.5, 1.5, st.session_state.get('val_contrast', 1.0), 0.05, key="val_contrast")
        p_temp = st.slider("Nhiệt độ màu", -50, 50, st.session_state.get('val_temp', 0), key="val_temp")
        col_b, col_w = st.columns(2)
        with col_b: p_blacks = st.slider("Màu Đen", 0, 50, st.session_state.get('val_blacks', 0), key="val_blacks")
        with col_w: p_whites = st.slider("Màu Trắng", 0, 50, st.session_state.get('val_whites', 0), key="val_whites")
    with tab2:
        p_smooth = st.slider("Mịn da", 0, 30, st.session_state.get('val_smooth', 0), key="val_smooth")
        p_makeup = st.slider("Trang điểm", 0, 50, st.session_state.get('val_makeup', 0), key="val_makeup")
    with tab3:
        p_zoom = st.slider("Phóng/Thu", 0.5, 1.5, st.session_state.get('val_zoom', 1.0), 0.05, key="val_zoom")
        col_m1, col_m2 = st.columns(2)
        with col_m1: p_move_x = st.number_input("Ngang", -200, 200, st.session_state.get('val_move_x', 0), key="val_move_x")
        with col_m2: p_move_y = st.number_input("Dọc", -200, 200, st.session_state.get('val_move_y', 0), key="val_move_y")
        st.markdown("---")
        p_sharp = st.slider("Sắc nét", 0, 50, st.session_state.get('val_sharp_amount', 0), key="val_sharp_amount")
        p_clarity = st.slider("Chi tiết", 0, 50, st.session_state.get('val_clarity', 0), key="val_clarity")
        p_denoise = st.slider("Khử nhiễu", 0, 20, st.session_state.get('val_denoise', 0), key="val_denoise")
        p_dehaze = st.slider("Khử sương", 0, 30, st.session_state.get('val_dehaze', 0), key="val_dehaze")
        p_edge = st.slider("Mềm biên", 0, 10, st.session_state.get('val_edge_soft', 0), key="val_edge_soft")

    params = {'smooth': p_smooth, 'makeup': p_makeup, 'exposure': p_exposure, 'contrast': p_contrast, 'temp': p_temp, 'sharp_amount': p_sharp, 'clarity': p_clarity, 'dehaze': p_dehaze, 'blacks': p_blacks, 'whites': p_whites, 'denoise': p_denoise, 'zoom': p_zoom, 'move_x': p_move_x, 'move_y': p_move_y, 'edge_soft': p_edge}

with col_result:
    if 'base' in st.session_state and st.session_state.base:
        with st.spinner("🚀 Đang xử lý..."):
            final_person = apply_advanced_effects(st.session_state.base, params)
        
        w, h = final_person.size
        final_img = Image.new("RGBA", (w, h), bg_val)
        final_img.paste(final_person, (0, 0), final_person)
        final_rgb = final_img.convert("RGB")

        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(final_rgb, caption=f"KẾT QUẢ: {size_option}", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📥 Xuất file")
        d_tab1, d_tab2 = st.tabs(["Lưu Ảnh", "In Ấn"])
        with d_tab1:
            buf = io.BytesIO(); final_rgb.save(buf, format="JPEG", quality=95, dpi=(300, 300))
            st.download_button("⬇️ Tải Ảnh JPG", data=buf.getvalue(), file_name="anh_the_pro.jpg", mime="image/jpeg", type="primary", use_container_width=True)
        with d_tab2:
            st.image(create_print_layout_preview(final_rgb, size_option), caption="Preview In A6", use_container_width=True)
            if HAS_FPDF:
                st.download_button("📄 Tải PDF In", data=create_pdf(final_rgb, size_option), file_name="file_in.pdf", mime="application/pdf", use_container_width=True)
    else:
        st.info("👈 Chọn ảnh bên trái để bắt đầu")
