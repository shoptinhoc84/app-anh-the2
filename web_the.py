import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc

# --- XỬ LÝ LỖI THƯ VIỆN FPDF ---
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# --- 1. CẤU HÌNH TRANG & CSS TRANG TRÍ ---
st.set_page_config(page_title="Studio Ảnh Thẻ STH", layout="wide", page_icon="📸")

# CSS làm đẹp giao diện
st.markdown("""
<style>
    /* Làm đẹp tiêu đề */
    .main-title {
        font-size: 2.5rem;
        color: #4B0082;
        text-align: center;
        font-weight: 800;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    /* Làm nổi bật nút Auto */
    div[data-testid="stButton"] > button:first-child {
        border-radius: 10px;
        font-weight: bold;
    }
    /* Đóng khung ảnh kết quả */
    .image-container {
        border: 3px solid #4B0082;
        padding: 10px;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rembg_session():
    return new_session("u2netp")

# --- 2. LOGIC HÀM ---
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

# --- HÀM MỚI: XỬ LÝ PRESET NAM/NỮ (SỬA LỖI CRASH) ---
def apply_gender_preset():
    # Chỉ chạy khi key 'gender_radio' đã tồn tại
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
            st.toast("👨 Đã áp dụng mẫu Nam")
        else:
            st.session_state.val_smooth = 25
            st.session_state.val_makeup = 20
            st.session_state.val_exposure = 1.1
            st.session_state.val_contrast = 1.05
            st.session_state.val_sharp_amount = 10
            st.session_state.val_clarity = 5
            st.session_state.val_denoise = 10
            st.session_state.val_whites = 15
            st.toast("👩 Đã áp dụng mẫu Nữ")

def set_auto_beauty():
    if 'auto_level' not in st.session_state:
        st.session_state.auto_level = 0
    current_level = st.session_state.auto_level
    next_level = (current_level + 1) % 3
    st.session_state.auto_level = next_level

    if next_level == 1:
        st.toast("✨ Auto Level 1: Nhẹ nhàng")
        st.session_state.val_smooth = 5
        st.session_state.val_makeup = 2
        st.session_state.val_exposure = 1.05
        st.session_state.val_whites = 6
        st.session_state.val_blacks = 4
        st.session_state.val_sharp_amount = 2
        st.session_state.val_edge_soft = 2
    elif next_level == 2:
        st.toast("✨✨ Auto Level 2: Rực rỡ")
        st.session_state.val_smooth = 10
        st.session_state.val_makeup = 4
        st.session_state.val_exposure = 1.10
        st.session_state.val_whites = 12
        st.session_state.val_blacks = 8
        st.session_state.val_sharp_amount = 4
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

def process_raw_to_nobg(file_input):
    image = Image.open(file_input)
    image = resize_image_input(image, max_height=1200)
    session = get_rembg_session()
    no_bg_pil = remove(image, session=session, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10, alpha_matting_erode_size=10)
    no_bg_cv = cv2.cvtColor(np.array(no_bg_pil), cv2.COLOR_RGBA2BGRA)
    return no_bg_cv

def crop_final_image(no_bg_img, manual_angle, target_ratio):
    try:
        img_working = no_bg_img.copy()
        gray = cv2.cvtColor(img_working, cv2.COLOR_BGRA2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0: return None, "Không tìm thấy khuôn mặt", 0

        face_rect = max(faces, key=lambda f: f[2] * f[3])
        auto_angle = get_face_angle(gray, face_rect)
        if abs(auto_angle) < 1.0 or abs(auto_angle) > 20.0: auto_angle = 0.0 

        total_angle = auto_angle + manual_angle
        img_rotated = rotate_image(img_working, total_angle) if abs(total_angle) > 0.1 else img_working

        gray_new = cv2.cvtColor(img_rotated, cv2.COLOR_BGRA2GRAY)
        faces_new = face_cascade.detectMultiScale(gray_new, 1.1, 5)
        (x, y, w, h) = max(faces_new, key=lambda f: f[2] * f[3]) if len(faces_new) > 0 else face_rect

        if target_ratio == 1.0: 
            zoom_factor = 1.8  
            top_offset = 0.55 
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
        return canvas, f"Góc Auto: {auto_angle:.1f}°", total_angle
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
        d = 5
        sigma = int(params['smooth'] * 2) + 10
        img_bgr = cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma, sigmaSpace=sigma)
    if params['dehaze'] > 0:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_c, a_c, b_c = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0 + (params['dehaze']/10.0), tileGridSize=(8,8))
        l_c = clahe.apply(l_c)
        img_bgr = cv2.cvtColor(cv2.merge((l_c, a_c, b_c)), cv2.COLOR_LAB2BGR)
    if params['temp'] != 0:
        temp = int(params['temp'])
        b_c, g_c, r_c = cv2.split(img_bgr)
        if temp > 0:
            r_c = cv2.add(r_c, temp)
            b_c = cv2.subtract(b_c, temp)
        else:
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
    pdf = FPDF(orientation='P', unit='mm', format=(105, 148))
    pdf.add_page()
    temp_img_path = "temp_print.jpg"
    img_person.save(temp_img_path, quality=100, dpi=(300, 300))
    
    if "5x5" in size_type:
        w_mm, h_mm = 50, 50
        cols, rows = 2, 2
        margin_x, margin_y = 2, 5
    elif "4x6" in size_type:
        w_mm, h_mm = 40, 60
        cols, rows = 2, 2
        margin_x, margin_y = 10, 10
    else: 
        w_mm, h_mm = 30, 40
        cols, rows = 3, 3
        margin_x, margin_y = 5, 10

    for r in range(rows):
        for c in range(cols):
            x = margin_x + c * (w_mm + 2)
            y = margin_y + r * (h_mm + 2)
            pdf.image(temp_img_path, x=x, y=y, w=w_mm, h=h_mm)
    return pdf.output(dest='S').encode('latin-1')

def create_print_layout_preview(img_person, size_type):
    PAPER_W, PAPER_H = 1748, 1181 
    bg_paper = Image.new("RGB", (PAPER_W, PAPER_H), (255, 255, 255))
    if "5x5" in size_type: 
        target_w, target_h = 600, 600
        rows, cols = 1, 2
        start_x, start_y = 200, 290 
        gap = 100
    elif "4x6" in size_type:
        target_w, target_h = 472, 708
        rows, cols = 1, 3
        start_x, start_y = 100, 200
        gap = 50
    else:
        target_w, target_h = 354, 472
        rows, cols = 2, 4
        start_x, start_y = 100, 100
        gap = 40
    img_resized = img_person.resize((target_w, target_h), Image.Resampling.LANCZOS)
    for r in range(rows):
        for c in range(cols):
            x = start_x + c * (target_w + gap)
            y = start_y + r * (target_h + gap)
            bg_paper.paste(img_resized, (x, y))
    return bg_paper

# --- 3. GIAO DIỆN CHÍNH (ĐÃ SẮP XẾP LẠI) ---

st.markdown('<div class="main-title">📸 ẢNH THẺ SHOPTINHOC</div>', unsafe_allow_html=True)
if not HAS_FPDF:
    st.warning("⚠️ Chưa cài thư viện in ấn. Chạy: `pip install fpdf`")

# --- A. THANH BÊN (SIDEBAR) - CÀI ĐẶT ĐẦU VÀO ---
with st.sidebar:
    st.header("⚙️ Thiết lập Đầu vào")
    st.info("Bước 1: Chọn ảnh và loại ảnh")
    
    input_method = st.radio("Nguồn ảnh:", ["📁 Tải ảnh lên", "📷 Chụp ảnh"], horizontal=True)
    input_file = None
    if input_method == "📁 Tải ảnh lên":
        input_file = st.file_uploader("Chọn file (JPG, PNG)", type=['jpg', 'png', 'jpeg'])
    else:
        input_file = st.camera_input("Chụp ảnh ngay")

    st.markdown("---")
    st.subheader("Kích thước & Phông nền")
    size_option = st.radio("Chọn cỡ ảnh:", ["5x5 cm (Visa Mỹ)", "4x6 cm (Hộ chiếu)", "3x4 cm (Giấy tờ)"])
    if "Visa Mỹ" in size_option: target_ratio = 1.0 
    elif "3x4" in size_option: target_ratio = 3/4
    else: target_ratio = 4/6
    
    bg_name = st.radio("Màu nền:", ["Trắng", "Xanh Chuẩn", "Xanh Nhạt"])
    bg_map = {"Trắng": (255, 255, 255, 255), "Xanh Chuẩn": (66, 135, 245, 255), "Xanh Nhạt": (135, 206, 250, 255)}
    bg_val = bg_map.get(bg_name)
    
    st.markdown("---")
    st.caption("Phiên bản V2.16 - Fix UI Error")

# --- B. XỬ LÝ ẢNH ĐẦU VÀO ---
if input_file:
    current_file_key = f"{input_file.name}_{input_file.size}"
    if 'current_file_key' in st.session_state and st.session_state.current_file_key != current_file_key:
        if 'raw_nobg' in st.session_state: del st.session_state.raw_nobg
        if 'base' in st.session_state: del st.session_state.base
        gc.collect()

    if 'current_file_key' not in st.session_state or st.session_state.current_file_key != current_file_key:
        with st.spinner('⏳ Đang tách nền AI...'):
            try:
                st.session_state.raw_nobg = process_raw_to_nobg(input_file)
                st.session_state.current_file_key = current_file_key
            except Exception as e: st.error(f"Lỗi tải ảnh: {e}")

# --- C. GIAO DIỆN CHÍNH (MAIN COLUMN) ---

# Khu vực nút bấm lớn
col_btn1, col_btn2, col_space = st.columns([1.5, 1, 3])
with col_btn1:
    current_lvl = st.session_state.get('auto_level', 0)
    label_auto = f"✨ AUTO ĐẸP (Level {current_lvl})" if current_lvl > 0 else "✨ AUTO ĐẸP NGAY"
    # Nút bấm Auto
    st.button(label_auto, on_click=set_auto_beauty, type="primary", use_container_width=True, help="Bấm để tự động làm đẹp")

with col_btn2:
    # Nút Reset
    st.button("🔄 Làm lại", on_click=reset_beauty_params, use_container_width=True)

st.divider()

# Chia cột: Bên trái là Công cụ chỉnh, Bên phải là Ảnh
col_tools, col_result = st.columns([1, 1.2])

with col_tools:
    st.subheader("🎛️ Bảng điều khiển")
    
    # Góc xoay thủ công (Để riêng ở trên cho dễ thấy)
    manual_rot = st.slider("Góc nghiêng đầu:", -15.0, 15.0, 0.0, 0.5)
    if 'raw_nobg' in st.session_state:
        final_crop, debug_info, _ = crop_final_image(st.session_state.raw_nobg, manual_rot, target_ratio)
        if final_crop: st.session_state.base = final_crop
        else: st.error(f"Lỗi: {debug_info}")

    # Sắp xếp Slider vào TAB cho gọn
    tab1, tab2, tab3 = st.tabs(["🎨 Màu & Ánh sáng", "👩 Khuôn mặt", "📐 Bố cục & Nét"])
    
    with tab1:
        st.caption("Chỉnh độ sáng và màu sắc")
        p_exposure = st.slider("Độ sáng", 0.5, 1.5, st.session_state.get('val_exposure', 1.0), 0.05, key="val_exposure")
        p_contrast = st.slider("Tương phản", 0.5, 1.5, st.session_state.get('val_contrast', 1.0), 0.05, key="val_contrast")
        p_temp = st.slider("Nhiệt độ màu (Ấm/Lạnh)", -50, 50, st.session_state.get('val_temp', 0), key="val_temp")
        col_b, col_w = st.columns(2)
        with col_b: p_blacks = st.slider("Màu Đen", 0, 50, st.session_state.get('val_blacks', 0), key="val_blacks")
        with col_w: p_whites = st.slider("Màu Trắng", 0, 50, st.session_state.get('val_whites', 0), key="val_whites")

    with tab2:
        st.caption("Làm đẹp da")
        p_smooth = st.slider("Mịn da", 0, 30, st.session_state.get('val_smooth', 0), key="val_smooth")
        p_makeup = st.slider("Trang điểm/Hồng hào", 0, 50, st.session_state.get('val_makeup', 0), key="val_makeup")
        st.markdown("---")
        
        # --- SỬA LỖI Ở ĐÂY: DÙNG CALLBACK CHO RADIO BUTTON ---
        ai_enabled = st.checkbox("Dùng Preset AI (Nam/Nữ)", key='ai_enabled')
        if ai_enabled:
            # Khi người dùng chọn, nó sẽ gọi hàm 'apply_gender_preset' ở trên
            gender_style = st.radio("Chọn giới tính:", ["Nam", "Nữ"], 
                                  horizontal=True, 
                                  key="gender_radio", 
                                  on_change=apply_gender_preset)
            # Đã xóa đoạn code if/else gây lỗi

    with tab3:
        st.caption("Căn chỉnh vị trí và độ nét")
        p_zoom = st.slider("Phóng to/Thu nhỏ", 0.5, 1.5, st.session_state.get('val_zoom', 1.0), 0.05, key="val_zoom")
        col_m1, col_m2 = st.columns(2)
        with col_m1: p_move_x = st.number_input("Dịch Ngang", -100, 100, st.session_state.get('val_move_x', 0), key="val_move_x")
        with col_m2: p_move_y = st.number_input("Dịch Dọc", -100, 100, st.session_state.get('val_move_y', 0), key="val_move_y")
        
        st.markdown("---")
        p_sharp_amount = st.slider("Độ sắc nét", 0, 50, st.session_state.get('val_sharp_amount', 0), key="val_sharp_amount")
        p_clarity = st.slider("Chi tiết (Clarity)", 0, 50, st.session_state.get('val_clarity', 0), key="val_clarity")
        p_denoise = st.slider("Giảm nhiễu (Denoise)", 0, 20, st.session_state.get('val_denoise', 0), key="val_denoise")
        p_dehaze = st.slider("Khử sương mù", 0, 30, st.session_state.get('val_dehaze', 0), key="val_dehaze")
        p_edge_soft = st.slider("Làm mềm biên", 0, 10, st.session_state.get('val_edge_soft', 0), key="val_edge_soft")

    params = {
        'smooth': p_smooth, 'makeup': p_makeup,
        'exposure': p_exposure, 'contrast': p_contrast, 'temp': p_temp,
        'sharp_amount': p_sharp_amount, 'clarity': p_clarity, 
        'dehaze': p_dehaze, 'blacks': p_blacks, 'whites': p_whites, 'denoise': p_denoise,
        'zoom': p_zoom, 'move_x': p_move_x, 'move_y': p_move_y,
        'edge_soft': p_edge_soft
    }

# --- D. HIỂN THỊ KẾT QUẢ ---
with col_result:
    if 'base' in st.session_state and st.session_state.base:
        # Xử lý ảnh
        with st.spinner("🚀 Đang xử lý ảnh..."):
            final_person = apply_advanced_effects(st.session_state.base, params)
        
        w, h = final_person.size
        final_img = Image.new("RGBA", (w, h), bg_val)
        final_img.paste(final_person, (0, 0), final_person)
        final_rgb = final_img.convert("RGB")

        # Hiển thị ảnh trong khung
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(final_rgb, caption=f"KẾT QUẢ: {size_option}", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📥 Tải về & In ấn")
        
        # Tabs cho phần download
        d_tab1, d_tab2 = st.tabs(["Lưu Ảnh (JPG)", "In Ấn (PDF)"])
        
        with d_tab1:
            buf = io.BytesIO()
            final_rgb.save(buf, format="JPEG", quality=95, dpi=(300, 300))
            safe_bg_name = {"Trắng": "white", "Xanh Chuẩn": "blue_standard", "Xanh Nhạt": "blue_light"}.get(bg_name, "custom")
            st.download_button(label="⬇️ Tải Ảnh JPG Chất Lượng Cao", data=buf.getvalue(), file_name=f"anh_the_{safe_bg_name}.jpg", mime="image/jpeg", type="primary", use_container_width=True)

        with d_tab2:
            st.image(create_print_layout_preview(final_rgb, size_option), caption="Xem trước bản in (Khổ A6)", use_container_width=True)
            if HAS_FPDF:
                pdf_data = create_pdf(final_rgb, size_option)
                st.download_button(label="📄 Tải File PDF để in", data=pdf_data, file_name="file_in_anh_the.pdf", mime="application/pdf", use_container_width=True)
            else:
                st.error("Thiếu thư viện fpdf.")
        
        # So sánh (ẩn trong expander cho gọn)
        with st.expander("👁️ So sánh Trước / Sau"):
            c_before, c_after = st.columns(2)
            with c_before: st.image(st.session_state.base, caption="Gốc")
            with c_after: st.image(final_rgb, caption="Sau chỉnh sửa")

    else:
        # Màn hình chờ khi chưa có ảnh
        st.info("👈 Mời bạn chọn ảnh ở cột bên trái để bắt đầu.")
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)

