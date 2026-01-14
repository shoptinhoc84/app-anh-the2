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

# --- 1. CẤU HÌNH & CACHE ---
st.set_page_config(page_title="Studio Ảnh Thẻ V2.16 - Auto Remove", layout="wide")

@st.cache_resource
def get_rembg_session():
    return new_session("u2netp")

st.title("📸 Studio Ảnh Thẻ - V2.16 (Xóa Mụn/Xăm)")
if not HAS_FPDF:
    st.warning("⚠️ Bạn chưa cài thư viện xuất PDF. Hãy chạy lệnh: `pip install fpdf` để mở khóa tính năng in.")
st.markdown("---")

# --- 2. HÀM RESET & AUTO ĐẸP ---

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
    # Reset params mới
    st.session_state.val_remove_blemish = False
    st.session_state.val_blemish_threshold = 20

def set_auto_beauty():
    if 'auto_level' not in st.session_state:
        st.session_state.auto_level = 0
    
    current_level = st.session_state.auto_level
    next_level = (current_level + 1) % 3
    st.session_state.auto_level = next_level

    if next_level == 1:
        st.toast("✨ Auto Level 1: Đẹp tự nhiên")
        st.session_state.val_smooth = 5
        st.session_state.val_makeup = 2
        st.session_state.val_exposure = 1.05
        st.session_state.val_whites = 6
        st.session_state.val_blacks = 4
        st.session_state.val_sharp_amount = 2
        st.session_state.val_edge_soft = 2
        
    elif next_level == 2:
        st.toast("✨✨ Auto Level 2: Đẹp rực rỡ (x2)")
        st.session_state.val_smooth = 10
        st.session_state.val_makeup = 4
        st.session_state.val_exposure = 1.10
        st.session_state.val_whites = 12
        st.session_state.val_blacks = 8
        st.session_state.val_sharp_amount = 4
        st.session_state.val_edge_soft = 4
        
    else:
        st.toast("🔄 Reset Default")
        reset_beauty_params()
        return

    st.session_state.val_contrast = 1.0
    st.session_state.val_temp = 0
    st.session_state.val_clarity = 0
    st.session_state.val_denoise = 0
    st.session_state.val_dehaze = 0
    # Không tự động bật xóa mụn để tránh xóa nhầm nốt ruồi duyên
    st.session_state.val_remove_blemish = False 

# --- 3. CÁC HÀM XỬ LÝ ẢNH CỐT LÕI ---

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

# --- 4. TÍNH NĂNG TRANSFORM & BIẾN ĐỔI ---

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

# --- 5. TÍNH NĂNG XÓA MỤN (BLEMISH REMOVAL) ---

def apply_blemish_removal(image_bgr, threshold=20):
    """
    Tự động xóa mụn/vết thâm bằng BlackHat Morphology + Inpainting
    """
    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. BlackHat để tìm các điểm tối trên nền sáng (đặc trưng của mụn/vết thâm)
    # Kernel size 9x9 -> 15x15 phù hợp cho các đốm nhỏ/trung bình
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 3. Tạo mask dựa trên ngưỡng sáng (threshold)
    # Các điểm có độ chênh lệch sáng > threshold sẽ được chọn
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    
    # 4. Nở vùng mask ra một chút để bao trùm vết thâm
    mask = cv2.dilate(mask, None, iterations=3)
    
    # 5. Inpainting (Vẽ bù đắp) vào các vùng mask
    inpainted = cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)
    
    return inpainted

# --- 6. BỘ LỌC NÂNG CAO ---

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
    # 1. Transform & Soft Edge
    img_transformed = apply_transform(base_img, params['zoom'], params['move_x'], params['move_y'])
    if params['edge_soft'] > 0:
        img_transformed = apply_edge_softness(img_transformed, params['edge_soft'])

    img_bgra = cv2.cvtColor(np.array(img_transformed), cv2.COLOR_RGBA2BGRA)
    b, g, r, a = cv2.split(img_bgra)
    img_bgr = cv2.merge([b, g, r])
    
    # --- MỚI: XÓA MỤN / HÌNH XĂM (Xử lý trên BGR trước khi chỉnh màu) ---
    if params['remove_blemish']:
        # Áp dụng xóa khuyết điểm
        img_bgr = apply_blemish_removal(img_bgr, params['blemish_threshold'])

    # 2. Hiệu ứng khác
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

# --- 7. GIAO DIỆN CHÍNH ---
col1, col2 = st.columns([1, 2.2])

with col1:
    st.header("🛠 Thiết lập")
    input_method = st.radio("Nguồn ảnh:", ["📁 Tải ảnh lên", "📷 Chụp ảnh"], horizontal=True)
    input_file = None
    if input_method == "📁 Tải ảnh lên":
        input_file = st.file_uploader("Chọn ảnh gốc", type=['jpg', 'png', 'jpeg'])
    else:
        input_file = st.camera_input("Chụp ảnh ngay")

    st.subheader("2. Loại ảnh")
    size_option = st.radio("Kích thước:", ["5x5 cm (Visa Mỹ)", "4x6 cm (Hộ chiếu)", "3x4 cm (Giấy tờ)"])
    if "Visa Mỹ" in size_option: target_ratio = 1.0 
    elif "3x4" in size_option: target_ratio = 3/4
    else: target_ratio = 4/6
    
    manual_rot = st.slider("Chỉnh nghiêng đầu:", -15.0, 15.0, 0.0, 0.5)
    
    bg_name = st.radio("Màu nền:", ["Trắng", "Xanh Chuẩn", "Xanh Nhạt"], horizontal=True)
    if "Visa Mỹ" in size_option and bg_name != "Trắng": st.warning("⚠️ Visa Mỹ bắt buộc nền TRẮNG.")
    bg_map = {"Trắng": (255, 255, 255, 255), "Xanh Chuẩn": (66, 135, 245, 255), "Xanh Nhạt": (135, 206, 250, 255)}
    bg_val = bg_map.get(bg_name)

    if input_file:
        current_file_key = f"{input_file.name}_{input_file.size}"
        if 'current_file_key' in st.session_state and st.session_state.current_file_key != current_file_key:
            if 'raw_nobg' in st.session_state: del st.session_state.raw_nobg
            if 'base' in st.session_state: del st.session_state.base
            gc.collect()

        if 'current_file_key' not in st.session_state or st.session_state.current_file_key != current_file_key:
            with st.spinner('Đang tách nền...'):
                try:
                    st.session_state.raw_nobg = process_raw_to_nobg(input_file)
                    st.session_state.current_file_key = current_file_key
                except Exception as e: st.error(f"Lỗi tải ảnh: {e}")
        
        if 'raw_nobg' in st.session_state:
            final_crop, debug_info, _ = crop_final_image(st.session_state.raw_nobg, manual_rot, target_ratio)
            if final_crop: st.session_state.base = final_crop
            else: st.error(f"Lỗi: {debug_info}")

    st.markdown("---")
    
    # --- PHẦN NÚT BẤM ---
    c_head, c_btn = st.columns([3, 2])
    with c_head: st.subheader("3. Chỉnh sửa")
    with c_btn: 
        b1, b2 = st.columns(2)
        current_lvl = st.session_state.get('auto_level', 0)
        label_auto = f"✨ Auto (Lv {current_lvl})" if current_lvl > 0 else "✨ Auto Đẹp"
        with b1: st.button(label_auto, on_click=set_auto_beauty, help="Click 1: Đẹp nhẹ - Click 2: Đẹp x2 - Click 3: Tắt")
        with b2: st.button("🔄 Reset", on_click=reset_beauty_params)

    with st.expander("🤖 AI Style (Tự động)", expanded=False):
        ai_enabled = st.checkbox("Bật chế độ AI Preset", key='ai_enabled')
        if ai_enabled:
            gender_style = st.radio("Phong cách:", ["Nam", "Nữ"])
            if gender_style == "Nam":
                st.session_state.val_smooth = 5
                st.session_state.val_makeup = 2
                st.session_state.val_exposure = 1.05
                st.session_state.val_contrast = 1.15
                st.session_state.val_sharp_amount = 20
                st.session_state.val_clarity = 15
                st.session_state.val_denoise = 5
                st.session_state.val_blacks = 10
                st.session_state.val_whites = 5
            else:
                st.session_state.val_smooth = 25
                st.session_state.val_makeup = 20
                st.session_state.val_exposure = 1.1
                st.session_state.val_contrast = 1.05
                st.session_state.val_sharp_amount = 10
                st.session_state.val_clarity = 5
                st.session_state.val_denoise = 10
                st.session_state.val_whites = 15
    
    with st.expander("📐 4. Bố cục (Ctrl + T)", expanded=True):
        p_zoom = st.slider("Phóng to / Thu nhỏ", 0.5, 1.5, st.session_state.get('val_zoom', 1.0), 0.05, key="val_zoom")
        p_move_x = st.slider("↔️ Dịch Trái / Phải", -100, 100, st.session_state.get('val_move_x', 0), 1, key="val_move_x")
        p_move_y = st.slider("↕️ Dịch Lên / Xuống", -100, 100, st.session_state.get('val_move_y', 0), 1, key="val_move_y")

    # --- MỤC XÓA MỤN MỚI ---
    with st.expander("🔍 Xóa mụn & Khuyết điểm", expanded=True):
        col_m1, col_m2 = st.columns([1, 1.5])
        with col_m1:
            p_remove_blemish = st.checkbox("Bật xóa tự động", value=st.session_state.get('val_remove_blemish', False), key='val_remove_blemish')
        with col_m2:
            p_blemish_threshold = st.slider("Ngưỡng phát hiện", 5, 50, st.session_state.get('val_blemish_threshold', 20), 1, key="val_blemish_threshold", help="Càng cao càng xóa mạnh (dễ xóa nhầm)")

    with st.expander("✨ 5. Công cụ chỉnh màu", expanded=True):
        st.markdown("**Chi tiết & Độ nét**")
        p_sharp_amount = st.slider("Độ sắc nét", 0, 50, st.session_state.get('val_sharp_amount', 0), key="val_sharp_amount")
        p_edge_soft = st.slider("Làm mềm viền tóc (Anti-Alias)", 0, 10, st.session_state.get('val_edge_soft', 0), 1, key="val_edge_soft")
        
        p_clarity = st.slider("Độ rõ nét (Clarity)", 0, 50, st.session_state.get('val_clarity', 0), key="val_clarity")
        p_dehaze = st.slider("Xóa lớp phủ mờ", 0, 30, st.session_state.get('val_dehaze', 0), key="val_dehaze")
        p_denoise = st.slider("Giảm nhiễu hạt", 0, 20, st.session_state.get('val_denoise', 0), key="val_denoise")

        st.markdown("**Ánh sáng & Màu sắc**")
        col_b, col_w = st.columns(2)
        with col_b: p_blacks = st.slider("Làm sâu màu Đen", 0, 50, st.session_state.get('val_blacks', 0), key="val_blacks")
        with col_w: p_whites = st.slider("Làm rực màu Trắng", 0, 50, st.session_state.get('val_whites', 0), key="val_whites")
            
        p_exposure = st.slider("Độ sáng tổng", 0.5, 1.5, st.session_state.get('val_exposure', 1.0), 0.05, key="val_exposure")
        p_contrast = st.slider("Tương phản", 0.5, 1.5, st.session_state.get('val_contrast', 1.0), 0.05, key="val_contrast")
        
        st.markdown("**Da & Trang điểm**")
        p_smooth = st.slider("Mịn da", 0, 30, st.session_state.get('val_smooth', 0), key="val_smooth")
        p_makeup = st.slider("Hồng hào", 0, 50, st.session_state.get('val_makeup', 0), key="val_makeup")
        p_temp = st.slider("Nhiệt độ màu", -50, 50, st.session_state.get('val_temp', 0), key="val_temp")

    params = {
        'smooth': p_smooth, 'makeup': p_makeup,
        'exposure': p_exposure, 'contrast': p_contrast, 'temp': p_temp,
        'sharp_amount': p_sharp_amount, 'clarity': p_clarity, 
        'dehaze': p_dehaze, 'blacks': p_blacks, 'whites': p_whites, 'denoise': p_denoise,
        'zoom': p_zoom, 'move_x': p_move_x, 'move_y': p_move_y,
        'edge_soft': p_edge_soft,
        'remove_blemish': p_remove_blemish,
        'blemish_threshold': p_blemish_threshold
    }

with col2:
    st.header(f"🖼 Kết quả ({size_option})")
    
    # --- TÍNH NĂNG SO SÁNH (COMPARE) ---
    show_compare = st.checkbox("👁️ Xem Trước / Sau (So sánh)", value=False)

    if 'base' in st.session_state and st.session_state.base:
        try:
            with st.spinner("Đang xử lý..."):
                final_person = apply_advanced_effects(st.session_state.base, params)
            
            w, h = final_person.size
            final_img = Image.new("RGBA", (w, h), bg_val)
            final_img.paste(final_person, (0, 0), final_person)
            final_rgb = final_img.convert("RGB")
            
            tab1, tab2 = st.tabs(["Ảnh Kết Quả", "Layout In"])
            
            with tab1:
                if show_compare:
                    c_before, c_after = st.columns(2)
                    with c_before:
                        st.image(st.session_state.base, caption="Gốc (Tách nền)", use_container_width=True)
                    with c_after:
                        st.image(final_rgb, caption="Đã chỉnh sửa", use_container_width=True)
                else:
                    st.image(final_rgb, width=350, caption="Ảnh hoàn thiện")
                
                st.markdown("---")
                c1, c2 = st.columns(2)
                buf = io.BytesIO()
                final_rgb.save(buf, format="JPEG", quality=95, dpi=(300, 300))
                safe_bg_name = {"Trắng": "white", "Xanh Chuẩn": "blue_standard", "Xanh Nhạt": "blue_light"}.get(bg_name, "custom")
                c1.download_button(label="⬇️ Tải Ảnh JPG", data=buf.getvalue(), file_name=f"ket_qua_{safe_bg_name}.jpg", mime="image/jpeg")
            
            with tab2:
                st.caption("Xem trước khi in (A6)")
                preview_paper = create_print_layout_preview(final_rgb, size_option)
                st.image(preview_paper, use_container_width=True)
                
                if HAS_FPDF:
                    pdf_data = create_pdf(final_rgb, size_option)
                    st.download_button(label="📄 Tải File PDF (In Chuẩn)", data=pdf_data, file_name="file_in_anh_the.pdf", mime="application/pdf")
                else:
                    st.error("Chức năng PDF đang tắt do thiếu thư viện `fpdf`.")

        except Exception as e:
            st.error(f"Lỗi: {e}. Thử Reset hoặc chọn ảnh khác.")
    else:
        st.info("👈 Hãy chọn ảnh ở cột bên trái để bắt đầu xử lý.")
