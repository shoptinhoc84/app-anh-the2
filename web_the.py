import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from rembg import remove, new_session
import io
import math

# --- 1. CẤU HÌNH & CACHE ---
st.set_page_config(page_title="Studio Ảnh Thẻ Pro Max", layout="wide")

@st.cache_resource
def get_rembg_session():
    return new_session("u2netp")

st.title("📸 Studio Ảnh Thẻ - Pro Max (STH)")
st.markdown("---")

# --- 2. HÀM RESET (MỚI) ---
def reset_beauty_params():
    """Đưa toàn bộ thông số làm đẹp về mặc định"""
    st.session_state.val_smooth = 0
    st.session_state.val_makeup = 0
    st.session_state.val_exposure = 1.0
    st.session_state.val_contrast = 1.0
    st.session_state.val_temp = 0
    st.session_state.val_sharp = 0
    st.session_state.val_dehaze = 0

# --- 3. CÁC HÀM XỬ LÝ ẢNH CỐT LÕI (GIỮ NGUYÊN BẢN ỔN ĐỊNH) ---

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def get_face_angle(gray_img, face_rect):
    (x, y, w, h) = face_rect
    roi_gray = gray_img[y:y+h, x:x+w]
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        (ex1, ey1, ew1, eh1) = eyes[0]
        (ex2, ey2, ew2, eh2) = eyes[-1]
        
        p1 = (ex1 + ew1//2, ey1 + eh1//2)
        p2 = (ex2 + ew2//2, ey2 + eh2//2)
        
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        
        if delta_x < w/4: return 0.0
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        return angle
    return 0.0

def process_raw_to_nobg(uploaded_file):
    image = Image.open(uploaded_file)
    session = get_rembg_session()
    no_bg_pil = remove(image, session=session)
    no_bg_cv = cv2.cvtColor(np.array(no_bg_pil), cv2.COLOR_RGBA2BGRA)
    return no_bg_cv

def crop_final_image(no_bg_img, manual_angle, target_ratio):
    try:
        img_working = no_bg_img.copy()
        gray = cv2.cvtColor(img_working, cv2.COLOR_BGRA2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            return None, "Không tìm thấy khuôn mặt", 0

        face_rect = max(faces, key=lambda f: f[2] * f[3])
        auto_angle = get_face_angle(gray, face_rect)
        
        if abs(auto_angle) < 1.0: auto_angle = 0.0
        if abs(auto_angle) > 30.0: auto_angle = 0.0 # Chặn góc ảo gây lật ảnh

        total_angle = auto_angle + manual_angle
        
        if abs(total_angle) > 0.1:
            img_rotated = rotate_image(img_working, total_angle)
        else:
            img_rotated = img_working

        gray_new = cv2.cvtColor(img_rotated, cv2.COLOR_BGRA2GRAY)
        faces_new = face_cascade.detectMultiScale(gray_new, 1.1, 5)
        
        if len(faces_new) > 0:
            (x, y, w, h) = max(faces_new, key=lambda f: f[2] * f[3])
        else:
            (x, y, w, h) = face_rect

        if target_ratio < 0.7: 
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

        return canvas, f"Auto: {auto_angle:.1f}° | Tổng: {total_angle:.1f}°", total_angle

    except Exception as e:
        return None, str(e), 0

# --- 4. BỘ LỌC NÂNG CAO ---

def adjust_temperature(image, temp):
    if temp == 0: return image
    b, g, r, a = cv2.split(image)
    if temp > 0:
        r = cv2.add(r, temp)
        b = cv2.subtract(b, temp)
    else:
        r = cv2.add(r, temp) 
        b = cv2.subtract(b, temp)
    return cv2.merge([b, g, r, a])

def apply_clahe(image, clip_limit=2.0):
    b, g, r, a = cv2.split(image)
    lab = cv2.cvtColor(cv2.merge([b,g,r]), cv2.COLOR_BGR2LAB)
    l, aa, bb = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,aa,bb))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    b, g, r = cv2.split(final)
    return cv2.merge([b, g, r, a])

def makeup_vitality(image, intensity):
    if intensity == 0: return image
    b, g, r, a = cv2.split(image)
    hsv = cv2.cvtColor(cv2.merge([b,g,r]), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, int(intensity * 1.5))
    v = cv2.add(v, int(intensity * 0.5))
    final_hsv = cv2.merge([h, s, v])
    final_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    fb, fg, fr = cv2.split(final_bgr)
    return cv2.merge([fb, fg, fr, a])

def apply_advanced_effects(base_img, params):
    img_cv = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGBA2BGRA)
    
    if params['smooth'] > 0:
        d = 5
        sigma = int(params['smooth'] * 2) + 10
        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
        rgb = cv2.bilateralFilter(rgb, d=d, sigmaColor=sigma, sigmaSpace=sigma)
        b,g,r = cv2.split(rgb)
        a = cv2.split(img_cv)[3]
        img_cv = cv2.merge([b,g,r,a])

    if params['dehaze'] > 0:
        img_cv = apply_clahe(img_cv, clip_limit=1.0 + (params['dehaze']/10.0))
        
    if params['temp'] != 0:
        img_cv = adjust_temperature(img_cv, int(params['temp']))

    if params['makeup'] > 0:
        img_cv = makeup_vitality(img_cv, params['makeup'])

    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA))
    
    if params['sharp'] > 0:
        img_pil = ImageEnhance.Sharpness(img_pil).enhance(1.0 + params['sharp']/10.0)
    if params['exposure'] != 1.0:
        img_pil = ImageEnhance.Brightness(img_pil).enhance(params['exposure'])
    if params['contrast'] != 1.0:
        img_pil = ImageEnhance.Contrast(img_pil).enhance(params['contrast'])

    return img_pil

def create_print_layout(img_person, size_type):
    PAPER_W, PAPER_H = 1748, 1181 
    bg_paper = Image.new("RGB", (PAPER_W, PAPER_H), (255, 255, 255))
    
    if "4x6" in size_type:
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
    count = 0
    for r in range(rows):
        for c in range(cols):
            x = start_x + c * (target_w + gap)
            y = start_y + r * (target_h + gap)
            if x + target_w < PAPER_W and y + target_h < PAPER_H:
                bg_paper.paste(img_resized, (x, y))
                count += 1
    return bg_paper, count

# --- 5. GIAO DIỆN CHÍNH ---

col1, col2 = st.columns([1, 2.2])

with col1:
    st.header("🛠 Thiết lập")
    uploaded_file = st.file_uploader("1. Tải ảnh lên", type=['jpg', 'png', 'jpeg'])

    st.subheader("2. Quy cách & Xoay")
    size_option = st.radio("Kích thước:", ["4x6 cm (Hộ chiếu)", "3x4 cm (Giấy tờ)"])
    target_ratio = 3/4 if "3x4" in size_option else 4/6
    
    manual_rot = st.slider("Góc xoay (Chỉnh lệch vai):", -15.0, 15.0, 0.0, 0.5)
    
    bg_name = st.radio("Màu nền:", ["Trắng", "Xanh Chuẩn", "Xanh Nhạt"], horizontal=True)
    bg_map = {"Trắng": (255, 255, 255, 255), "Xanh Chuẩn": (66, 135, 245, 255), "Xanh Nhạt": (135, 206, 250, 255)}
    bg_val = bg_map.get(bg_name)

    if uploaded_file:
        if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
            with st.spinner('Đang tách nền...'):
                st.session_state.raw_nobg = process_raw_to_nobg(uploaded_file)
                st.session_state.current_file_name = uploaded_file.name
        
        if 'raw_nobg' in st.session_state:
            final_crop, debug_info, _ = crop_final_image(st.session_state.raw_nobg, manual_rot, target_ratio)
            
            if final_crop:
                st.session_state.base = final_crop
                st.caption(f"✅ {debug_info}")
            else:
                st.error(debug_info)

    st.markdown("---")
    # --- PHẦN GIAO DIỆN MỚI CÓ NÚT RESET ---
    c_head, c_btn = st.columns([3, 2])
    with c_head:
        st.subheader("3. Làm đẹp Pro")
    with c_btn:
        # Nút Reset nằm ngay góc phải tiêu đề
        st.button("🔄 Mặc định", on_click=reset_beauty_params, help="Quay về ảnh gốc chưa chỉnh sửa")
    
    # Gán key cho từng slider để hàm reset có thể điều khiển được
    with st.expander("✨ Da & Sức Sống", expanded=True):
        p_smooth = st.slider("Mịn da", 0, 30, 0, key="val_smooth")
        p_makeup = st.slider("Hồng hào (Môi/Má)", 0, 50, 0, key="val_makeup")

    with st.expander("💡 Ánh sáng", expanded=False):
        p_exposure = st.slider("Phơi sáng", 0.5, 1.5, 1.0, 0.05, key="val_exposure")
        p_contrast = st.slider("Tương phản", 0.5, 1.5, 1.0, 0.05, key="val_contrast")
        p_temp = st.slider("Nhiệt độ màu", -50, 50, 0, key="val_temp")

    with st.expander("👁️ Chi tiết ảnh", expanded=False):
        p_sharp = st.slider("Độ nét", 0, 20, 0, key="val_sharp")
        p_dehaze = st.slider("Giảm mù/Trong ảnh", 0, 20, 0, key="val_dehaze")

    params = {
        'smooth': p_smooth, 'makeup': p_makeup,
        'exposure': p_exposure, 'contrast': p_contrast, 'temp': p_temp,
        'sharp': p_sharp, 'dehaze': p_dehaze
    }

with col2:
    st.header(f"🖼 Kết quả ({size_option})")
    
    if 'base' in st.session_state and st.session_state.base:
        with st.spinner("Đang xử lý hiệu ứng..."):
            final_person = apply_advanced_effects(st.session_state.base, params)
        
        w, h = final_person.size
        final_img = Image.new("RGBA", (w, h), bg_val)
        final_img.paste(final_person, (0, 0), final_person)
        final_rgb = final_img.convert("RGB")
        
        st.image(final_rgb, width=350, caption="Ảnh hoàn thiện")
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        buf = io.BytesIO()
        final_rgb.save(buf, format="JPEG", quality=100, dpi=(300, 300))
        
        # --- SỬA LỖI TẢI FILE TRÊN ĐIỆN THOẠI ---
        # Đổi tên file chứa dấu Tiếng Việt sang Tiếng Anh
        name_mapping = {
            "Trắng": "white",
            "Xanh Chuẩn": "blue_standard",
            "Xanh Nhạt": "blue_light"
        }
        # Nếu không tìm thấy tên thì để mặc định là custom
        safe_bg_name = name_mapping.get(bg_name, "custom")
        
        c1.download_button(
            label="⬇️ Tải ảnh đơn", 
            data=buf.getvalue(), 
            file_name=f"anh_the_{safe_bg_name}.jpg", 
            mime="image/jpeg"
        )
        # ------------------------------------------

        if c2.button("🖨️ Xem file in 10x15cm"):
            paper, qty = create_print_layout(final_rgb, size_option)
            st.image(paper, caption=f"In {qty} ảnh", use_container_width=True)
            buf_p = io.BytesIO()
            paper.save(buf_p, format="JPEG", quality=100, dpi=(300, 300))
            st.download_button("⬇️ Tải File In", buf_p.getvalue(), "file_in_10x15.jpg", "image/jpeg", key='dl_print')
            
    else:
        st.info("👈 Tải ảnh lên để bắt đầu.")
