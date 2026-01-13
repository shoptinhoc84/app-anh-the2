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

st.title("📸 Studio Ảnh Thẻ - Pro Max")
st.markdown("---")

# --- 2. CÁC HÀM XỬ LÝ ẢNH CỐT LÕI ---

def rotate_image(image, angle):
    """Xoay ảnh giữ nguyên kích thước và kênh Alpha"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def get_face_angle(gray_img, face_rect):
    """Tính góc nghiêng dựa trên 2 mắt (AI)"""
    (x, y, w, h) = face_rect
    roi_gray = gray_img[y:y+h, x:x+w]
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
    
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        e1, e2 = eyes[0], eyes[-1]
        p1 = (e1[0] + e1[2]//2, e1[1] + e1[3]//2)
        p2 = (e2[0] + e2[2]//2, e2[1] + e2[3]//2)
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
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
        gray = cv2.cvtColor(no_bg_img, cv2.COLOR_BGRA2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            return None, "Không tìm thấy mặt", 0

        face_rect = max(faces, key=lambda f: f[2] * f[3])
        auto_angle = get_face_angle(gray, face_rect)
        final_angle = auto_angle + manual_angle
        
        if abs(final_angle) > 0.1:
            img_rotated = rotate_image(no_bg_img, final_angle)
        else:
            img_rotated = no_bg_img.copy()

        gray_new = cv2.cvtColor(img_rotated, cv2.COLOR_BGRA2GRAY)
        faces_new = face_cascade.detectMultiScale(gray_new, 1.1, 5)
        
        if len(faces_new) == 0:
            (x, y, w, h) = face_rect 
        else:
            (x, y, w, h) = max(faces_new, key=lambda f: f[2] * f[3])

        # CẤU HÌNH CẮT (Zoom 2.0 / Offset 0.45)
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

        return canvas, f"Góc xoay: {final_angle:.1f}°", final_angle

    except Exception as e:
        return None, str(e), 0

# --- 3. BỘ LỌC NÂNG CAO & TRANG ĐIỂM ---

def adjust_temperature(image, temp):
    """Thay đổi nhiệt độ màu (Vàng ấm / Xanh lạnh)"""
    if temp == 0: return image
    b, g, r, a = cv2.split(image)
    if temp > 0: # Ấm hơn (Tăng R, Giảm B)
        r = cv2.add(r, temp)
        b = cv2.subtract(b, temp)
    else: # Lạnh hơn (Giảm R, Tăng B)
        r = cv2.add(r, temp) # temp là âm
        b = cv2.subtract(b, temp)
    return cv2.merge([b, g, r, a])

def apply_clahe(image, clip_limit=2.0):
    """Giảm mù / Tăng chi tiết (Dehaze)"""
    b, g, r, a = cv2.split(image)
    # Chuyển sang LAB để xử lý kênh L (Lightness)
    lab = cv2.cvtColor(cv2.merge([b,g,r]), cv2.COLOR_BGR2LAB)
    l, aa, bb = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,aa,bb))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    b, g, r = cv2.split(final)
    return cv2.merge([b, g, r, a])

def makeup_vitality(image, intensity):
    """
    Giả lập trang điểm: Tăng sức sống (Làm môi đỏ, má hồng tự nhiên)
    bằng cách tăng bão hòa kênh đỏ/hồng.
    """
    if intensity == 0: return image
    
    # Chuyển sang HSV
    b, g, r, a = cv2.split(image)
    hsv = cv2.cvtColor(cv2.merge([b,g,r]), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Tăng Saturation (độ rực màu)
    s = cv2.add(s, int(intensity * 1.5))
    
    # Tăng nhẹ Value (độ sáng màu) để da hồng hào
    v = cv2.add(v, int(intensity * 0.5))
    
    final_hsv = cv2.merge([h, s, v])
    final_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    fb, fg, fr = cv2.split(final_bgr)
    
    return cv2.merge([fb, fg, fr, a])

def apply_advanced_effects(base_img, params):
    # 1. Chuyển sang OpenCV
    img_cv = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGBA2BGRA)
    
    # --- GIAI ĐOẠN 1: OpenCV (Pixel) ---
    
    # 1.1 Mịn da (Bilateral)
    if params['smooth'] > 0:
        d = 5
        sigma = int(params['smooth'] * 2) + 10
        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
        rgb = cv2.bilateralFilter(rgb, d=d, sigmaColor=sigma, sigmaSpace=sigma)
        b,g,r = cv2.split(rgb)
        a = cv2.split(img_cv)[3]
        img_cv = cv2.merge([b,g,r,a])

    # 1.2 Giảm mù / Tăng chi tiết (CLAHE)
    if params['dehaze'] > 0:
        img_cv = apply_clahe(img_cv, clip_limit=1.0 + (params['dehaze']/10.0))
        
    # 1.3 Nhiệt độ màu
    if params['temp'] != 0:
        img_cv = adjust_temperature(img_cv, int(params['temp']))

    # 1.4 Trang điểm / Sức sống
    if params['makeup'] > 0:
        img_cv = makeup_vitality(img_cv, params['makeup'])

    # --- GIAI ĐOẠN 2: PIL (Global) ---
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA))
    
    # 2.1 Độ nét (Sharpness)
    if params['sharp'] > 0:
        # 1.0 là gốc, 2.0 là nét gấp đôi
        factor = 1.0 + (params['sharp'] / 10.0) 
        img_pil = ImageEnhance.Sharpness(img_pil).enhance(factor)
        
    # 2.2 Phơi sáng (Exposure/Brightness)
    if params['exposure'] != 1.0:
        img_pil = ImageEnhance.Brightness(img_pil).enhance(params['exposure'])
        
    # 2.3 Độ đậm nhạt (Contrast)
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

# --- 4. GIAO DIỆN CHÍNH ---

col1, col2 = st.columns([1, 2.2]) # Chia cột rộng hơn cho phần kết quả

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

    # --- XỬ LÝ CẮT & XOAY ---
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
    # --- PHẦN CHỈNH SỬA NÂNG CAO ---
    st.subheader("3. Làm đẹp & Chỉnh ảnh")
    
    with st.expander("✨ Da & Trang Điểm (Cơ bản)", expanded=True):
        p_smooth = st.slider("Mịn da", 0, 30, 10)
        p_makeup = st.slider("Sức sống (Môi/Má hồng)", 0, 50, 0, help="Tăng độ hồng hào tươi tắn")

    with st.expander("💡 Ánh sáng & Màu sắc", expanded=False):
        p_exposure = st.slider("Độ phơi sáng (Sáng/Tối)", 0.5, 1.5, 1.0, 0.05)
        p_contrast = st.slider("Độ tương phản (Đậm/Nhạt)", 0.5, 1.5, 1.0, 0.05)
        p_temp = st.slider("Nhiệt độ màu (Vàng/Xanh)", -50, 50, 0, help="Kéo phải cho ảnh ấm hơn, trái cho lạnh hơn")

    with st.expander("👁️ Độ nét & Chi tiết", expanded=False):
        p_sharp = st.slider("Tăng độ nét (Sharpness)", 0, 20, 0)
        p_dehaze = st.slider("Giảm mù / Trong ảnh", 0, 20, 0, help="Giúp ảnh trong trẻo hơn, bớt sương mù")

    # Gom tham số
    params = {
        'smooth': p_smooth, 'makeup': p_makeup,
        'exposure': p_exposure, 'contrast': p_contrast, 'temp': p_temp,
        'sharp': p_sharp, 'dehaze': p_dehaze
    }

with col2:
    st.header(f"🖼 Kết quả ({size_option})")
    
    if 'base' in st.session_state and st.session_state.base:
        # Áp dụng tất cả hiệu ứng
        with st.spinner("Đang làm đẹp..."):
            final_person = apply_advanced_effects(st.session_state.base, params)
        
        # Ghép nền
        w, h = final_person.size
        final_img = Image.new("RGBA", (w, h), bg_val)
        final_img.paste(final_person, (0, 0), final_person)
        final_rgb = final_img.convert("RGB")
        
        st.image(final_rgb, width=350, caption="Ảnh hoàn thiện")
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        buf = io.BytesIO()
        final_rgb.save(buf, format="JPEG", quality=100, dpi=(300, 300))
        c1.download_button("⬇️ Tải ảnh đơn", buf.getvalue(), f"anh_the_{bg_name}.jpg", "image/jpeg")

        if c2.button("🖨️ Xem file in 10x15cm"):
            paper, qty = create_print_layout(final_rgb, size_option)
            st.image(paper, caption=f"In {qty} ảnh", use_container_width=True)
            buf_p = io.BytesIO()
            paper.save(buf_p, format="JPEG", quality=100, dpi=(300, 300))
            st.download_button("⬇️ Tải File In", buf_p.getvalue(), "file_in_10x15.jpg", "image/jpeg", key='dl_print')
            
    else:
        st.info("👈 Tải ảnh lên để trải nghiệm bộ công cụ Pro Max.")
