import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import io

# --- 1. CẤU HÌNH ---
st.set_page_config(page_title="Studio Ảnh Thẻ Online", layout="wide")
st.title("📸 Studio Ảnh Thẻ - Web Version")
st.markdown("---")

# --- 2. CÁC HÀM XỬ LÝ (GIỮ NGUYÊN TỐI ƯU TỐC ĐỘ) ---

@st.cache_resource
def get_rembg_session():
    # Import lười để app chạy nhanh
    from rembg import new_session
    return new_session("u2netp")

def process_input_image(uploaded_file, target_ratio=4/6):
    try:
        image = Image.open(uploaded_file)
        
        # 1. Tách nền
        with st.spinner('Đang xử lý ảnh...'):
            from rembg import remove 
            session = get_rembg_session()
            no_bg = remove(image, session=session)

        # 2. Tìm mặt
        cv_img = cv2.cvtColor(np.array(no_bg.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            st.error("Không tìm thấy khuôn mặt nào!")
            return None, None

        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

        # 3. Crop
        crop_h = int(h * 2.5) 
        crop_w = int(crop_h * target_ratio)
        
        face_center_x = x + w // 2
        top_y = int(y - (h * 0.6))
        left_x = int(face_center_x - crop_w // 2)

        canvas_layer = Image.new("RGBA", (crop_w, crop_h), (0,0,0,0))
        canvas_layer.paste(no_bg, (-left_x, -top_y), no_bg)

        face_info = {
            "chin_y": (y + h) - top_y, 
            "center_x": crop_w // 2, 
            "face_w": w
        }
        return canvas_layer, face_info

    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
        return None, None

def apply_effects(base_img, auto_beautify, smooth, sharp, brightness, color_sat):
    img_cv = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGBA2BGRA)
    
    if auto_beautify:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < 120:
            img_cv = cv2.convertScaleAbs(img_cv, alpha=1.2, beta=10)

    if smooth > 0:
        d = 5
        sigma = int(smooth * 2) + 10
        b, g, r, a = cv2.split(img_cv)
        rgb = cv2.merge([b,g,r])
        rgb = cv2.bilateralFilter(rgb, d=d, sigmaColor=sigma, sigmaSpace=sigma)
        img_cv = cv2.merge([rgb, a])

    if sharp > 0:
        b, g, r, a = cv2.split(img_cv)
        rgb = cv2.merge([b,g,r])
        gaussian = cv2.GaussianBlur(rgb, (0, 0), 2.0)
        weight = 1.0 + (sharp / 5.0)
        rgb = cv2.addWeighted(rgb, weight, gaussian, - (weight - 1.0), 0)
        img_cv = cv2.merge([rgb, a])

    img_result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA))

    if color_sat != 1.0:
        img_result = ImageEnhance.Color(img_result).enhance(color_sat)
    if brightness != 1.0:
        img_result = ImageEnhance.Brightness(img_result).enhance(brightness)
        
    return img_result

# --- 3. GIAO DIỆN ---

col1, col2 = st.columns([1, 2])

with col1:
    st.header("🛠 Thiết lập")
    uploaded_file = st.file_uploader("1. Tải ảnh lên", type=['jpg', 'png', 'jpeg'])
    
    st.subheader("2. Quy cách & Nền")
    size_option = st.radio("Kích thước:", ["4x6 cm", "3x4 cm"])
    target_ratio = 3/4 if "3x4" in size_option else 4/6
    
    # --- CẬP NHẬT MÀU NỀN MỚI ---
    bg_color_name = st.radio("Màu nền:", ["Trắng (Siêu sáng)", "Xanh dương (Sáng)", "Xanh dương (Đậm)"], horizontal=True)
    
    if "Trắng" in bg_color_name:
        bg_color_val = (255, 255, 255, 255) # Trắng tuyệt đối
    elif "Sáng" in bg_color_name:
        bg_color_val = (135, 206, 250, 255) # Xanh da trời nhạt (Light Sky Blue) -> Rất sáng
    else:
        bg_color_val = (66, 135, 245, 255) # Xanh đậm cũ

    if uploaded_file:
        current_state_key = f"{uploaded_file.name}_{size_option}"
        if 'last_state_key' not in st.session_state or st.session_state.last_state_key != current_state_key:
            base_img, face_info = process_input_image(uploaded_file, target_ratio)
            if base_img:
                st.session_state.base_img = base_img
                st.session_state.face_info = face_info
                st.session_state.last_state_key = current_state_key

    st.markdown("---")
    st.subheader("3. Làm đẹp")
    auto_check = st.checkbox("Auto Sáng Da", value=True)
    smooth_val = st.slider("Mịn da", 0, 30, 0)
    bright_val = st.slider("Độ sáng", 0.8, 1.3, 1.0, 0.05)
    
    st.markdown("---")
    st.subheader("4. Ghép Áo")
    suit_file = st.file_uploader("Tải áo (PNG)", type=['png'])
    if suit_file:
        suit_size = st.slider("Kích thước áo", 0.8, 2.0, 1.0, 0.05)
        suit_y = st.slider("Vị trí áo", -50, 150, 0, 5)

with col2:
    st.header(f"🖼 Kết quả")
    
    if 'base_img' in st.session_state and st.session_state.base_img:
        current_base = st.session_state.base_img
        info = st.session_state.face_info
        
        processed_person = apply_effects(current_base, auto_check, smooth_val, 0, bright_val, 1.0)
        
        w, h = processed_person.size
        final_img = Image.new("RGBA", (w, h), bg_color_val)
        final_img.paste(processed_person, (0, 0), processed_person)
        
        if suit_file:
            try:
                suit_img = Image.open(suit_file)
                target_w_suit = int(info["face_w"] * 2.8 * suit_size)
                ratio_s = target_w_suit / suit_img.width
                target_h_suit = int(suit_img.height * ratio_s)
                
                suit_resized = suit_img.resize((target_w_suit, target_h_suit), Image.LANCZOS)
                pos_x = info["center_x"] - target_w_suit // 2
                pos_y = int(info["chin_y"] + suit_y)
                
                final_img.paste(suit_resized, (pos_x, pos_y), suit_resized)
            except: pass
        
        final_rgb = final_img.convert("RGB")
        st.image(final_rgb, width=350)
        
        buf = io.BytesIO()
        final_rgb.save(buf, format="JPEG", quality=100, dpi=(300, 300))
        st.download_button("💾 TẢI ẢNH VỀ", buf.getvalue(), "anh_the.jpg", "image/jpeg")
    else:
        st.info("👈 Tải ảnh lên để bắt đầu.")
