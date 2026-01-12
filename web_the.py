import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from rembg import remove
import io

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Studio Ảnh Thẻ Online", layout="wide")

st.title("📸 Studio Ảnh Thẻ - Web Version")
st.markdown("---")

# --- CÁC HÀM XỬ LÝ ẢNH (GIỮ NGUYÊN LOGIC CŨ) ---

def process_input_image(uploaded_file):
    """Xử lý tách nền và crop mặt tự động"""
    try:
        # Đọc ảnh từ file upload
        image = Image.open(uploaded_file)
        
        # 1. Tách nền
        with st.spinner('Đang tách nền và tìm khuôn mặt...'):
            no_bg = remove(image)

        # 2. Tìm mặt (OpenCV)
        cv_img = cv2.cvtColor(np.array(no_bg.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Load model nhận diện khuôn mặt
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            st.error("Không tìm thấy khuôn mặt nào trong ảnh!")
            return None, None

        # Lấy mặt lớn nhất
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

        # 3. Crop chuẩn ảnh thẻ (75% mặt)
        crop_h = int(h * 2.4)
        crop_w = int(crop_h * (4/6))
        face_center_x = x + w // 2
        top_y = int(y - (h * 0.50))
        left_x = int(face_center_x - crop_w // 2)

        # Tạo canvas trong suốt
        canvas_layer = Image.new("RGBA", (crop_w, crop_h), (0,0,0,0))
        
        # Paste ảnh đã tách nền vào vị trí đã tính toán
        # Lưu ý: Coordinates trong paste của PIL cần tính toán kỹ khi crop âm
        canvas_layer.paste(no_bg, (-left_x, -top_y), no_bg)

        face_info = {
            "chin_y": (y + h) - top_y, 
            "center_x": crop_w // 2, 
            "face_w": w
        }
        
        return canvas_layer, face_info

    except Exception as e:
        st.error(f"Lỗi xử lý: {str(e)}")
        return None, None

def apply_natural_enhance(img_cv):
    """Làm đẹp tự động (Gamma + Unsharp)"""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 120: gamma = 0.8 
    elif mean_brightness > 200: gamma = 1.1
    else: gamma = 0.95

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    b, g, r, a = cv2.split(img_cv)
    img_bgr = cv2.merge([b, g, r])
    img_bgr = cv2.LUT(img_bgr, table)

    gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 2.0)
    img_bgr = cv2.addWeighted(img_bgr, 1.2, gaussian, -0.2, 0)

    final_cv = cv2.merge([*cv2.split(img_bgr), a])
    return final_cv

def apply_effects(base_img, auto_beautify, smooth, sharp, brightness, color_sat):
    """Áp dụng các hiệu ứng từ thanh trượt"""
    img_cv = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGBA2BGRA)
    
    # 1. Auto Beauty
    if auto_beautify:
        img_cv = apply_natural_enhance(img_cv)

    # 2. Smooth Skin
    if smooth > 0:
        d = 5
        sigma = int(smooth * 2) + 10
        b, g, r, a = cv2.split(img_cv)
        rgb = cv2.merge([b,g,r])
        rgb = cv2.bilateralFilter(rgb, d=d, sigmaColor=sigma, sigmaSpace=sigma)
        img_cv = cv2.merge([rgb, a])

    # 3. Sharpness
    if sharp > 0:
        b, g, r, a = cv2.split(img_cv)
        rgb = cv2.merge([b,g,r])
        gaussian = cv2.GaussianBlur(rgb, (0, 0), 2.0)
        weight = 1.0 + (sharp / 4.0)
        rgb = cv2.addWeighted(rgb, weight, gaussian, - (weight - 1.0), 0)
        img_cv = cv2.merge([rgb, a])

    img_result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA))

    # 4. Color & Brightness
    if color_sat != 1.0:
        enhancer = ImageEnhance.Color(img_result)
        img_result = enhancer.enhance(color_sat)
        
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img_result)
        img_result = enhancer.enhance(brightness)
        
    return img_result

# --- GIAO DIỆN CHÍNH (STREAMLIT) ---

# Chia 2 cột: Cột trái (Công cụ), Cột phải (Hiển thị)
col1, col2 = st.columns([1, 2])

with col1:
    st.header("🛠 Công cụ")
    
    # 1. Upload Ảnh gốc
    uploaded_file = st.file_uploader("1. Chọn ảnh chân dung", type=['jpg', 'png', 'jpeg'])
    
    # Logic quản lý Session State để không phải xử lý lại khi kéo thanh trượt
    if uploaded_file:
        # Nếu là file mới thì xử lý lại từ đầu
        if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            base_img, face_info = process_input_image(uploaded_file)
            if base_img:
                st.session_state.base_img = base_img
                st.session_state.face_info = face_info
                st.session_state.last_uploaded = uploaded_file.name
    
    # 2. Các thanh trượt chỉnh màu
    st.subheader("2. Màu sắc & Làm đẹp")
    auto_check = st.checkbox("✨ Auto Trong Trẻo (Soft Studio)", value=True)
    
    color_val = st.slider("Đậm / Nhạt màu", 0.0, 2.0, 1.0, 0.1)
    smooth_val = st.slider("Mịn da (Soft Skin)", 0, 50, 0)
    sharp_val = st.slider("Độ nét (Detail)", 0, 10, 0)
    bright_val = st.slider("Độ sáng", 0.8, 1.5, 1.0, 0.05)
    
    st.markdown("---")
    
    # 3. Ghép áo
    st.subheader("3. Ghép Áo Vest")
    suit_file = st.file_uploader("Chọn file áo (PNG)", type=['png'])
    
    suit_size = 1.0
    suit_y = 0
    
    if suit_file:
        suit_size = st.slider("Kích thước áo", 0.8, 2.5, 1.0, 0.05)
        suit_y = st.slider("Vị trí áo (Lên/Xuống)", -100, 200, 0, 5)

with col2:
    st.header("🖼 Kết quả (Preview)")
    
    if 'base_img' in st.session_state and st.session_state.base_img:
        # Lấy ảnh cơ bản từ session
        current_base = st.session_state.base_img
        info = st.session_state.face_info
        
        # Áp dụng hiệu ứng
        processed_person = apply_effects(current_base, auto_check, smooth_val, sharp_val, bright_val, color_val)
        
        # Tạo nền trắng
        w, h = processed_person.size
        final_img = Image.new("RGBA", (w, h), "WHITE")
        final_img.paste(processed_person, (0, 0), processed_person)
        
        # Ghép áo (nếu có)
        if suit_file:
            suit_img = Image.open(suit_file)
            
            target_w = int(info["face_w"] * 2.8 * suit_size)
            if target_w < 10: target_w = 10
            ratio = target_w / suit_img.width
            target_h = int(suit_img.height * ratio)
            
            suit_resized = suit_img.resize((target_w, target_h), Image.LANCZOS)
            pos_x = info["center_x"] - target_w // 2
            pos_y = int(info["chin_y"] + suit_y)
            
            final_img.paste(suit_resized, (pos_x, pos_y), suit_resized)
        
        # Hiển thị ảnh
        final_rgb = final_img.convert("RGB")
        st.image(final_rgb, width=400)
        
        # Nút tải về
        # Chuyển ảnh sang bytes để download
        buf = io.BytesIO()
        final_rgb.save(buf, format="JPEG", quality=100, dpi=(600, 600))
        byte_im = buf.getvalue()
        
        st.download_button(
            label="💾 TẢI ẢNH VỀ (JPEG 600 DPI)",
            data=byte_im,
            file_name="anh_the_web.jpg",
            mime="image/jpeg"
        )
            
    else:
        st.info("👈 Vui lòng tải ảnh lên ở cột bên trái để bắt đầu.")
