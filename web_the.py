import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from rembg import remove, new_session
import io

# --- 1. CẤU HÌNH & CACHE ---
st.set_page_config(page_title="Studio Ảnh Thẻ Online", layout="wide")

# [QUAN TRỌNG] Dùng model 'u2netp' (bản nhẹ) để tránh bị lỗi hết RAM trên web miễn phí
@st.cache_resource
def get_rembg_session():
    return new_session("u2netp")

st.title("📸 Studio Ảnh Thẻ - Web Version")
st.markdown("---")

# --- 2. CÁC HÀM XỬ LÝ ẢNH ---

def process_input_image(uploaded_file, target_ratio=4/6):
    """
    Xử lý tách nền và crop mặt theo tỷ lệ (3:4 hoặc 4:6)
    target_ratio: 0.75 (3x4) hoặc 0.666 (4x6)
    """
    try:
        image = Image.open(uploaded_file)
        
        # 1. Tách nền
        with st.spinner('Đang tách nền và tìm khuôn mặt...'):
            session = get_rembg_session()
            no_bg = remove(image, session=session)

        # 2. Tìm mặt (OpenCV)
        cv_img = cv2.cvtColor(np.array(no_bg.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            st.error("Không tìm thấy khuôn mặt nào trong ảnh!")
            return None, None

        # Lấy mặt lớn nhất
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

        # 3. Tính toán Crop theo tỷ lệ
        # Quy chuẩn: Mặt chiếm khoảng 70-75% chiều cao ảnh
        crop_h = int(h * 2.5) 
        crop_w = int(crop_h * target_ratio) # Tính chiều rộng dựa trên tỷ lệ
        
        face_center_x = x + w // 2
        top_y = int(y - (h * 0.6)) # Lấy dư phần đầu một chút
        left_x = int(face_center_x - crop_w // 2)

        # Tạo canvas
        canvas_layer = Image.new("RGBA", (crop_w, crop_h), (0,0,0,0))
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

def apply_effects(base_img, auto_beautify, smooth, sharp, brightness, color_sat):
    """Áp dụng bộ lọc làm đẹp"""
    img_cv = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGBA2BGRA)
    
    # Auto Beauty đơn giản
    if auto_beautify:
        # Tăng sáng nhẹ nếu ảnh tối
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < 120:
            img_cv = cv2.convertScaleAbs(img_cv, alpha=1.2, beta=10)

    # Làm mịn da (Bilateral Filter)
    if smooth > 0:
        d = 5
        sigma = int(smooth * 2) + 10
        b, g, r, a = cv2.split(img_cv)
        rgb = cv2.merge([b,g,r])
        rgb = cv2.bilateralFilter(rgb, d=d, sigmaColor=sigma, sigmaSpace=sigma)
        img_cv = cv2.merge([rgb, a])

    # Tăng độ nét
    if sharp > 0:
        b, g, r, a = cv2.split(img_cv)
        rgb = cv2.merge([b,g,r])
        gaussian = cv2.GaussianBlur(rgb, (0, 0), 2.0)
        weight = 1.0 + (sharp / 5.0)
        rgb = cv2.addWeighted(rgb, weight, gaussian, - (weight - 1.0), 0)
        img_cv = cv2.merge([rgb, a])

    img_result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA))

    # Chỉnh màu & Sáng
    if color_sat != 1.0:
        img_result = ImageEnhance.Color(img_result).enhance(color_sat)
    if brightness != 1.0:
        img_result = ImageEnhance.Brightness(img_result).enhance(brightness)
        
    return img_result

# --- 3. GIAO DIỆN CHÍNH (STREAMLIT) ---

col1, col2 = st.columns([1, 2])

with col1:
    st.header("🛠 Thiết lập")
    
    # --- PHẦN 1: QUY ĐỊNH ẢNH ---
    with st.expander("📋 QUY ĐỊNH CHỤP ẢNH (Đọc kỹ)", expanded=True):
        st.markdown("""
        * **Khuôn mặt:** Nhìn thẳng, biểu cảm tự nhiên.
        * **Mắt & Môi:** Mắt mở to, mím môi, không cười hở răng.
        * **Tư thế:** Không nghiêng đầu, không cúi đầu.
        * **Tóc:** Gọn gàng, **không che trán**, không che lông mày.
        * **Trạng thái:** Không cau mày, không bực tức.
        """)
    
    # --- PHẦN 2: UPLOAD ---
    uploaded_file = st.file_uploader("1. Tải ảnh chân dung lên", type=['jpg', 'png', 'jpeg'])

    # --- PHẦN 3: TÙY CHỌN KÍCH THƯỚC & MÀU NỀN ---
    st.subheader("2. Chọn quy cách")
    
    # Chọn kích thước
    size_option = st.radio("Kích thước ảnh:", ["4x6 cm (Hộ chiếu/Quốc tế)", "3x4 cm (GPLX/Khám sức khỏe)"])
    target_ratio = 3/4 if "3x4" in size_option else 4/6
    
    # Chọn màu nền
    bg_color_name = st.radio("Màu nền:", ["Trắng", "Xanh dương"], horizontal=True)
    
    # Mã màu (RGBA)
    if bg_color_name == "Xanh dương":
        bg_color_val = (66, 135, 245, 255) # Xanh dương chuẩn thẻ
    else:
        bg_color_val = (255, 255, 255, 255) # Trắng

    # --- LOGIC XỬ LÝ LẠI KHI ĐỔI KÍCH THƯỚC ---
    if uploaded_file:
        current_state_key = f"{uploaded_file.name}_{size_option}"
        
        if 'last_state_key' not in st.session_state or st.session_state.last_state_key != current_state_key:
            base_img, face_info = process_input_image(uploaded_file, target_ratio)
            if base_img:
                st.session_state.base_img = base_img
                st.session_state.face_info = face_info
                st.session_state.last_state_key = current_state_key

    # --- PHẦN 4: CHỈNH ĐẸP ---
    st.markdown("---")
    st.subheader("3. Làm đẹp")
    auto_check = st.checkbox("✨ Auto Sáng Da", value=True)
    smooth_val = st.slider("Mịn da", 0, 30, 0)
    bright_val = st.slider("Độ sáng", 0.8, 1.3, 1.0, 0.05)

with col2:
    st.header(f"🖼 Kết quả ({size_option})")
    
    if 'base_img' in st.session_state and st.session_state.base_img:
        current_base = st.session_state.base_img
        info = st.session_state.face_info
        
        # 1. Áp dụng hiệu ứng làm đẹp
        processed_person = apply_effects(current_base, auto_check, smooth_val, 0, bright_val, 1.0)
        
        # 2. Tạo nền theo màu đã chọn
        w, h = processed_person.size
        final_img = Image.new("RGBA", (w, h), bg_color_val)
        
        # 3. Ghép người lên nền
        final_img.paste(processed_person, (0, 0), processed_person)
        
        # 4. Hiển thị
        final_rgb = final_img.convert("RGB")
        st.image(final_rgb, width=350)
        
        # 5. Nút tải về
        buf = io.BytesIO()
        final_rgb.save(buf, format="JPEG", quality=100, dpi=(300, 300))
        byte_im = buf.getvalue()
        
        file_name_dl = "anh_3x4_xanh.jpg" if "3x4" in size_option else "anh_4x6_trang.jpg"
        
        st.download_button(
            label="💾 TẢI ẢNH VỀ MÁY",
            data=byte_im,
            file_name=file_name_dl,
            mime="image/jpeg"
        )
            
    else:
        st.info("👈 Vui lòng tải ảnh lên và đọc kỹ quy định bên trái.")
