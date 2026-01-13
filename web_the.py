import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from rembg import remove, new_session
import io
import math

# --- 1. CẤU HÌNH & CACHE ---
st.set_page_config(page_title="Studio Ảnh Thẻ Pro", layout="wide")

@st.cache_resource
def get_rembg_session():
    return new_session("u2netp")

st.title("📸 Studio Ảnh Thẻ - Chuyên Nghiệp")
st.markdown("---")

# --- 2. CÁC HÀM XỬ LÝ ẢNH ---

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
    # Zoom vào vùng mặt để tìm mắt dễ hơn
    roi_gray = gray_img[y:y+h, x:x+w]
    
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    # Giảm minNeighbors xuống 3 để nhạy hơn (dễ tìm thấy mắt hơn)
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
    
    if len(eyes) >= 2:
        # Sắp xếp mắt trái/phải
        eyes = sorted(eyes, key=lambda e: e[0])
        e1 = eyes[0]
        e2 = eyes[-1] # Lấy mắt xa nhất
        
        # Tâm mắt
        p1 = (e1[0] + e1[2]//2, e1[1] + e1[3]//2)
        p2 = (e2[0] + e2[2]//2, e2[1] + e2[3]//2)
        
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        return angle
    return 0.0

def process_raw_to_nobg(uploaded_file):
    """Bước 1: Chỉ tách nền (Chạy nặng, cần Cache)"""
    image = Image.open(uploaded_file)
    session = get_rembg_session()
    no_bg_pil = remove(image, session=session)
    # Convert sang OpenCV Format (BGRA)
    no_bg_cv = cv2.cvtColor(np.array(no_bg_pil), cv2.COLOR_RGBA2BGRA)
    return no_bg_cv

def crop_final_image(no_bg_img, manual_angle, target_ratio):
    """Bước 2: Xoay và Cắt (Chạy nhẹ, gọi liên tục khi kéo slider)"""
    try:
        # 1. Tìm mặt để lấy thông tin ban đầu
        gray = cv2.cvtColor(no_bg_img, cv2.COLOR_BGRA2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            return None, "Không tìm thấy mặt", 0

        face_rect = max(faces, key=lambda f: f[2] * f[3])
        
        # 2. Tính góc Auto
        auto_angle = get_face_angle(gray, face_rect)
        
        # 3. Tổng góc xoay = Auto + Thủ công
        final_angle = auto_angle + manual_angle
        
        # Xoay ảnh
        if abs(final_angle) > 0.1:
            img_rotated = rotate_image(no_bg_img, final_angle)
        else:
            img_rotated = no_bg_img.copy()

        # 4. Tìm lại mặt trên ảnh đã xoay (Quan trọng)
        gray_new = cv2.cvtColor(img_rotated, cv2.COLOR_BGRA2GRAY)
        faces_new = face_cascade.detectMultiScale(gray_new, 1.1, 5)
        
        if len(faces_new) == 0:
            # Nếu xoay xong mất mặt, dùng tọa độ cũ (chấp nhận sai số nhẹ)
            (x, y, w, h) = face_rect 
        else:
            (x, y, w, h) = max(faces_new, key=lambda f: f[2] * f[3])

        # 5. CẮT ẢNH (Cấu hình Zoom 2.0 / Offset 0.45)
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

        return canvas, f"Auto: {auto_angle:.1f}° | Chỉnh thêm: {manual_angle:.1f}°", final_angle

    except Exception as e:
        return None, str(e), 0

def apply_effects(base_img, auto_beautify, smooth, sharp, brightness):
    img_cv = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGBA2BGRA)
    
    if auto_beautify:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < 120:
            img_cv = cv2.convertScaleAbs(img_cv, alpha=1.2, beta=10)

    if smooth > 0:
        d = 5
        sigma = int(smooth * 2) + 10
        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
        rgb = cv2.bilateralFilter(rgb, d=d, sigmaColor=sigma, sigmaSpace=sigma)
        b,g,r = cv2.split(rgb)
        a = cv2.split(img_cv)[3]
        img_cv = cv2.merge([b,g,r,a])

    if sharp > 0:
        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
        gaussian = cv2.GaussianBlur(rgb, (0, 0), 2.0)
        weight = 1.0 + (sharp / 5.0)
        rgb = cv2.addWeighted(rgb, weight, gaussian, - (weight - 1.0), 0)
        b,g,r = cv2.split(rgb)
        a = cv2.split(img_cv)[3]
        img_cv = cv2.merge([b,g,r,a])

    img_result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA))
    if brightness != 1.0:
        img_result = ImageEnhance.Brightness(img_result).enhance(brightness)
        
    return img_result

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

# --- 3. GIAO DIỆN CHÍNH ---

col1, col2 = st.columns([1, 2])

with col1:
    st.header("🛠 Thiết lập")
    uploaded_file = st.file_uploader("1. Tải ảnh lên", type=['jpg', 'png', 'jpeg'])

    st.subheader("2. Quy cách & Chỉnh nghiêng")
    size_option = st.radio("Kích thước:", ["4x6 cm (Hộ chiếu)", "3x4 cm (Giấy tờ)"])
    target_ratio = 3/4 if "3x4" in size_option else 4/6
    
    # --- THANH TRƯỢT XOAY THỦ CÔNG ---
    st.info("💡 Nếu ảnh bị lệch vai, kéo thanh này để chỉnh:")
    manual_rot = st.slider("Góc xoay (Độ):", -15.0, 15.0, 0.0, 0.5)
    
    bg_name = st.radio("Màu nền:", ["Trắng", "Xanh Chuẩn", "Xanh Nhạt"], horizontal=True)
    bg_map = {
        "Trắng": (255, 255, 255, 255),
        "Xanh Chuẩn": (66, 135, 245, 255),
        "Xanh Nhạt": (135, 206, 250, 255)
    }
    bg_val = bg_map.get(bg_name, (255,255,255,255))

    # --- LOGIC XỬ LÝ MỚI (TỐI ƯU TỐC ĐỘ) ---
    if uploaded_file:
        # Bước 1: Tách nền (Chỉ chạy 1 lần khi đổi ảnh)
        if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
            with st.spinner('Đang tách nền...'):
                st.session_state.raw_nobg = process_raw_to_nobg(uploaded_file)
                st.session_state.current_file_name = uploaded_file.name
        
        # Bước 2: Xoay & Cắt (Chạy mỗi khi chỉnh thanh trượt)
        if 'raw_nobg' in st.session_state:
            final_crop, debug_info, total_angle = crop_final_image(st.session_state.raw_nobg, manual_rot, target_ratio)
            
            if final_crop:
                st.session_state.base = final_crop
                st.success(f"✅ {debug_info}") # Hiển thị thông số góc xoay
            else:
                st.error(debug_info)

    st.markdown("---")
    st.subheader("3. Làm đẹp")
    auto_check = st.checkbox("Auto Sáng Da", value=True)
    smooth_val = st.slider("Mịn da", 0, 30, 0)
    bright_val = st.slider("Độ sáng", 0.8, 1.3, 1.0, 0.05)

with col2:
    st.header(f"🖼 Kết quả ({size_option})")
    
    if 'base' in st.session_state and st.session_state.base:
        # 1. Hiệu ứng
        final_person = apply_effects(st.session_state.base, auto_check, smooth_val, 0, bright_val)
        
        # 2. Ghép nền
        w, h = final_person.size
        final_img = Image.new("RGBA", (w, h), bg_val)
        final_img.paste(final_person, (0, 0), final_person)
        final_rgb = final_img.convert("RGB")
        
        st.image(final_rgb, width=300, caption="Ảnh hoàn thiện")
        
        # 3. Tải về
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        buf = io.BytesIO()
        final_rgb.save(buf, format="JPEG", quality=100, dpi=(300, 300))
        with c1:
            st.download_button("⬇️ Tải ảnh đơn", buf.getvalue(), f"anh_the_{bg_name}.jpg", "image/jpeg")

        with c2:
            if st.button("🖨️ Xem file in 10x15cm"):
                paper, qty = create_print_layout(final_rgb, size_option)
                st.image(paper, caption=f"In {qty} ảnh", use_container_width=True)
                
                buf_p = io.BytesIO()
                paper.save(buf_p, format="JPEG", quality=100, dpi=(300, 300))
                st.download_button("⬇️ Tải File In", buf_p.getvalue(), "file_in_10x15.jpg", "image/jpeg")
            
    else:
        st.info("👈 Tải ảnh lên. Nếu vai bị lệch, hãy kéo thanh trượt 'Góc xoay' bên trái.")
