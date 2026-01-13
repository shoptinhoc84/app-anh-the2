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

st.title("📸 Studio Ảnh Thẻ - AI Chuyên Nghiệp")
st.markdown("---")

# --- 2. CÁC HÀM XỬ LÝ ẢNH ---

def rotate_image(image, angle):
    """
    Xoay ảnh theo góc (độ) mà không làm mất góc ảnh (giữ nguyên alpha)
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def get_face_angle(gray_img, face_rect):
    """
    Tính góc nghiêng dựa trên 2 mắt
    """
    (x, y, w, h) = face_rect
    roi_gray = gray_img[y:y+h, x:x+w]
    
    # Tìm mắt
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    
    if len(eyes) >= 2:
        # Sắp xếp mắt trái/phải theo trục x
        eyes = sorted(eyes, key=lambda e: e[0])
        (ex1, ey1, ew1, eh1) = eyes[0]
        (ex2, ey2, ew2, eh2) = eyes[-1] # Lấy mắt xa nhất để tránh nhầm mũi
        
        # Tọa độ tâm mắt
        p1 = (ex1 + ew1//2, ey1 + eh1//2)
        p2 = (ex2 + ew2//2, ey2 + eh2//2)
        
        # Tính góc
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        return angle
    return 0

def process_input_image(uploaded_file, target_ratio=4/6):
    try:
        image = Image.open(uploaded_file)
        
        # 1. Tách nền trước
        with st.spinner('Đang tách nền & cân chỉnh...'):
            session = get_rembg_session()
            no_bg_pil = remove(image, session=session)
            
        # Chuyển sang OpenCV để xử lý
        no_bg = cv2.cvtColor(np.array(no_bg_pil), cv2.COLOR_RGBA2BGRA)
        
        # 2. Tìm mặt lần 1 (để lấy vùng tìm mắt)
        # Tách kênh alpha để tìm mặt trên nền ảnh gốc (chính xác hơn) hoặc convert sang gray
        # Ở đây dùng gray từ ảnh đã tách nền cũng ổn
        gray = cv2.cvtColor(no_bg, cv2.COLOR_BGRA2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            st.error("Không tìm thấy khuôn mặt!")
            return None, None

        # Lấy mặt lớn nhất
        face_rect = max(faces, key=lambda f: f[2] * f[3])
        
        # --- 3. TỰ ĐỘNG XOAY THẲNG MẶT (NEW) ---
        angle = get_face_angle(gray, face_rect)
        
        # Chỉ xoay nếu nghiêng đáng kể (> 1 độ) và không quá lố (< 45 độ)
        if abs(angle) > 1 and abs(angle) < 45:
            # st.info(f"Phát hiện đầu nghiêng {angle:.1f} độ. Đang tự động xoay thẳng...") 
            # Xoay ảnh no_bg
            no_bg = rotate_image(no_bg, angle)
            
            # QUAN TRỌNG: Phải tìm lại mặt sau khi xoay vì tọa độ đã đổi
            gray_new = cv2.cvtColor(no_bg, cv2.COLOR_BGRA2GRAY)
            faces_new = face_cascade.detectMultiScale(gray_new, 1.1, 5)
            if len(faces_new) > 0:
                face_rect = max(faces_new, key=lambda f: f[2] * f[3])
        
        (x, y, w, h) = face_rect

        # --- 4. CẮT ẢNH (GIỮ CẤU HÌNH BẠN THÍCH) ---
        if target_ratio < 0.7: 
            # 4x6 (Hộ chiếu): Zoom 2.0, Offset 0.45
            zoom_factor = 2.0  
            top_offset = 0.45   
        else:
            # 3x4 (Giấy tờ): Zoom 2.2, Offset 0.5
            zoom_factor = 2.2
            top_offset = 0.5

        crop_h = int(h * zoom_factor) 
        crop_w = int(crop_h * target_ratio)
        
        face_center_x = x + w // 2
        top_y = int(y - (h * top_offset)) 
        left_x = int(face_center_x - crop_w // 2)

        # Convert ngược lại PIL để crop an toàn (xử lý tràn viền)
        img_final_pil = Image.fromarray(cv2.cvtColor(no_bg, cv2.COLOR_BGRA2RGBA))
        
        # Tạo canvas trong suốt
        canvas = Image.new("RGBA", (crop_w, crop_h), (0,0,0,0))
        
        # Paste ảnh vào canvas (tự động xử lý phần âm)
        canvas.paste(img_final_pil, (-left_x, -top_y), img_final_pil)

        face_info = {"chin_y": (y + h) - top_y, "center_x": crop_w // 2}
        
        return canvas, face_info

    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
        return None, None

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
    # Khổ 10x15cm (4x6 inch) - 300 DPI
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
    uploaded_file = st.file_uploader("Tải ảnh lên", type=['jpg', 'png', 'jpeg'])

    st.subheader("Quy cách")
    size_option = st.radio("Kích thước:", ["4x6 cm (Hộ chiếu)", "3x4 cm (Giấy tờ)"])
    target_ratio = 3/4 if "3x4" in size_option else 4/6
    
    bg_name = st.radio("Màu nền:", ["Trắng", "Xanh Chuẩn", "Xanh Nhạt"], horizontal=True)
    bg_map = {
        "Trắng": (255, 255, 255, 255),
        "Xanh Chuẩn": (66, 135, 245, 255),
        "Xanh Nhạt": (135, 206, 250, 255)
    }
    bg_val = bg_map.get(bg_name, (255,255,255,255))

    if uploaded_file:
        state_key = f"{uploaded_file.name}_{size_option}"
        if 'last_key' not in st.session_state or st.session_state.last_key != state_key:
            base_img, info = process_input_image(uploaded_file, target_ratio)
            if base_img:
                st.session_state.base = base_img
                st.session_state.last_key = state_key

    st.markdown("---")
    st.subheader("Làm đẹp")
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
        
        st.image(final_rgb, width=300, caption="Ảnh thẻ hoàn thiện")
        
        # 3. Khu vực tải về
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        # Nút tải ảnh đơn
        buf = io.BytesIO()
        final_rgb.save(buf, format="JPEG", quality=100, dpi=(300, 300))
        with c1:
            st.download_button("⬇️ Tải ảnh đơn (File gốc)", buf.getvalue(), f"anh_the_{bg_name}.jpg", "image/jpeg")

        # Nút tải file in
        with c2:
            if st.button("🖨️ Xem file in 10x15cm"):
                paper, qty = create_print_layout(final_rgb, size_option)
                st.image(paper, caption=f"Demo in {qty} ảnh", use_container_width=True)
                
                buf_p = io.BytesIO()
                paper.save(buf_p, format="JPEG", quality=100, dpi=(300, 300))
                st.download_button("⬇️ Tải File In (Ra tiệm in luôn)", buf_p.getvalue(), "file_in_10x15.jpg", "image/jpeg")
            
    else:
        st.info("👈 Tải ảnh lên để bắt đầu. Hệ thống sẽ tự động cân bằng mặt nghiêng.")
