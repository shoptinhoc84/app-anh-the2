import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from rembg import remove, new_session
import io

# --- 1. CẤU HÌNH & CACHE ---
st.set_page_config(page_title="Studio Ảnh Thẻ V2.0", layout="wide")

@st.cache_resource
def get_rembg_session():
    return new_session("u2netp")

st.title("📸 Studio Ảnh Thẻ - Pro Max (AI Edition V2)")
st.markdown("---")

# --- 2. HÀM RESET ---
def reset_beauty_params():
    """Đưa toàn bộ thông số về mặc định"""
    st.session_state.val_smooth = 0
    st.session_state.val_makeup = 0
    st.session_state.val_exposure = 1.0
    st.session_state.val_contrast = 1.0
    st.session_state.val_temp = 0
    st.session_state.val_sharp_amount = 0 # Thay đổi thành Smart Sharpen
    st.session_state.val_denoise = 0      # Mới: Giảm nhiễu
    st.session_state.val_blacks = 0       # Mới: Màu đen
    st.session_state.val_whites = 0       # Mới: Màu trắng
    st.session_state.val_dehaze = 0
    st.session_state.ai_enabled = False

# --- 3. CÁC HÀM XỬ LÝ ẢNH CỐT LÕI (CORE) ---

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Dùng borderReplicate để tránh viền đen khi xoay
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
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
        
        # Nếu khoảng cách 2 mắt quá gần (lỗi nhận diện), bỏ qua
        if delta_x < w/5: return 0.0
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        return angle
    return 0.0

def process_raw_to_nobg(file_input):
    image = Image.open(file_input)
    session = get_rembg_session()
    no_bg_pil = remove(image, session=session, alpha_matting=True) # Thêm alpha_matting cho viền mượt hơn
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

        # Lấy khuôn mặt to nhất
        face_rect = max(faces, key=lambda f: f[2] * f[3])
        
        # 1. Tự động tính góc nghiêng đầu
        auto_angle = get_face_angle(gray, face_rect)
        
        # Giới hạn góc auto để tránh xoay bậy
        if abs(auto_angle) < 1.0: auto_angle = 0.0
        if abs(auto_angle) > 20.0: auto_angle = 0.0 

        total_angle = auto_angle + manual_angle
        
        if abs(total_angle) > 0.1:
            img_rotated = rotate_image(img_working, total_angle)
        else:
            img_rotated = img_working

        # Detect lại mặt sau khi xoay để crop chuẩn
        gray_new = cv2.cvtColor(img_rotated, cv2.COLOR_BGRA2GRAY)
        faces_new = face_cascade.detectMultiScale(gray_new, 1.1, 5)
        
        if len(faces_new) > 0:
            (x, y, w, h) = max(faces_new, key=lambda f: f[2] * f[3])
        else:
            (x, y, w, h) = face_rect # Fallback về tọa độ cũ

        # Tỷ lệ zoom khung hình (Cắt cúp)
        if target_ratio < 0.7: # 4x6 (hẹp ngang)
            zoom_factor = 2.0  
            top_offset = 0.45   
        else: # 3x4 (rộng hơn chút)
            zoom_factor = 2.2
            top_offset = 0.5

        crop_h = int(h * zoom_factor) 
        crop_w = int(crop_h * target_ratio)
        
        face_center_x = x + w // 2
        top_y = int(y - (h * top_offset)) 
        left_x = int(face_center_x - crop_w // 2)

        # Tạo canvas trong suốt để paste mặt vào
        img_pil = Image.fromarray(cv2.cvtColor(img_rotated, cv2.COLOR_BGRA2RGBA))
        canvas = Image.new("RGBA", (crop_w, crop_h), (0,0,0,0))
        canvas.paste(img_pil, (-left_x, -top_y), img_pil)

        return canvas, f"Góc xoay Auto: {auto_angle:.1f}° | Tổng: {total_angle:.1f}°", total_angle

    except Exception as e:
        return None, str(e), 0

# --- 4. BỘ LỌC NÂNG CAO (NEW FEATURES) ---

def adjust_levels(image, blacks=0, whites=0):
    """
    Điều chỉnh Levels (Blacks/Whites) giống Photoshop.
    blacks: 0-50 (kéo vùng tối tối hơn)
    whites: 0-50 (kéo vùng sáng sáng hơn)
    """
    if blacks == 0 and whites == 0: return image
    
    # Chuyển đổi phạm vi 0-255
    in_black = blacks
    in_white = 255 - whites
    
    in_black = max(0, min(in_black, 100))
    in_white = max(150, min(in_white, 255))
    
    # Tạo bảng tra (LUT) để xử lý nhanh
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        val = (i - in_black) * 255 / (in_white - in_black)
        lut[i] = np.clip(val, 0, 255)
        
    b, g, r, a = cv2.split(image)
    b = cv2.LUT(b, lut)
    g = cv2.LUT(g, lut)
    r = cv2.LUT(r, lut)
    return cv2.merge([b, g, r, a])

def apply_unsharp_mask(image, amount=0.0):
    """Làm sắc nét thông minh (Unsharp Mask) - Xóa mờ"""
    if amount == 0: return image
    # Amount slider 0-20 -> đổi sang scale thực tế 0.0 - 2.0
    strength = amount / 10.0
    
    # Gaussian Blur làm mờ để tạo mask
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    
    # Công thức: Original + (Original - Blurred) * strength
    sharp = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
    return sharp

def apply_denoise(image, strength=0):
    """Giảm nhiễu màu"""
    if strength == 0: return image
    # Strength 0-20. Chuyển sang parameter cho hàm
    # Lưu ý: Hàm này khá nặng, chạy trên ảnh crop nhỏ thì ok
    b, g, r, a = cv2.split(image)
    rgb = cv2.merge([b, g, r])
    
    # h: độ mạnh lọc nhiễu
    h_val = strength
    denoised_rgb = cv2.fastNlMeansDenoisingColored(rgb, None, h_val, h_val, 7, 21)
    
    b, g, r = cv2.split(denoised_rgb)
    return cv2.merge([b, g, r, a])

def apply_advanced_effects(base_img, params):
    # Convert PIL to CV2
    img_cv = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGBA2BGRA)
    
    # 1. Giảm nhiễu (Chạy đầu tiên để làm sạch ảnh)
    if params['denoise'] > 0:
        img_cv = apply_denoise(img_cv, params['denoise'])

    # 2. Mịn da (Smooth - Bilateral Filter)
    if params['smooth'] > 0:
        d = 5
        sigma = int(params['smooth'] * 2) + 10
        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
        rgb = cv2.bilateralFilter(rgb, d=d, sigmaColor=sigma, sigmaSpace=sigma)
        b,g,r = cv2.split(rgb)
        a = cv2.split(img_cv)[3]
        img_cv = cv2.merge([b,g,r,a])

    # 3. Giảm mù / Phủ mờ (Dehaze - CLAHE)
    if params['dehaze'] > 0:
        b, g, r, a = cv2.split(img_cv)
        lab = cv2.cvtColor(cv2.merge([b,g,r]), cv2.COLOR_BGR2LAB)
        l, aa, bb = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0 + (params['dehaze']/10.0), tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,aa,bb))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        b, g, r = cv2.split(final)
        img_cv = cv2.merge([b, g, r, a])
        
    # 4. Nhiệt độ màu
    if params['temp'] != 0:
        temp = int(params['temp'])
        b, g, r, a = cv2.split(img_cv)
        if temp > 0:
            r = cv2.add(r, temp)
            b = cv2.subtract(b, temp)
        else:
            r = cv2.add(r, temp)
            b = cv2.subtract(b, temp)
        img_cv = cv2.merge([b, g, r, a])

    # 5. Hồng hào / Sức sống
    if params['makeup'] > 0:
        b, g, r, a = cv2.split(img_cv)
        hsv = cv2.cvtColor(cv2.merge([b,g,r]), cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, int(params['makeup'] * 1.5))
        v = cv2.add(v, int(params['makeup'] * 0.5))
        final_hsv = cv2.merge([h, s, v])
        final_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        fb, fg, fr = cv2.split(final_bgr)
        img_cv = cv2.merge([fb, fg, fr, a])

    # 6. Chỉnh Levels (Blacks / Whites)
    if params['blacks'] > 0 or params['whites'] > 0:
        img_cv = adjust_levels(img_cv, params['blacks'], params['whites'])
    
    # 7. Làm sắc nét thông minh (Smart Sharpen / Unsharp Mask)
    if params['sharp_amount'] > 0:
        img_cv = apply_unsharp_mask(img_cv, params['sharp_amount'])

    # Convert back to PIL for Contrast/Brightness (PIL is faster/better for this)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA))
    
    if params['exposure'] != 1.0:
        img_pil = ImageEnhance.Brightness(img_pil).enhance(params['exposure'])
    if params['contrast'] != 1.0:
        img_pil = ImageEnhance.Contrast(img_pil).enhance(params['contrast'])

    return img_pil

def create_print_layout(img_person, size_type):
    PAPER_W, PAPER_H = 1748, 1181 # A6 300 DPI
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
    
    input_method = st.radio("Nguồn ảnh:", ["📁 Tải ảnh lên", "📷 Chụp ảnh"], horizontal=True)
    input_file = None
    if input_method == "📁 Tải ảnh lên":
        input_file = st.file_uploader("Chọn ảnh từ máy", type=['jpg', 'png', 'jpeg'])
    else:
        input_file = st.camera_input("Chụp ảnh ngay")

    st.subheader("2. Cắt & Xoay")
    size_option = st.radio("Kích thước:", ["4x6 cm (Hộ chiếu)", "3x4 cm (Giấy tờ)"])
    target_ratio = 3/4 if "3x4" in size_option else 4/6
    
    st.info("💡 Hệ thống tự động xoay mặt theo mắt. Dùng thanh trượt dưới để chỉnh thêm nếu chưa chuẩn.")
    manual_rot = st.slider("Chỉnh nghiêng đầu (Thủ công):", -15.0, 15.0, 0.0, 0.5)
    
    bg_name = st.radio("Màu nền:", ["Trắng", "Xanh Chuẩn", "Xanh Nhạt"], horizontal=True)
    bg_map = {"Trắng": (255, 255, 255, 255), "Xanh Chuẩn": (66, 135, 245, 255), "Xanh Nhạt": (135, 206, 250, 255)}
    bg_val = bg_map.get(bg_name)

    # --- XỬ LÝ ẢNH ĐẦU VÀO ---
    if input_file:
        current_file_key = f"{input_file.name}_{input_file.size}"
        if 'current_file_key' not in st.session_state or st.session_state.current_file_key != current_file_key:
            with st.spinner('Đang tách nền & nhận diện...'):
                st.session_state.raw_nobg = process_raw_to_nobg(input_file)
                st.session_state.current_file_key = current_file_key
        
        if 'raw_nobg' in st.session_state:
            final_crop, debug_info, _ = crop_final_image(st.session_state.raw_nobg, manual_rot, target_ratio)
            if final_crop:
                st.session_state.base = final_crop
                st.caption(f"ℹ️ {debug_info}")
            else:
                st.error(f"Lỗi: {debug_info}")

    st.markdown("---")
    
    # --- PHẦN 3: LÀM ĐẸP & AI STYLE ---
    c_head, c_btn = st.columns([3, 2])
    with c_head:
        st.subheader("3. Xử lý ảnh")
    with c_btn:
        st.button("🔄 Reset", on_click=reset_beauty_params)

    # --- TÍNH NĂNG AI STYLE ---
    with st.expander("🤖 AI Style (Tự động)", expanded=False):
        ai_enabled = st.checkbox("Bật chế độ AI Preset", key='ai_enabled')
        if ai_enabled:
            gender_style = st.radio("Phong cách:", ["Nam (Rõ nét, Tương phản)", "Nữ (Mịn da, Sáng hồng)"])
            if gender_style == "Nam (Rõ nét, Tương phản)":
                st.session_state.val_smooth = 5
                st.session_state.val_makeup = 2
                st.session_state.val_exposure = 1.05
                st.session_state.val_contrast = 1.15
                st.session_state.val_sharp_amount = 15 # Nét cao
                st.session_state.val_denoise = 5
                st.session_state.val_blacks = 10       # Đen sâu
                st.session_state.val_whites = 5
                st.session_state.val_dehaze = 5
            else:
                st.session_state.val_smooth = 25
                st.session_state.val_makeup = 20
                st.session_state.val_exposure = 1.1
                st.session_state.val_contrast = 1.05
                st.session_state.val_sharp_amount = 8
                st.session_state.val_denoise = 10
                st.session_state.val_blacks = 0
                st.session_state.val_whites = 15       # Trắng sáng

    # --- SLIDER THỦ CÔNG ---
    with st.expander("✨ Công cụ chỉnh sửa (Mới)", expanded=True):
        st.markdown("**1. Chi tiết & Xóa mờ**")
        p_sharp_amount = st.slider("Độ sắc nét (Smart Sharpen)", 0, 30, st.session_state.get('val_sharp_amount', 0), key="val_sharp_amount", help="Làm nét ảnh bị out nét hoặc mờ")
        p_dehaze = st.slider("Xóa lớp phủ mờ (Dehaze)", 0, 30, st.session_state.get('val_dehaze', 0), key="val_dehaze", help="Loại bỏ lớp sương mờ")
        p_denoise = st.slider("Giảm nhiễu hạt (Denoise)", 0, 20, st.session_state.get('val_denoise', 0), key="val_denoise", help="Làm sạch ảnh bị noise/sạn")

        st.markdown("**2. Ánh sáng & Màu sắc**")
        col_b, col_w = st.columns(2)
        with col_b:
            p_blacks = st.slider("Nâng màu Đen", 0, 50, st.session_state.get('val_blacks', 0), key="val_blacks", help="Làm đậm vùng tối")
        with col_w:
            p_whites = st.slider("Nâng màu Trắng", 0, 50, st.session_state.get('val_whites', 0), key="val_whites", help="Làm sáng vùng sáng")
            
        p_exposure = st.slider("Độ sáng tổng (Exposure)", 0.5, 1.5, st.session_state.get('val_exposure', 1.0), 0.05, key="val_exposure")
        p_contrast = st.slider("Tương phản", 0.5, 1.5, st.session_state.get('val_contrast', 1.0), 0.05, key="val_contrast")
        
        st.markdown("**3. Da & Trang điểm**")
        p_smooth = st.slider("Mịn da", 0, 30, st.session_state.get('val_smooth', 0), key="val_smooth")
        p_makeup = st.slider("Hồng hào", 0, 50, st.session_state.get('val_makeup', 0), key="val_makeup")
        p_temp = st.slider("Nhiệt độ màu", -50, 50, st.session_state.get('val_temp', 0), key="val_temp")

    params = {
        'smooth': p_smooth, 'makeup': p_makeup,
        'exposure': p_exposure, 'contrast': p_contrast, 'temp': p_temp,
        'sharp_amount': p_sharp_amount, 'dehaze': p_dehaze,
        'blacks': p_blacks, 'whites': p_whites, 'denoise': p_denoise
    }

with col2:
    st.header(f"🖼 Kết quả ({size_option})")
    
    if 'base' in st.session_state and st.session_state.base:
        with st.spinner("Đang áp dụng hiệu ứng nâng cao..."):
            final_person = apply_advanced_effects(st.session_state.base, params)
        
        w, h = final_person.size
        final_img = Image.new("RGBA", (w, h), bg_val)
        final_img.paste(final_person, (0, 0), final_person)
        final_rgb = final_img.convert("RGB")
        
        st.image(final_rgb, width=350, caption="Ảnh hoàn thiện")
        
        # --- DOWNLOAD ---
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        buf = io.BytesIO()
        final_rgb.save(buf, format="JPEG", quality=100, dpi=(300, 300))
        
        name_mapping = {"Trắng": "white", "Xanh Chuẩn": "blue_standard", "Xanh Nhạt": "blue_light"}
        safe_bg_name = name_mapping.get(bg_name, "custom")
        
        c1.download_button(
            label="⬇️ Tải ảnh JPEG", 
            data=buf.getvalue(), 
            file_name=f"anh_the_{safe_bg_name}.jpg", 
            mime="image/jpeg"
        )

        if c2.button("🖨️ In ghép khổ A6"):
            paper, qty = create_print_layout(final_rgb, size_option)
            st.image(paper, caption=f"Layout in: {qty} ảnh", use_container_width=True)
            buf_p = io.BytesIO()
            paper.save(buf_p, format="JPEG", quality=100, dpi=(300, 300))
            st.download_button("⬇️ Tải file in", buf_p.getvalue(), "layout_in_A6.jpg", "image/jpeg", key='dl_print')
            
    else:
        st.info("👈 Hãy chọn ảnh ở cột bên trái để bắt đầu xử lý.")
