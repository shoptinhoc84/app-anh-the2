import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import io
import gc
import streamlit.components.v1 as components

# --- BẢO VỆ CHỐNG SẬP KHI THIẾU THƯ VIỆN ---
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_detection as mp_face_detection
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# --- 1. CẤU HÌNH TRANG & CSS TRANG TRÍ ---
st.set_page_config(page_title="Studio Ảnh Thẻ SHOPTINHOC", layout="wide", page_icon="📸")

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        color: #B22222;
        text-align: center;
        font-weight: 800;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    div[data-testid="stButton"] > button:first-child {
        border-radius: 10px;
        font-weight: bold;
    }
    .image-container {
        border: 3px solid #B22222;
        padding: 10px;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOGIC HÀM ---
# ... (Các hàm khác giữ nguyên)

def create_pdf(photos, size_type):
    if not HAS_FPDF: return None
    
    # Thiết lập trang dọc A4 mặc định
    doc = FPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' })
    
    # Kích thước ảnh
    if "3x4" in size_type:
        img_w, img_h = 30, 40
    else:
        img_w, img_h = 40, 60
    
    # Số lượng ảnh cho mỗi người
    qty_per_person = 9
    
    # Vị trí bắt đầu xếp ảnh
    start_x, start_y = 10, 15
    
    # Khoảng cách giữa các ảnh
    gap_x, gap_y = 3, 3
    
    cur_x = start_x
    cur_y = start_y
    
    # Lặp qua từng người
    for person_num in range(1, 5):
        # Lấy ảnh của người tương ứng
        photo = next((p for p in photos if p['person_num'] == person_num), None)
        
        if photo:
            # Lặp qua từng ảnh của người tương ứng
            for i in range(qty_per_person):
                # Xuống hàng nếu vượt quá chiều ngang hữu dụng
                if cur_x + img_w > 210 - start_x:
                    cur_x = start_x
                    cur_y += img_h + gap_y
                
                # Sang trang mới nếu vượt quá chiều dọc hữu dụng
                if cur_y + img_h > 297 - start_y:
                    doc.add_page()
                    cur_x = start_x
                    cur_y = start_y
                
                # Vẽ ảnh vào PDF
                doc.add_image(photo['data'], photo['type'], cur_x, cur_y, img_w, img_h)
                
                # Tịnh tiến vị trí cột tiếp theo
                cur_x += img_w + gap_x
                
            # Xuống hàng sau mỗi nhóm 9 ảnh
            cur_x = start_x
            cur_y += img_h + gap_y
            
    return bytes(doc.output())

# --- 3. GIAO DIỆN CHÍNH ---
# ... (Phần giao diện giữ nguyên)

# ==============================================================================
# HOẠT ĐỘNG KHI CHỌN CHẾ ĐỘ GHÉP SỐ LƯỢNG LỚN (ĐÃ ĐƯỢC NÂNG CẤP TOÀN DIỆN)
# ==============================================================================
if app_mode == "👥 Tool Ghép In A4 (Số lượng lớn)":
    # ... (Phần HTML giữ nguyên)
    
    # XỬ LÝ NÚT XEM TRƯỚC (BẢN ĐỒ HOẠT ĐỘNG KHÍT THEO MA TRẬN ĐỘNG)
    document.getElementById('previewBtn').addEventListener('click', function() {
        // ... (Phần lấy danh sách ảnh hợp lệ giữ nguyên)
        
        const printSize = document.getElementById('printSize').value;
        
        // Quy đổi thông số mm giả lập trên khung tỉ lệ A4 đứng (210 x 297)
        const a4W = 210;
        const a4H = 297;
        
        let imgW = (printSize === '3x4') ? 30 : 40;
        let imgH = (printSize === '3x4') ? 40 : 60;
        let gapX = 3;
        let gapY = 3;
        let marginX = 10;
        let marginY = 15;

        let curX = marginX;
        let curY = marginY;
        
        let pagesHtml = '';
        let currentPageContent = '';

        function openPageBox() {
            return `<div class="a4-page-preview" style="aspect-ratio: 210/297; border: 1px solid #777; background:#fff; margin-bottom:20px; position:relative;">`;
        }

        currentPageContent = openPageBox();

        // Lặp qua từng người
        for (let personNum = 1; personNum <= 4; personNum++) {
            // Lấy ảnh của người tương ứng
            const photo = photos.find(p => p.personNum === personNum);
            
            if (photo) {
                // Lặp qua từng ảnh của người tương ứng
                for (let i = 0; i < 9; i++) {
                    // Kiểm tra xem ảnh tiếp theo có bị tràn chiều ngang không, nếu có thì xuống hàng
                    if (curX + imgW > a4W - marginX) {
                        curX = marginX;
                        curY += imgH + gapY;
                    }
                    
                    // Kiểm tra tràn trang đứng, nếu có thì ngắt trang mới
                    if (curY + imgH > a4H - marginY) {
                        currentPageContent += `</div>`; // Đóng trang cũ
                        pagesHtml += currentPageContent;
                        
                        currentPageContent = openPageBox(); // Mở trang mới
                        curX = marginX;
                        curY = marginY;
                    }

                    // Tính tỷ lệ % để hiển thị chuẩn xác trên CSS tuyệt đối
                    let pLeft = (curX / a4W) * 100 + '%';
                    let pTop = (curY / a4H) * 100 + '%';
                    let pWidth = (imgW / a4W) * 100 + '%';
                    let pHeight = (imgH / a4H) * 100 + '%';

                    currentPageContent += `<img src="${photo.data}" style="position: absolute; left: ${pLeft}; top: ${pTop}; width: ${pWidth}; height: ${pHeight}; object-fit: cover; border: 1px solid #E0E0E0; box-sizing: border-box;">`;
                    
                    // Tịnh tiến sang phải cho tấm ảnh tiếp theo
                    curX += imgW + gapX;
                }
                
                // Xuống hàng sau mỗi nhóm 9 ảnh
                curX = marginX;
                curY += imgH + gapY;
            }
        }

        currentPageContent += `</div>`; // Đóng trang cuối cùng
        pagesHtml += currentPageContent;

        document.getElementById('pdfIframeContainer').innerHTML = pagesHtml;
        document.getElementById('previewContainer').style.display = 'block';
        document.getElementById('downloadBtn').style.display = 'inline-block';
    });

    // XỬ LÝ IN PDF VỚI ĐƯỜNG CẮT (CUT LINES) CHUYÊN NGHIỆP
    document.getElementById('downloadBtn').addEventListener('click', function() {
        // ... (Phần lấy danh sách ảnh hợp lệ giữ nguyên)
        
        const printSize = document.getElementById('printSize').value;
        
        // ... (Phần thiết lập PDF giữ nguyên)
        
        // ... (Phần lặp qua từng người giữ nguyên)
        
        doc.save('Ghep_Anh_The_A4_SHOPTINHOC.pdf');
    });
}
