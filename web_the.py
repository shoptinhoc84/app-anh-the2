import streamlit as st
import streamlit.components.v1 as components

# --- 1. CẤU HÌNH TRANG VÀ GIAO DIỆN ---
st.set_page_config(
    page_title="In Ảnh Thẻ Hàng Loạt - SHOPTINHOC", 
    layout="wide", 
    page_icon="🖨️"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }
    .brand-container {
        text-align: center;
        padding: 15px 10px;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    .main-title {
        font-size: 1.5rem;
        color: #ffffff;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 0.85rem;
        color: #bfdbfe;
        margin-top: 5px;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. TIÊU ĐỀ TRANG ---
st.markdown("""
<div class="brand-container">
    <div class="main-title">🖨️ TỰ ĐỘNG XẾP IN ẢNH THẺ HÀNG LOẠT</div>
    <div class="sub-title">Hệ thống ghép khổ A4 tối ưu khoảng cách dành cho 10 người / học viên</div>
</div>
""", unsafe_allow_html=True)

# --- 3. MÃ HTML/JS CHỨC NĂNG IN HÀNG LOẠT (TỐI ƯU MOBILE) ---
html_code = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
            background: #f4f6f8; 
            margin: 0; padding: 10px;
            display: flex; justify-content: center;
        }
        .container { 
            background: #ffffff; padding: 15px; border-radius: 16px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.06); max-width: 850px; width: 100%; text-align: center;
        }
        h2 { color: #2c3e50; font-weight: 800; text-transform: uppercase; font-size: 16px; margin-bottom: 15px; margin-top: 5px;}
        
        .upload-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        @media (max-width: 600px) {
            .upload-grid { grid-template-columns: 1fr; }
        }

        .person-box { 
            border: 1.5px dashed #b8c2cc; padding: 10px; border-radius: 12px; 
            background: #fafafa; text-align: center; position: relative;
        }
        .person-box h4 { margin: 0 0 6px 0; color: #0056b3; font-size: 14px; font-weight: 700;}
        .name-input {
            width: 90%; padding: 8px; margin: 4px auto; border: 1px solid #ced4da;
            border-radius: 6px; font-size: 13px; outline: none; text-align: center; display: block;
        }
        
        /* Tối ưu Upload File cho Mobile: Ẩn ẩn bằng opacity thay vì display:none */
        .file-upload-wrapper {
            position: relative;
            display: inline-block;
            width: 90%;
            margin: 4px auto;
        }
        .custom-file-upload { 
            display: block; padding: 8px 10px; cursor: pointer; background-color: #edf2f7; 
            color: #4a5568; border-radius: 8px; font-weight: 600; font-size: 12px; 
            border: 1px solid #cbd5e0; text-align: center;
        }
        .real-file-input {
            position: absolute; left: 0; top: 0; width: 100%; height: 100%;
            opacity: 0; cursor: pointer; font-size: 0;
        }

        .qty-area {
            margin-top: 6px; background: #edf2f7; padding: 6px; border-radius: 8px;
            display: flex; flex-direction: column; gap: 4px;
        }
        .qty-row { display: flex; justify-content: space-between; align-items: center; font-size: 12px; font-weight: bold; color: #444;}
        .qty-row input { width: 55px; text-align: center; padding: 4px; border-radius: 4px; border: 1px solid #ccc; font-size: 13px; font-weight: bold;}
        .badge { color: white; padding: 2px 5px; border-radius: 4px; font-size: 10px;}
        .bg-3x4 { background: #007bff; }
        .bg-4x6 { background: #28a745; }

        .img-wrapper { position: relative; display: inline-block; margin-top: 6px; }
        .preview { 
            max-width: 70px; max-height: 90px; border-radius: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); 
            border: 2px solid #fff; display: none; object-fit: cover;
        }
        .clear-btn { 
            position: absolute; top: -6px; right: -6px; background: #ff4757; color: white; 
            border: none; border-radius: 50%; width: 22px; height: 22px; font-size: 11px; 
            font-weight: bold; cursor: pointer; display: none; align-items: center; justify-content: center;
            z-index: 10;
        }
        .btn-group { display: flex; flex-direction: column; gap: 8px; margin-top: 20px;}
        @media (min-width: 600px) { .btn-group { flex-direction: row; } }
        
        .btn { 
            border-radius: 12px; padding: 12px 15px; font-size: 13px; font-weight: 700; 
            text-transform: uppercase; cursor: pointer; color: white; border: none; 
            box-shadow: 0 3px 10px rgba(0,0,0,0.1); transition: all 0.2s ease; flex: 1; 
        }
        #previewBtn { background: linear-gradient(135deg, #36D1DC 0%, #5B86E5 100%); }
        #downloadBtn { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); display: none; }
        #directPrintBtn { background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%); display: none; }
        #previewContainer { display: none; margin-top: 20px; border-top: 2px dashed #e2e8f0; padding-top: 15px; }
        #previewContainer h4 { color: #4a5568; margin-bottom: 15px; font-size: 14px;}
        .a4-page-preview {
            position: relative; width: 100%; max-width: 480px; background: white; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.15); margin: 0 auto 20px auto; 
            border: 1px solid #ccc; overflow: hidden; border-radius: 4px;
        }
        .label-text-style {
            position: absolute; width: 100%; text-align: center; color: #333;
            font-family: Arial, sans-serif; font-weight: bold; overflow: hidden; white-space: nowrap;
        }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
</head>
<body>
    <div class="container">
        <h2>DANH SÁCH ẢNH IN HỒ SƠ (10 NGƯỜI)</h2>
        
        <div class="upload-grid">
            <script>
                document.write(Array.from({length: 10}, (_, i) => {
                    const num = i + 1;
                    return `
                        <div class="person-box">
                            <h4>👤 Người thứ ${num}</h4>
                            <input type="text" id="name${num}" class="name-input" placeholder="Nhập tên học viên...">
                            <div class="file-upload-wrapper">
                                <span class="custom-file-upload" id="labelInput${num}">📁 Chọn Ảnh...</span>
                                <input type="file" id="imgInput${num}" class="real-file-input" accept="image/*">
                            </div>
                            <center><div class="img-wrapper"><img id="preview${num}" class="preview" alt="Preview ${num}"><button id="clearBtn${num}" class="clear-btn">✖</button></div></center>
                            <div class="qty-area">
                                <div class="qty-row"><span><span class="badge bg-3x4">3x4</span> SL:</span><input type="number" id="qty3x4_${num}" value="${num <= 2 ? 9 : 0}" min="0" max="24"></div>
                                <div class="qty-row"><span><span class="badge bg-4x6">4x6</span> SL:</span><input type="number" id="qty4x6_${num}" value="0" min="0" max="24"></div>
                            </div>
                        </div>
                    `;
                }).join(''));
            </script>
        </div>

        <div class="btn-group">
            <button id="previewBtn" class="btn">👁️ Xem Trước Bản Xếp</button>
            <button id="downloadBtn" class="btn">⬇️ TẢI XUỐNG PDF A4</button>
            <button id="directPrintBtn" class="btn">🖨️ TIẾN HÀNH IN</button>
        </div>
        <div id="previewContainer">
            <h4>📄 MÔ PHỎNG TRANG IN CHUẨN A4</h4>
            <div id="pdfIframeContainer"></div>
        </div>
    </div>

    <script>
        let dataStore = Array(11).fill(null);
        let typeStore = Array(11).fill('JPEG');

        function handleImageUpload(inputId, previewId, clearBtnId, labelId, personNum) {
            document.getElementById(inputId).addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    typeStore[personNum] = (file.type === 'image/png') ? 'PNG' : 'JPEG';
                    const reader = new FileReader();
                    reader.onload = function(event) {
                        dataStore[personNum] = event.target.result;
                        const imgElement = document.getElementById(previewId);
                        imgElement.src = event.target.result;
                        imgElement.style.display = 'block';
                        document.getElementById(clearBtnId).style.display = 'flex';
                        document.getElementById(labelId).innerHTML = '🔄 Đổi Ảnh';
                    }
                    reader.readAsDataURL(file);
                }
            });
            document.getElementById(clearBtnId).addEventListener('click', function(e) {
                e.stopPropagation();
                document.getElementById(inputId).value = "";
                document.getElementById(previewId).style.display = 'none';
                document.getElementById(previewId).src = "";
                this.style.display = 'none';
                document.getElementById(labelId).innerHTML = '📁 Chọn Ảnh...';
                dataStore[personNum] = null;

                document.getElementById('previewContainer').style.display = 'none';
                document.getElementById('downloadBtn').style.display = 'none';
                document.getElementById('directPrintBtn').style.display = 'none';
            });
        }

        for(let i = 1; i <= 10; i++) {
            handleImageUpload(`imgInput${i}`, `preview${i}`, `clearBtn${i}`, `labelInput${i}`, i);
        }

        function getPersonsData() {
            let list = [];
            for(let i = 1; i <= 10; i++) {
                let q3x4 = parseInt(document.getElementById(`qty3x4_${i}`).value) || 0;
                let q4x6 = parseInt(document.getElementById(`qty4x6_${i}`).value) || 0;
                let pName = document.getElementById(`name${i}`).value.trim();
                if (dataStore[i] && (q3x4 > 0 || q4x6 > 0)) {
                    list.push({ data: dataStore[i], type: typeStore[i], qty3x4: q3x4, qty4x6: q4x6, name: pName });
                }
            }
            return list;
        }

        function buildLayoutData(persons) {
            const a4W = 210, a4H = 297;
            let gapX = 0.6, gapY = 0.6, marginX = 5, marginY = 3;
            let pages = [], currentPage = [], curX = marginX, curY = marginY;
            let maxRowHeight = 0;

            let allItems = [];
            persons.forEach((person) => {
                for (let i = 0; i < person.qty3x4; i++) {
                    allItems.push({ data: person.data, type: person.type, w: 30, h: 40, name: person.name });
                }
                for (let i = 0; i < person.qty4x6; i++) {
                    allItems.push({ data: person.data, type: person.type, w: 40, h: 60, name: person.name });
                }
            });

            allItems.forEach((item) => {
                if (curX + item.w > a4W - marginX) {
                    curX = marginX;
                    curY += maxRowHeight + gapY;
                    maxRowHeight = 0;
                }

                if (curY + item.h > a4H - marginY) {
                    pages.push(currentPage);
                    currentPage = [];
                    curX = marginX;
                    curY = marginY;
                    maxRowHeight = 0;
                }

                currentPage.push({ data: item.data, type: item.type, x: curX, y: curY, w: item.w, h: item.h, name: item.name });

                if (item.h > maxRowHeight) {
                    maxRowHeight = item.h;
                }

                curX += item.w + gapX;
            });

            if (currentPage.length > 0) pages.push(currentPage);
            return pages;
        }

        document.getElementById('previewBtn').addEventListener('click', function() {
            let persons = getPersonsData();
            if (persons.length === 0) return alert("Vui lòng tải ảnh lên và nhập số lượng!");
            let pages = buildLayoutData(persons);
            let pagesHtml = '';

            pages.forEach(page => {
                pagesHtml += `<div class="a4-page-preview" style="aspect-ratio: 210/297; border: 1px solid #777; background:#fff; margin-bottom:20px; position:relative;">`;
                page.forEach(img => {
                    let pLeft = (img.x / 210) * 100 + '%';
                    let pTop = (img.y / 297) * 100 + '%';
                    let pWidth = (img.w / 210) * 100 + '%';
                    let pHeight = (img.h / 297) * 100 + '%';
                    pagesHtml += `<img src="${img.data}" style="position: absolute; left: ${pLeft}; top: ${pTop}; width: ${pWidth}; height: ${pHeight}; object-fit: cover; border: 1px solid #E5E5E5; box-sizing: border-box;">`;
                    if (img.name) {
                        let labelTop = ((img.y + img.h - 3.2) / 297) * 100 + '%';
                        let labelFontSize = (img.w === 30) ? '8px' : '9px';
                        pagesHtml += `<div class="label-text-style" style="left: ${pLeft}; top: ${labelTop}; font-size: ${labelFontSize}; background: rgba(255,255,255,0.8); height:13px; line-height:13px;">${img.name}</div>`;
                    }
                });
                pagesHtml += `</div>`;
            });
            document.getElementById('pdfIframeContainer').innerHTML = pagesHtml;
            document.getElementById('previewContainer').style.display = 'block';
            document.getElementById('downloadBtn').style.display = 'block';
            document.getElementById('directPrintBtn').style.display = 'block';
        });

        function generateJsPDFObject() {
            let persons = getPersonsData();
            const { jsPDF } = window.jspdf;
            let doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
            let pages = buildLayoutData(persons);
            pages.forEach((page, pageIdx) => {
                if (pageIdx > 0) doc.addPage();
                page.forEach(img => {
                    doc.addImage(img.data, img.type, img.x, img.y, img.w, img.h);
                    doc.setDrawColor(225, 225, 225); doc.setLineWidth(0.08); doc.rect(img.x, img.y, img.w, img.h, 'S');
                    if (img.name) {
                        doc.setFillColor(255, 255, 255); doc.rect(img.x + 0.2, img.y + img.h - 3.0, img.w - 0.4, 2.8, 'F');
                        doc.setTextColor(60, 60, 60); let fSize = (img.w === 30) ? 5.5 : 6.5;
                        doc.setFontSize(fSize); doc.setFont("Helvetica", "bold");
                        doc.text(img.name, img.x + (img.w / 2), img.y + img.h - 0.8, { align: 'center' });
                    }
                });
            });
            return doc;
        }

        document.getElementById('downloadBtn').addEventListener('click', function() { 
            generateJsPDFObject().save('SmartStudio_Print_Layout.pdf'); 
        });

        document.getElementById('directPrintBtn').addEventListener('click', function() {
            let doc = generateJsPDFObject(); 
            const blobUrl = doc.output('bloburl'); 
            const printWindow = window.open(blobUrl, '_blank');
            if (printWindow) { 
                printWindow.onload = function() { printWindow.focus(); printWindow.print(); }; 
            } else { 
                alert("Vui lòng cho phép Pop-up trên trình duyệt di động để in!"); 
            }
        });
    </script>
</body>
</html>"""

# Hiển thị giao diện với chiều cao điều chỉnh linh hoạt trên di động
components.html(html_code, height=2200, scrolling=True)
