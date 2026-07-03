# ==============================================================================
# CHẾ ĐỘ SỐ LƯỢNG LỚN (KHOẢNG CÁCH ẢNH ĐƯỢC THU GỌN XUỐNG CÒN CỐ ĐỊNH 0.6MM)
# ==============================================================================
if app_mode == "👥 Ghép In Hàng Loạt (Số lượng lớn)":
    st.info("⚙️ Giao diện module xử lý hồ sơ hàng loạt - Khoảng cách ảnh siêu sát (0.6mm)")
    
    html_code = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%); 
            display: flex; justify-content: center; align-items: center; 
            min-height: 100vh; margin: 0; padding: 20px;
        }
        .container { 
            background: #ffffff; padding: 35px; border-radius: 20px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.08); max-width: 850px; width: 100%; text-align: center;
        }
        h2 { color: #2c3e50; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 25px; margin-top: 0;}
        .upload-group { display: flex; justify-content: space-between; gap: 20px; margin-bottom: 20px;}
        .person-box { 
            flex: 1; border: 2px dashed #b8c2cc; padding: 15px 10px; border-radius: 14px; 
            background: #fafafa; transition: all 0.3s ease; position: relative; text-align: center;
        }
        .person-box:hover { border-color: #007bff; background: #f0f7ff;}
        .person-box h4 { margin: 0 0 10px 0; color: #0056b3; font-size: 15px; font-weight: 700;}
        .name-input {
            width: 85%; padding: 6px 10px; margin: 8px auto; border: 1px solid #ced4da;
            border-radius: 6px; font-size: 13px; outline: none; text-align: center; display: block;
        }
        .name-input:focus { border-color: #007bff; box-shadow: 0 0 4px rgba(0,123,255,0.2); }
        .qty-area {
            margin-top: 10px; background: #eee; padding: 10px; border-radius: 8px;
            display: flex; flex-direction: column; gap: 8px;
        }
        .qty-row { display: flex; justify-content: space-between; align-items: center; font-size: 13px; font-weight: bold; color: #444;}
        .qty-row input { width: 50px; text-align: center; padding: 4px; border-radius: 4px; border: 1px solid #ccc; font-weight: bold;}
        .badge { color: white; padding: 3px 6px; border-radius: 4px; font-size: 11px;}
        .bg-3x4 { background: #007bff; }
        .bg-4x6 { background: #28a745; }
        input[type="file"] { display: none; }
        .custom-file-upload { 
            display: inline-block; padding: 8px 12px; cursor: pointer; background-color: #edf2f7; 
            color: #4a5568; border-radius: 8px; font-weight: 600; font-size: 12px; 
            border: 1px solid #e2e8f0; width: 85%; margin: 0 auto;
        }
        .custom-file-upload:hover { background-color: #e2e8f0; }
        .img-wrapper { position: relative; display: inline-block; margin-top: 10px; }
        .preview { 
            max-width: 80px; max-height: 100px; border-radius: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); 
            border: 2px solid #fff; display: none; object-fit: cover;
        }
        .clear-btn { 
            position: absolute; top: -8px; right: -8px; background: #ff4757; color: white; 
            border: none; border-radius: 50%; width: 20px; height: 20px; font-size: 10px; 
            font-weight: bold; cursor: pointer; display: none; align-items: center; justify-content: center;
        }
        .btn-group { display: flex; gap: 12px; justify-content: center; margin-top: 30px;}
        .btn { 
            border-radius: 50px; padding: 15px 20px; font-size: 14px; font-weight: 700; 
            text-transform: uppercase; letter-spacing: 0.5px; cursor: pointer; color: white; border: none; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1); transition: all 0.3s ease; flex: 1; 
        }
        #previewBtn { background: linear-gradient(135deg, #36D1DC 0%, #5B86E5 100%); }
        #downloadBtn { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); display: none; }
        #directPrintBtn { background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%); display: none; }
        #previewContainer { display: none; margin-top: 35px; border-top: 2px dashed #e2e8f0; padding-top: 25px; }
        #previewContainer h4 { color: #4a5568; margin-bottom: 20px; font-weight: 700;}
        .a4-page-preview {
            position: relative; width: 100%; max-width: 480px; background: white; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin: 0 auto 30px auto; 
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
        <h2>HỆ THỐNG XẾP IN HỒ SƠ CAO CẤP SMART STUDIO</h2>
        
        <!-- Hàng 1: Người 1 & 2 -->
        <div class="upload-group">
            <div class="person-box">
                <h4>👤 Người thứ 1</h4>
                <input type="text" id="name1" class="name-input" placeholder="Nhập tên học viên...">
                <label for="imgInput1" class="custom-file-upload" id="labelInput1">📁 Chọn Ảnh...</label>
                <input type="file" id="imgInput1" accept="image/png, image/jpeg, image/jpg">
                <center><div class="img-wrapper"><img id="preview1" class="preview" alt="Preview 1"><button id="clearBtn1" class="clear-btn">✖</button></div></center>
                <div class="qty-area">
                    <div class="qty-row"><span><span class="badge bg-3x4">3x4</span> SL:</span><input type="number" id="qty3x4_1" value="9" min="0" max="24"></div>
                    <div class="qty-row"><span><span class="badge bg-4x6">4x6</span> SL:</span><input type="number" id="qty4x6_1" value="0" min="0" max="24"></div>
                </div>
            </div>
            <div class="person-box">
                <h4>👤 Người thứ 2</h4>
                <input type="text" id="name2" class="name-input" placeholder="Nhập tên học viên...">
                <label for="imgInput2" class="custom-file-upload" id="labelInput2">📁 Chọn Ảnh...</label>
                <input type="file" id="imgInput2" accept="image/png, image/jpeg, image/jpg">
                <center><div class="img-wrapper"><img id="preview2" class="preview" alt="Preview 2"><button id="clearBtn2" class="clear-btn">✖</button></div></center>
                <div class="qty-area">
                    <div class="qty-row"><span><span class="badge bg-3x4">3x4</span> SL:</span><input type="number" id="qty3x4_2" value="9" min="0" max="24"></div>
                    <div class="qty-row"><span><span class="badge bg-4x6">4x6</span> SL:</span><input type="number" id="qty4x6_2" value="0" min="0" max="24"></div>
                </div>
            </div>
        </div>
        
        <!-- Hàng 2: Người 3 & 4 -->
        <div class="upload-group">
            <div class="person-box">
                <h4>👤 Người thứ 3</h4>
                <input type="text" id="name3" class="name-input" placeholder="Nhập tên học viên...">
                <label for="imgInput3" class="custom-file-upload" id="labelInput3">📁 Chọn Ảnh...</label>
                <input type="file" id="imgInput3" accept="image/png, image/jpeg, image/jpg">
                <center><div class="img-wrapper"><img id="preview3" class="preview" alt="Preview 3"><button id="clearBtn3" class="clear-btn">✖</button></div></center>
                <div class="qty-area">
                    <div class="qty-row"><span><span class="badge bg-3x4">3x4</span> SL:</span><input type="number" id="qty3x4_3" value="0" min="0" max="24"></div>
                    <div class="qty-row"><span><span class="badge bg-4x6">4x6</span> SL:</span><input type="number" id="qty4x6_3" value="0" min="0" max="24"></div>
                </div>
            </div>
            <div class="person-box">
                <h4>👤 Người thứ 4</h4>
                <input type="text" id="name4" class="name-input" placeholder="Nhập tên học viên...">
                <label for="imgInput4" class="custom-file-upload" id="labelInput4">📁 Chọn Ảnh...</label>
                <input type="file" id="imgInput4" accept="image/png, image/jpeg, image/jpg">
                <center><div class="img-wrapper"><img id="preview4" class="preview" alt="Preview 4"><button id="clearBtn4" class="clear-btn">✖</button></div></center>
                <div class="qty-area">
                    <div class="qty-row"><span><span class="badge bg-3x4">3x4</span> SL:</span><input type="number" id="qty3x4_4" value="0" min="0" max="24"></div>
                    <div class="qty-row"><span><span class="badge bg-4x6">4x6</span> SL:</span><input type="number" id="qty4x6_4" value="0" min="0" max="24"></div>
                </div>
            </div>
        </div>

        <!-- Hàng 3: Người 5 & 6 -->
        <div class="upload-group">
            <div class="person-box">
                <h4>👤 Người thứ 5</h4>
                <input type="text" id="name5" class="name-input" placeholder="Nhập tên học viên...">
                <label for="imgInput5" class="custom-file-upload" id="labelInput5">📁 Chọn Ảnh...</label>
                <input type="file" id="imgInput5" accept="image/png, image/jpeg, image/jpg">
                <center><div class="img-wrapper"><img id="preview5" class="preview" alt="Preview 5"><button id="clearBtn5" class="clear-btn">✖</button></div></center>
                <div class="qty-area">
                    <div class="qty-row"><span><span class="badge bg-3x4">3x4</span> SL:</span><input type="number" id="qty3x4_5" value="0" min="0" max="24"></div>
                    <div class="qty-row"><span><span class="badge bg-4x6">4x6</span> SL:</span><input type="number" id="qty4x6_5" value="0" min="0" max="24"></div>
                </div>
            </div>
            <div class="person-box">
                <h4>👤 Người thứ 6</h4>
                <input type="text" id="name6" class="name-input" placeholder="Nhập tên học viên...">
                <label for="imgInput6" class="custom-file-upload" id="labelInput6">📁 Chọn Ảnh...</label>
                <input type="file" id="imgInput6" accept="image/png, image/jpeg, image/jpg">
                <center><div class="img-wrapper"><img id="preview6" class="preview" alt="Preview 6"><button id="clearBtn6" class="clear-btn">✖</button></div></center>
                <div class="qty-area">
                    <div class="qty-row"><span><span class="badge bg-3x4">3x4</span> SL:</span><input type="number" id="qty3x4_6" value="0" min="0" max="24"></div>
                    <div class="qty-row"><span><span class="badge bg-4x6">4x6</span> SL:</span><input type="number" id="qty4x6_6" value="0" min="0" max="24"></div>
                </div>
            </div>
        </div>

        <div class="btn-group">
            <button id="previewBtn" class="btn">👁️ Xem Trước Bản Xếp</button>
            <button id="downloadBtn" class="btn">⬇️ Tải Xuống PDF</button>
            <button id="directPrintBtn" class="btn">🖨️ Tiến Hành In</button>
        </div>
        <div id="previewContainer">
            <h4>📄 MÔ PHỎNG TRANG IN CHUẨN A4</h4>
            <div id="pdfIframeContainer"></div>
        </div>
    </div>
    <script>
        let data1 = null, type1 = 'JPEG';
        let data2 = null, type2 = 'JPEG';
        let data3 = null, type3 = 'JPEG';
        let data4 = null, type4 = 'JPEG';
        let data5 = null, type5 = 'JPEG';
        let data6 = null, type6 = 'JPEG';

        function handleImageUpload(inputId, previewId, clearBtnId, labelId, personNum) {
            document.getElementById(inputId).addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    let type = (file.type === 'image/png') ? 'PNG' : 'JPEG';
                    if(personNum === 1) type1 = type;
                    if(personNum === 2) type2 = type;
                    if(personNum === 3) type3 = type;
                    if(personNum === 4) type4 = type;
                    if(personNum === 5) type5 = type;
                    if(personNum === 6) type6 = type;

                    const reader = new FileReader();
                    reader.onload = function(event) {
                        if(personNum === 1) data1 = event.target.result;
                        if(personNum === 2) data2 = event.target.result;
                        if(personNum === 3) data3 = event.target.result;
                        if(personNum === 4) data4 = event.target.result;
                        if(personNum === 5) data5 = event.target.result;
                        if(personNum === 6) data6 = event.target.result;

                        const imgElement = document.getElementById(previewId);
                        imgElement.src = event.target.result;
                        imgElement.style.display = 'block';
                        document.getElementById(clearBtnId).style.display = 'flex';
                        document.getElementById(labelId).innerHTML = '🔄 Đổi Ảnh';
                    }
                    reader.readAsDataURL(file);
                }
            });
            document.getElementById(clearBtnId).addEventListener('click', function() {
                document.getElementById(inputId).value = "";
                document.getElementById(previewId).style.display = 'none';
                document.getElementById(previewId).src = "";
                this.style.display = 'none';
                document.getElementById(labelId).innerHTML = '📁 Chọn Ảnh...';
                if(personNum === 1) data1 = null;
                if(personNum === 2) data2 = null;
                if(personNum === 3) data3 = null;
                if(personNum === 4) data4 = null;
                if(personNum === 5) data5 = null;
                if(personNum === 6) data6 = null;

                document.getElementById('previewContainer').style.display = 'none';
                document.getElementById('downloadBtn').style.display = 'none';
                document.getElementById('directPrintBtn').style.display = 'none';
            });
        }

        handleImageUpload('imgInput1', 'preview1', 'clearBtn1', 'labelInput1', 1);
        handleImageUpload('imgInput2', 'preview2', 'clearBtn2', 'labelInput2', 2);
        handleImageUpload('imgInput3', 'preview3', 'clearBtn3', 'labelInput3', 3);
        handleImageUpload('imgInput4', 'preview4', 'clearBtn4', 'labelInput4', 4);
        handleImageUpload('imgInput5', 'preview5', 'clearBtn5', 'labelInput5', 5);
        handleImageUpload('imgInput6', 'preview6', 'clearBtn6', 'labelInput6', 6);

        function getPersonsData() {
            let list = [];
            let dArr = [null, data1, data2, data3, data4, data5, data6];
            let tArr = [null, type1, type2, type3, type4, type5, type6];
            for(let i=1; i<=6; i++) {
                let q3x4 = parseInt(document.getElementById(`qty3x4_${i}`).value) || 0;
                let q4x6 = parseInt(document.getElementById(`qty4x6_${i}`).value) || 0;
                let pName = document.getElementById(`name${i}`).value.trim();
                if (dArr[i] && (q3x4 > 0 || q4x6 > 0)) {
                    list.push({ data: dArr[i], type: tArr[i], qty3x4: q3x4, qty4x6: q4x6, name: pName });
                }
            }
            return list;
        }

        function buildLayoutData(persons) {
            const a4W = 210, a4H = 297;
            let gapX = 0.6, gapY = 0.6, marginX = 10, marginY = 15;
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
            document.getElementById('downloadBtn').style.display = 'inline-block';
            document.getElementById('directPrintBtn').style.display = 'inline-block';
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

        document.getElementById('downloadBtn').addEventListener('click', function() { generateJsPDFObject().save('SmartStudio_Print_Layout.pdf'); });
        document.getElementById('directPrintBtn').addEventListener('click', function() {
            let doc = generateJsPDFObject(); const blobUrl = doc.output('bloburl'); const printWindow = window.open(blobUrl, '_blank');
            if (printWindow) { printWindow.onload = function() { printWindow.focus(); printWindow.print(); }; }
            else { alert("Vui lòng cho phép Pop-up trên trình duyệt để in trực tiếp!"); }
        });
    </script>
</body>
</html>"""
    
    # ⚠️ Chú ý: Đã tăng thuộc tính height từ 1950 lên 2350 để phù hợp với 3 hàng ảnh (6 người)
    components.html(html_code, height=2350, scrolling=True)
    st.stop()
