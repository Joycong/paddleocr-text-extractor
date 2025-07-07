import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import ImageFont, ImageDraw, Image

# 🔤 폰트 경로
font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 20)

# 1. OCR 초기화 (predict 방식)
ocr = PaddleOCR(lang="korean", use_textline_orientation=False, text_det_box_thresh=0.3)

# 2. 이미지 padding
image_path = "test_picture.jpg"
img = cv2.imread(image_path)
padded_img = cv2.copyMakeBorder(
    img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255]
)
cv2.imwrite("temp_padded.jpg", padded_img)

# 3. predict 방식 사용
ocr_result = ocr.predict("temp_padded.jpg")[0]  # OCRResult 객체

# 4. 결과 추출 (dict 형태처럼 접근)
boxes = ocr_result["rec_polys"]
texts = ocr_result["rec_texts"]
scores = ocr_result["rec_scores"]

# 5. 이미지 변환 (PIL)
img_pil = Image.fromarray(cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img_pil)

# 6. 출력 파일 준비
output_txt = open("ocr_result.txt", "w", encoding="utf-8")
total_score = 0

for i, (box, text, score) in enumerate(zip(boxes, texts, scores)):
    try:
        poly = [tuple(p) for p in box]
        draw.line(poly + [poly[0]], fill=(0, 255, 0), width=2)
        draw.text(
            (poly[0][0], poly[0][1] - 22),
            f"[{score:.2f}] {text}",
            font=font,
            fill=(255, 0, 0),
        )
        output_txt.write(f"[{score:.2f}] {text}\n")
        total_score += score
    except Exception as e:
        print(f"❌ {i}번 오류: {e}")

# 7. 평균 정확도
avg_score = total_score / len(scores) if scores else 0
output_txt.write(f"\n📊 평균 정확도: {avg_score:.4f}\n")
output_txt.close()
print(f"📊 평균 정확도: {avg_score:.4f}")

# 8. 결과 저장
img_pil.save("recognized_result.jpg")
print("✅ 저장 완료: recognized_result.jpg / ocr_result.txt")
