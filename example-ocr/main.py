import os
from rapidocr_onnxruntime import RapidOCR

img = "tests/test1.jpg"
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, img)

engine = RapidOCR()
result, elapse = engine(img_path)

for item in result:
    print(f"score={item[2]} \t text={str(item[1])}")
