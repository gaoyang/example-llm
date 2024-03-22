import os
import platform
from rapidocr_onnxruntime import RapidOCR
from rapidocr_paddle import RapidOCR as RapidOCRGpu

# import paddle
# paddle.utils.run_check()

def test_cpu(img_path):
    engine = RapidOCR()
    result, elapse = engine(img_path)
    for item in result:
        print(f"score={item[2]} \t text={str(item[1])}")
    print(f"Elapsed time (CPU): {elapse}")


def test_gpu(img_path):
    engine = RapidOCRGpu(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
    result, elapse = engine(img_path)
    for item in result:
        print(f"score={item[2]} \t text={str(item[1])}")
    print(f"Elapsed time (GPU): {elapse}")


img = "tests/test1.jpg"
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, img)

print("Testing CPU version...")
test_cpu(img_path)

if platform.system() != "Darwin":
    print("Testing GPU version...")
    test_gpu(img_path)
else:
    print("Skipping GPU version test on Mac...")
