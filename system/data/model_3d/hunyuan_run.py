import os
import sys
sys.path.insert(0, r"C:\Users\cristian\Documents\UNLZ-AI-STUDIO\system\3d-backends\Hunyuan3D-2")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
input_image = r"C:/Users/cristian/Pictures/Iconos/10080-0-1455866940.webp"
out_dir = r"C:\Users\cristian\Documents\UNLZ-AI-STUDIO\system\3d-out\model3d_2026-01-08_13-44-55"
os.makedirs(out_dir, exist_ok=True)
mesh_path = os.path.join(out_dir, "mesh.glb")
weights_dir = r"C:\Users\cristian\Documents\UNLZ-AI-STUDIO\system\3d-weights\Hunyuan3D-2"
local_ok = False
if weights_dir and os.path.exists(weights_dir):
    for name in (
        "hunyuan3d-dit-v2-0",
        "hunyuan3d-dit-v2-0-fast",
        "hunyuan3d-dit-v2-0-turbo",
    ):
        if os.path.exists(os.path.join(weights_dir, name)):
            local_ok = True
            break
    base = weights_dir if local_ok else "tencent/Hunyuan3D-2"
else:
    base = "tencent/Hunyuan3D-2"
if local_ok:
    os.environ["HF_HUB_OFFLINE"] = "1"
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(base)
mesh = pipeline(image=input_image)[0]
mesh.export(mesh_path)
paint = Hunyuan3DPaintPipeline.from_pretrained(base)
mesh = paint(mesh, image=input_image)
mesh.export(os.path.join(out_dir, "mesh_textured.glb"))
