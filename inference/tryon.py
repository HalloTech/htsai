# hts-ai/inference/tryon.py

from PIL import Image
import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'leffa_utils')))


from leffa.inference import LeffaInference
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose


class LeffaPredictor:
    def __init__(self):
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp"
        )
        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl"
        )
        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx"
        )
        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth"
        )
        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)
        self.transform = LeffaTransform()

    def run_tryon(self, user_path, cloth_path):
        # Load and preprocess
        user_image = Image.open(user_path).convert("RGB")
        user_image = resize_and_center(user_image, 768, 1024)

        cloth_image = Image.open(cloth_path).convert("RGB")
        cloth_image = resize_and_center(cloth_image, 768, 1024)

        # Mask + pose
        model_parse, _ = self.parsing(user_image.resize((384, 512)))
        keypoints = self.openpose(user_image.resize((384, 512)))
        mask = get_agnostic_mask_hd(model_parse, keypoints, "upper_body").resize((768, 1024))

        densepose_arr = self.densepose_predictor.predict_seg(np.array(user_image))[:, :, ::-1]
        densepose = Image.fromarray(densepose_arr)

        data = {
            "src_image": [user_image],
            "ref_image": [cloth_image],
            "mask": [mask],
            "densepose": [densepose],
        }

        data = self.transform(data)

        result = self.vt_inference_hd(
            data,
            ref_acceleration=False,
            num_inference_steps=30,
            guidance_scale=2.5,
            seed=42,
            repaint=False,
        )

        return result["generated_image"][0]
