import argparse
import albumentations as albu
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from retinaface.box_utils import decode, decode_landm
from retinaface.network import RetinaFace
from retinaface.prior_box import priorbox
from retinaface.utils import tensor_from_rgb_image
from collections import OrderedDict
import warnings
import os
import onnx

warnings.filterwarnings('ignore')

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".ckpt"):
            path_to_model = (os.path.join(root, file))
            
state_dict = torch.load(path_to_model, map_location="cpu")
new_state_dict = OrderedDict()
for k, v in state_dict["state_dict"].items():
    name = k[6:]  # remove `module.`
    new_state_dict[name] = v

class ModelToOnnx(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = RetinaFace(
            name="Resnet50",
            pretrained=False,
            return_layers={"layer2": 1, "layer3": 2, "layer4": 3},
            in_channels=256,
            out_channels=256,
        )
        self.model.load_state_dict(new_state_dict)
        self.nms_threshold: float = 0.4
        self.variance = [0.1, 0.2]
        self.confidence_threshold: float = 0.7
    
    def forward(self, x: torch.Tensor):
        prior_box = priorbox(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
            image_size=x.shape[1:3],
        )
        scale_landmarks = torch.from_numpy(np.tile((x.shape[2], x.shape[1]), 5))
        scale_bboxes = torch.from_numpy(np.tile((x.shape[2], x.shape[1]), 2))
        loc, conf, land = self.model(x.unsqueeze(0))
        conf = F.softmax(conf, dim=-1)

        boxes = decode(loc.data[0], prior_box, self.variance)
        boxes *= scale_bboxes
        scores = conf[0][:, 1]

        landmarks = decode_landm(land.data[0], prior_box, self.variance)
        landmarks *= scale_landmarks

        # ignore low scores
        valid_index = torch.where(scores > self.confidence_threshold)[0]
        boxes = boxes[valid_index]
        landmarks = landmarks[valid_index]
        scores = scores[valid_index]

        # do NMS
        keep = nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep, :]

        landmarks = landmarks[keep]
        scores = scores[keep]
        return boxes, scores, landmarks


def prepare_image(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
    return albu.Compose([albu.LongestMaxSize(max_size=max_size), albu.Normalize(p=1)])(image=image)["image"]
     


def main() -> None:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "-ms",
        "--max_size",
        type=int,
        help="Size of the input image. The onnx model will predict on (max_size, max_size)",
        required=True,
    )
    arg("-o", "--output_file", type=str, help="Path to save onnx model.", required=True)
    args = parser.parse_args()

    raw_image = cv2.imread("imgs/example_01.jpg")
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    image = prepare_image(raw_image, args.max_size)
    x = tensor_from_rgb_image(image).float()
    model = ModelToOnnx()
    model.eval()
    torch.onnx.export(
        model,
        x,
        args.output_file,
        verbose=False,
        opset_version=12,
        input_names=["input"],
        export_params=True,
        do_constant_folding=True,
    )

    # Load the ONNX model
    model = onnx.load(args.output_file)
    # Check that the model is well formed
    onnx.checker.check_model(model)

if __name__ == "__main__":
    main()