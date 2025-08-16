import argparse
import torch

from model import MattingNetwork


class StaticExporter:
    def __init__(self):
        self.parse_args()
        self.init_model()
        self.export()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
        parser.add_argument('--model-refiner', type=str, default='deep_guided_filter',
                            choices=['deep_guided_filter', 'fast_guided_filter'])
        parser.add_argument('--checkpoint', type=str, required=False)
        parser.add_argument('--output', type=str, required=True)
        self.args = parser.parse_args()

    def init_model(self):
        self.precision = torch.float32
        self.device = torch.device("cpu")
        self.model = MattingNetwork(
            self.args.model_variant,
            self.args.model_refiner
        ).eval().to(self.device, self.precision)

        if self.args.checkpoint is not None:
            self.model.load_state_dict(
                torch.load(self.args.checkpoint, map_location=self.device), strict=False)

    def export(self):
        print(self.args)
        # rec = (torch.zeros([1, 1, 1, 1]).to(self.args.device, self.precision),) * 4
        src = torch.randn(1, 3, 1080, 1920).to(self.device, self.precision)  # h=720 w=1280
        # r1i = torch.randn(1, 16, 135, 240).to(self.device, self.precision)
        # r2i = torch.randn(1, 20, 68, 120).to(self.device, self.precision)
        # r3i = torch.randn(1, 40, 34, 60).to(self.device, self.precision)
        # r4i = torch.randn(1, 64, 17, 30).to(self.device, self.precision)
        r1i = torch.randn(1, 16, 216, 384).to(self.device, self.precision)
        r2i = torch.randn(1, 20, 108, 192).to(self.device, self.precision)
        r3i = torch.randn(1, 40, 54, 96).to(self.device, self.precision)
        r4i = torch.randn(1, 64, 27, 48).to(self.device, self.precision)
        # 假设你的model的forward方法，已经指定downsample_ratio的默认值为torch.tensor([0.125]).
        # downsample_ratio = torch.tensor([0.375]).to(self.args.device)

        torch.onnx.export(
            self.model,
            (src, r1i, r2i, r3i, r4i),  # 不需要导出downsample_ratio，直接使用默认值
            self.args.output,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['img', 's1i', 's2i', 's3i', 's4i'],  # 不需要导出downsample_ratio，直接使用默认值
            output_names=['fgr', 'alp', 's1o', 's2o', 's3o', 's4o'])


if __name__ == '__main__':
    StaticExporter()