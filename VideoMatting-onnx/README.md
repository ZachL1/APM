# Donwload weight
https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0

# Export ONNX

## 导出动态 ONNX

参考 RVM 官方仓库的 [onnx 分支](https://github.com/PeterL1n/RobustVideoMatting/tree/onnx)。

- 使用 `inference_onnx.py` 测试动态 ONNX：
  ```bash
  python ./inference_onnx.py --input ./demo/TEST_02.mp4 --output ./demo/TEST_02_onnx.mp4

## 导出静态 ONNX

> 参考 [🔧填坑: RobustVideoMatting(5k+🔥 star)视频抠图静态ONNX模型转换](https://zhuanlan.zhihu.com/p/459088407)

- 修改model.py跳过不必要的自定义op
    ```python
        def forward(self, src, r1, r2, r3, r4,
                downsample_ratio: float = 0.375,  # 注意，需要一个默认值，直接导出成ONNX常量
                segmentation_pass: bool = False):
        
        if torch.onnx.is_in_onnx_export():
            # 如果是为了导出静态的ONNX 可以不使用该自定义算子
            # src_sm = CustomOnnxResizeByFactorOp.apply(src, downsample_ratio)
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
    ```
- PyTorch >= 1.10 不需要修改源代码
- 使用 `export_onnx_static.py` 脚本导出：
  ```bash
  python ./export_onnx_static.py --model-variant mobilenetv3 --checkpoint weight/rvm_mobilenetv3.pth --output weight/rvm_mobilenetv3_1080x1920.onnx
  ```
- 使用 `inference_onnx_static.py` 测试静态 ONNX：
  ```bash
  python3 ./inference_onnx_static.py --input ./demo/TEST_02.mp4 --output ./demo/TEST_02_0.25_onnx.mp4
  ```

# Export OpenVINO

- 配置好 OpenVINO 环境后导出：
  ```bash
  python mo_onnx.py --input_model .onnx --output_dir IR --input src,r1i,r2i,r3i,r4i --input_shape "[1,3,1080,1920],[1,16,68,120],[1,20,34,60],[1,40,17,30],[1,64,9,15]"
  ```

- 使用 `inference_openvino.py` 测试 OpenVINO：
  ```bash
  python3 ./inference_openvino.py --input ./demo/TEST_02.mp4 --output ./demo/TEST_02_0.25_ov.mp4
  ```