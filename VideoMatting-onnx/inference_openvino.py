import cv2
import time
import argparse
import numpy as np

from openvino.inference_engine import IECore


def normalize(frame: np.ndarray) -> np.ndarray:
    """
    Args:
        frame: BGR
    Returns: normalized 0~1 BCHW RGB
    """
    img = frame.astype(np.float32).copy() / 255.0
    img = img[:, :, ::-1]  # RGB
    img = np.transpose(img, (2, 0, 1))  # (C,H,W)
    img = np.expand_dims(img, axis=0)  # (B=1,C,H,W)
    return img


def infer_rvm_video(weight: str = "rvm_mobilenetv3_1080x1920.onnx",
                    video_path: str = "./demo/1917.mp4",
                    output_path: str = "./demo/1917_onnx.mp4"):

    core = IECore() # 初始化OpenVINO runtime引擎
    model = core.read_network(model=weight) # 装载模型
    compiled_model = core.load_network(network=model, device_name="CPU") # 将模型装载到设备
    print(f"Load {weight} done!")

    for _ in model.input_info:
        print("Input: ", _)
    for _ in model.outputs:
        print("Output: ", _)

    # 读取视频
    video_capture = cv2.VideoCapture(video_path)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Caputer: Height: {height}, Width: {width}, Frame Count: {frame_count}")

    # 写出视频
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Create Video Writer: {output_path}")

    i = 0
    r1i = np.zeros((1, 16, 68, 120), dtype=np.float32)
    r2i = np.zeros((1, 20, 34, 60), dtype=np.float32)
    r3i = np.zeros((1, 40, 17, 30), dtype=np.float32)
    r4i = np.zeros((1, 64, 9, 15), dtype=np.float32)
    bgr = np.array([0.47, 1., 0.6]).reshape((3, 1, 1))

    print(f"Infer {video_path} start ...")
    while video_capture.isOpened():
        success, frame = video_capture.read()

        if success:
            i += 1
            frame = cv2.resize(frame, (1920, 1080))  # (w, h)
            src = normalize(frame)
            # src 张量是 [B, C, H, W] 形状，必须用模型一样的 dtype
            t1 = time.time()
            fgr, pha, r1o, r2o, r3o, r4o = compiled_model.infer(inputs={
                'src': src,
                'r1i': r1i,
                'r2i': r2i,
                'r3i': r3i,
                'r4i': r4i,
            }).values()
            t2 = time.time()
            # 更新context
            r1i = r1o
            r2i = r2o
            r3i = r3o
            r4i = r4o

            print(f"Infer {i}/{frame_count} done! -> cost {(t2 - t1) * 1000} ms", end=" ")
            merge_frame = fgr * pha + bgr * (1. - pha)  # (1,3,H,W)
            merge_frame = merge_frame[0] * 255.  # (3,H,W)
            merge_frame = merge_frame.astype(np.uint8)  # RGB
            merge_frame = np.transpose(merge_frame, (1, 2, 0))  # (H,W,3)
            merge_frame = cv2.cvtColor(merge_frame, cv2.COLOR_BGR2RGB)

            # 调整输出的宽高
            merge_frame = cv2.resize(merge_frame, (width, height))

            video_writer.write(merge_frame)
            print(f"write {i}/{frame_count} done.")
        else:
            print("can not read video! skip!")
            break

    video_capture.release()
    video_writer.release()
    print(f"Infer {video_path} done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="video")
    parser.add_argument("--weight", type=str, default="./rvm_mobilenetv3_1080x1920.xml")
    parser.add_argument("--input", type=str, default="./demo/TEST_08.mp4")
    parser.add_argument("--output", type=str, default="./demo/TEST_08_py.mp4")
    args = parser.parse_args()

    infer_rvm_video(weight=args.weight, video_path=args.input, output_path=args.output)
    
    """
    PYTHONPATH=. python3 ./inference_onnx.py --input ./demo/1917.mp4 --output ./demo/1917_onnx.mp4
    PYTHONPATH=. python3 ./inference_onnx.py --mode img --input test.jpg --output test_onnx.jpg
    """