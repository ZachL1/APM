# 视频人像抠图算法 - Windows 部署项目

> 这是我几年前完成的一个项目，因此描述细节可能不准确。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2022.x-orange.svg)](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)
[![Visual Studio](https://img.shields.io/badge/Visual%20Studio-2019-purple.svg)](https://visualstudio.microsoft.com/)

本项目基于 [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) 实现，使用 OpenVINO 将其部署到 Windows 平台上，提供高性能的实时视频人像抠图功能。

## 🌟 项目特性

- ✅ **实时人像分割**: 基于深度学习的视频抠图算法
- 🚀 **高性能**: 使用 OpenVINO 优化推理速度  
- 🎥 **虚拟摄像头**: 支持虚拟摄像头插件，可用于视频会议软件
- 💻 **Windows 原生**: 完全适配 Windows 系统
- 🔧 **易于集成**: 提供完整的 C++ SDK

## 📁 项目结构

```
├── VideoMatting-onnx/          # 模型导出工具
│   ├── export_onnx.py          # 动态 ONNX 导出
│   ├── export_onnx_static.py   # 静态 ONNX 导出  
│   ├── inference_onnx.py       # ONNX 模型推理测试
│   ├── inference_openvino.py   # OpenVINO 推理测试
│   ├── model/                  # 模型相关代码
│   └── weight/                 # 预训练权重和转换后的模型
├── AwesomePortraitMatting/     # C++ 控制台应用
│   ├── AwesomePortraitMatting.sln
│   └── AwesomePortraitMatting/
│       ├── AwesomePortraitMatting.cpp  # 主程序
│       ├── portrait_matting.cpp        # 核心算法实现
│       └── portrait_matting.h          # 头文件
└── APMvcam/                    # 虚拟摄像头插件  
    ├── APMvcam.sln
    └── Filters/
        ├── APMvcam.cpp         # 虚拟摄像头实现
        └── APMvcam.h
```

## 🛠️ 环境要求

### 系统要求
- Windows 10/11 (x64)
- Visual Studio 2019 或更高版本

### 依赖库
- **OpenVINO** 2022.1+ 
- **OpenCV** 4.5+
- **DirectShow SDK** (用于虚拟摄像头)

### Python 依赖 (模型导出)
```bash
pip install torch==1.9.0 torchvision==0.10.0 onnx opencv-python
```

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/ZachL1/APM.git
cd APM
```

### 2. 模型转换【可选】
直接下载已导出模型：https://github.com/ZachL1/APM/releases

#### 下载预训练模型
从 [RobustVideoMatting Releases](https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0) 下载 `rvm_mobilenetv3.pth` 到 `VideoMatting-onnx/weight` 目录。

#### 导出 ONNX 模型
```bash
cd VideoMatting-onnx
python export_onnx_static.py --model-variant mobilenetv3 --checkpoint weight/rvm_mobilenetv3.pth --output weight/awesome_portrait_matting.onnx
```

#### 转换为 OpenVINO IR 格式
```bash
# 配置 OpenVINO 环境
call "C:\Program Files (x86)\Intel\openvino_2022\setupvars.bat"

# 转换模型
python mo_onnx.py --input_model weight/awesome_portrait_matting.onnx --output_dir weight/ --input src,r1i,r2i,r3i,r4i --input_shape "[1,3,1080,1920],[1,16,68,120],[1,20,34,60],[1,40,17,30],[1,64,9,15]"
```

### 3. 构建 C++ 项目

#### 控制台应用
1. 打开 `AwesomePortraitMatting/AwesomePortraitMatting.sln`
2. 配置构建方式为 **Release x64**
3. 配置依赖库路径：
   - **包含目录**: OpenVINO、OpenCV 头文件路径
   - **库目录**: OpenVINO、OpenCV 库文件路径  
   - **链接器输入**: 添加必要的 .lib 文件
4. 构建项目

#### 虚拟摄像头插件
1. 首先构建 DirectShow BaseClasses:
   ```bash
   git clone https://github.com/roman380/Windows-classic-samples.git -b directshow
   cd Windows-classic-samples/Samples/Win7Samples/multimedia/directshow
   # 使用 Visual Studio 打开 directshow.sln 并构建 BaseClasses
   ```

2. 修改 `APMvcam/directshow.props` 中的 `WindowsClassicSamplesDir` 路径

3. 打开 `APMvcam/APMvcam.sln` 并构建项目

## 📖 使用说明

### 控制台应用使用 (apm.exe)

控制台应用程序是 `apm.exe`，需要在命令行窗口中使用，运行时需要为其提供命令行参数，因此直接双击 .exe 文件无法运行。

#### 查看帮助文档
```bash
# 在 Windows cmd 中进入应用程序目录，使用 -h 参数查看帮助
.\apm.exe -h
```

#### 首次使用 - 编译模型
```bash
# 第一次使用前，使用 --install 参数编译模型
.\apm.exe --install
```
> apm 使用模型缓存技术将编译好的模型导出到 cl_cache 文件夹下，以后使用时将节省编译模型的时间，程序启动更快，测试推理耗时更准确。

#### 处理文件和目录
使用 `-i` 参数指定输入路径，输入可以是单个文件或目录：

```bash
# 处理单个文件
.\apm.exe -i ..\TEST\TEST_01.mp4

# 处理目录中的所有文件（推荐使用批量处理）
.\apm.exe -i ..\TEST

# 指定输出目录（可选，输出目录必须已存在）
.\apm.exe -i ..\TEST -o ..\OUTPUT

# 输出融合结果（抠出主体并融合在黑色背景上）
.\apm.exe -i ..\TEST -m merge
```

#### 实时摄像头处理
```bash
# 从默认摄像头（0号）捕获输入，实时抠图并展示效果
.\apm.exe -c -m merge

# 指定摄像头编号
.\apm.exe -c -i 1 -m merge
```

#### 重要注意事项
1. **路径格式**: apm 同时支持正斜杠和反斜杠，使用正常路径即可，无需转义
2. **字符限制**: 不支持非 ASCII 字符路径（不能有中文），路径中有空格需用双引号包裹
3. **文件格式**: 支持图片和视频处理（OpenCV 支持的格式），自动识别文件类型
4. **批量处理**: 推荐指定目录进行批量处理，减少读取和加载模型时间
5. **输出文件**: 默认输出在输入同一目录下，文件名以 `_result` 结尾
6. **输出格式**: 默认输出主体的 mask（灰度），使用 `-m merge` 输出融合结果
7. **程序路径**: 避免将 apm 存放在中文路径下，可能导致运行失败

### 虚拟摄像头插件使用 (APMvcam.dll)

虚拟摄像头插件是一个 DLL 文件：`APMvcam.dll`。开启该插件需要使用 regsvr32 命令行工具进行注册。

#### 注册和注销插件
```bash
# 方式1: 使用命令行（需要管理员权限）
regsvr32 APMvcam.dll

# 方式2: 使用提供的脚本（推荐）
# 双击 APMregister.bat   - 注册插件
# 双击 APMunregister.bat - 注销插件
```

#### 使用虚拟摄像头
1. 注册插件后，在支持摄像头的应用程序中选择 "APM Virtual Cam"
2. 将获得从默认摄像头捕获视频流并实时抠图的结果
3. 可在 ZOOM、Teams 等视频会议软件中使用

#### 重要注意事项
1. **系统架构**: APM 虚拟摄像头只能被 64 位应用程序识别
2. **路径限制**: 插件存放路径中不能有中文或空格，否则可能导致调用失败
3. **管理员权限**: 注册插件需要管理员权限

## 🔧 配置选项

### 性能调优
- **线程数**: 根据 CPU 核心数调整 OpenVINO 推理线程
- **输入分辨率**: 支持 1080p、720p 等多种分辨率
- **设备类型**: 支持 CPU、GPU 加速

### 算法参数
- **阈值设置**: 调整分割精度和速度平衡
- **时序一致性**: 启用/禁用帧间平滑
- **背景替换**: 支持纯色、图片、视频背景

## 🐛 常见问题

### Q: 编译时找不到 OpenVINO 头文件？
A: 确保已正确安装 OpenVINO 并配置了环境变量，检查 VS 项目中的包含目录设置。

### Q: 虚拟摄像头无法注册？
A: 需要以管理员权限运行 `regsvr32` 命令，确保 DirectShow BaseClasses 已正确编译。

### Q: 推理速度慢？
A: 可以尝试：
- 使用 GPU 加速 (`-d GPU`)
- 降低输入分辨率
- 调整线程数设置

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) - 原始算法实现
- [roman380/tmhare.mvps.org-vcam](https://github.com/roman380/tmhare.mvps.org-vcam) - 虚拟摄像头基础
- Intel OpenVINO 团队 - 推理框架支持
