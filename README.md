# Video Portrait Matting - Windows Deployment

> This is a project I completed a few years ago, so some details may not be accurate.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2022.x-orange.svg)](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)
[![Visual Studio](https://img.shields.io/badge/Visual%20Studio-2019-purple.svg)](https://visualstudio.microsoft.com/)

[中文说明](README_zh.md) | English

This project is based on [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) and uses OpenVINO to deploy it on Windows platform, providing high-performance real-time video portrait matting.

## 🌟 Features

- ✅ **Real-time Portrait Matting**: Deep learning-based video matting algorithm
- 🚀 **High Performance**: Optimized inference speed with OpenVINO
- 🎥 **Virtual Camera**: Virtual camera plugin for video conferencing software
- 💻 **Windows Native**: Fully adapted for Windows systems
- 🔧 **Easy Integration**: Complete C++ SDK provided

## 📁 Project Structure

```
├── VideoMatting-onnx/          # Model export tools
│   ├── export_onnx.py          # Dynamic ONNX export
│   ├── export_onnx_static.py   # Static ONNX export
│   ├── inference_onnx.py       # ONNX model inference test
│   ├── inference_openvino.py   # OpenVINO inference test
│   ├── model/                  # Model related code
│   └── weight/                 # Pre-trained weights and converted models
├── AwesomePortraitMatting/     # C++ console application
│   ├── AwesomePortraitMatting.sln
│   └── AwesomePortraitMatting/
│       ├── AwesomePortraitMatting.cpp  # Main program
│       ├── portrait_matting.cpp        # Core algorithm implementation
│       └── portrait_matting.h          # Header file
└── APMvcam/                    # Virtual camera plugin
    ├── APMvcam.sln
    └── Filters/
        ├── APMvcam.cpp         # Virtual camera implementation
        └── APMvcam.h
```

## 🛠️ Requirements

### System Requirements
- Windows 10/11 (x64)
- Visual Studio 2019 or higher

### Dependencies
- **OpenVINO** 2022.1+
- **OpenCV** 4.5+
- **DirectShow SDK** (for virtual camera)

### Python Dependencies (for model export)
```bash
pip install torch==1.9.0 torchvision==0.10.0 onnx opencv-python
```

## 🚀 Quick Start

### 1. Clone Project
```bash
git clone https://github.com/ZachL1/APM.git
cd APM
```

### 2. Model Conversion [Optional]
Download pre-converted models directly: https://github.com/ZachL1/APM/releases

#### Download Pre-trained Model
Download `rvm_mobilenetv3.pth` from [RobustVideoMatting Releases](https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0) to `VideoMatting-onnx/weight` directory.

#### Export ONNX Model
```bash
cd VideoMatting-onnx
python export_onnx_static.py --model-variant mobilenetv3 --checkpoint weight/rvm_mobilenetv3.pth --output weight/awesome_portrait_matting.onnx
```

#### Convert to OpenVINO IR Format
```bash
# Setup OpenVINO environment
call "C:\Program Files (x86)\Intel\openvino_2022\setupvars.bat"

# Convert model
python mo_onnx.py --input_model weight/awesome_portrait_matting.onnx --output_dir weight/ --input src,r1i,r2i,r3i,r4i --input_shape "[1,3,1080,1920],[1,16,68,120],[1,20,34,60],[1,40,17,30],[1,64,9,15]"
```

### 3. Build C++ Projects

#### Console Application
1. Open `AwesomePortraitMatting/AwesomePortraitMatting.sln`
2. Set build configuration to **Release x64**
3. Configure dependency paths:
   - **Include Directories**: OpenVINO, OpenCV header paths
   - **Library Directories**: OpenVINO, OpenCV library paths
   - **Linker Input**: Add necessary .lib files
4. Build project

#### Virtual Camera Plugin
1. First, build DirectShow BaseClasses:
   ```bash
   git clone https://github.com/roman380/Windows-classic-samples.git -b directshow
   cd Windows-classic-samples/Samples/Win7Samples/multimedia/directshow
   # Open directshow.sln with Visual Studio and build BaseClasses
   ```

2. Modify `WindowsClassicSamplesDir` path in `APMvcam/directshow.props`

3. Open `APMvcam/APMvcam.sln` and build project

## 📖 Usage

### Console Application Usage (apm.exe)

The console application is `apm.exe`, which requires command-line usage with parameters. It cannot be run by double-clicking the .exe file.

#### View Help Documentation
```bash
# Navigate to application directory in Windows cmd, use -h parameter to view help
.\apm.exe -h
```

#### First-time Use - Compile Model
```bash
# Before first use, use --install parameter to compile model
.\apm.exe --install
```
> APM uses model caching technology to export compiled models to cl_cache folder. Future use will save model compilation time, start faster, and provide more accurate inference timing.

#### Process Files and Directories
Use `-i` parameter to specify input path, which can be a single file or directory:

```bash
# Process single file
.\apm.exe -i ..\TEST\TEST_01.mp4

# Process all files in directory (batch processing recommended)
.\apm.exe -i ..\TEST

# Specify output directory (optional, output directory must exist)
.\apm.exe -i ..\TEST -o ..\OUTPUT

# Output merged result (extract subject and merge on black background)
.\apm.exe -i ..\TEST -m merge
```

#### Real-time Camera Processing
```bash
# Capture from default camera (camera 0), real-time matting and display
.\apm.exe -c -m merge

# Specify camera number
.\apm.exe -c -i 1 -m merge
```

#### Important Notes
1. **Path Format**: APM supports both forward and backward slashes, use normal paths without escaping
2. **Character Limitations**: Non-ASCII character paths not supported (no Chinese), paths with spaces need double quotes
3. **File Formats**: Supports image and video processing (OpenCV supported formats), automatically recognizes file types
4. **Batch Processing**: Recommended to specify directory for batch processing to reduce model loading time
5. **Output Files**: Default output in same directory as input, filenames end with `_result`
6. **Output Format**: Default outputs subject mask (grayscale), use `-m merge` for merged results
7. **Program Path**: Avoid placing APM in Chinese paths, may cause execution failure

### Virtual Camera Plugin Usage (APMvcam.dll)

The virtual camera plugin is a DLL file: `APMvcam.dll`. Enabling this plugin requires registration using regsvr32 command-line tool.

#### Register and Unregister Plugin
```bash
# Method 1: Using command line (requires administrator privileges)
regsvr32 APMvcam.dll

# Method 2: Using provided scripts (recommended)
# Double-click APMregister.bat   - Register plugin
# Double-click APMunregister.bat - Unregister plugin
```

#### Using Virtual Camera
1. After registering plugin, select "APM Virtual Cam" in camera-supported applications
2. Get real-time matting results from default camera video stream
3. Can be used in ZOOM, Teams, and other video conferencing software

#### Important Notes
1. **System Architecture**: APM virtual camera can only be recognized by 64-bit applications
2. **Path Limitations**: Plugin storage path cannot contain Chinese characters or spaces, may cause invocation failure
3. **Administrator Privileges**: Plugin registration requires administrator privileges

## 🔧 Configuration Options

### Performance Tuning
- **Thread Count**: Adjust OpenVINO inference threads based on CPU cores
- **Input Resolution**: Supports 1080p, 720p, and other resolutions
- **Device Type**: Supports CPU, GPU acceleration

### Algorithm Parameters
- **Threshold Settings**: Adjust segmentation accuracy and speed balance
- **Temporal Consistency**: Enable/disable frame smoothing
- **Background Replacement**: Supports solid color, image, video backgrounds

## 🐛 FAQ

### Q: Cannot find OpenVINO headers during compilation?
A: Ensure OpenVINO is properly installed and environment variables are configured. Check include directories in VS project settings.

### Q: Cannot register virtual camera?
A: Run `regsvr32` command with administrator privileges. Ensure DirectShow BaseClasses are properly compiled.

### Q: Slow inference speed?
A: Try:
- Use GPU acceleration (`-d GPU`)
- Reduce input resolution
- Adjust thread count settings

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) - Original algorithm implementation
- [roman380/tmhare.mvps.org-vcam](https://github.com/roman380/tmhare.mvps.org-vcam) - Virtual camera foundation
- Intel OpenVINO Team - Inference framework support
