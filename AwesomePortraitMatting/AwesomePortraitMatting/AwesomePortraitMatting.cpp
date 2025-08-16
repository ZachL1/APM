// AwesomePortraitMatting.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <filesystem>
#include <cctype>

#include "portrait_matting.h"
#include "argengine.hpp"

void help_info()
{
    std::string input_file_help =
        "\t\tPath to the video or image file.\n"\
        "\t\tYou should also use --format to specify the format of the input file. The default\n"\
        "\t\tformat is video, so you can leave it out if the input is video. But must specify for\n"\
        "\t\timage: use --format image.";
    std::string install_help =
        "\t\tIf you have never run APM on this computer, use this option to install it first.\n"\
        "\t\tThe installation actually compiles the model and exports it as a cache blob to avoid\n"\
        "\t\tinference time calculation errors caused by compiling model when launching the program.";
    std::string input_dir_help =
        "\t\tPath to the directory containing video or image files.\n"\
        "\t\tThe program will process all files contained in the directory. These files must be\n"\
        "\t\tin the same format: video or image. You should also use --format to specify the format\n"\
        "\t\tof the input file.The default format is video, so you can leave it out if the input is\n"\
        "\t\tvideo. But must specify for image: use --format image.";
    std::string output_dir_help =
        "\t\tPath to the directory where the output is stored.\n"\
        "\t\tThe path should only include the directory where the results are stored. The filename\n"\
        "\t\twill be the same as the input filename, but ending with _result.";
    std::string camera_help =
        "\t\tIf this parameter is specified, --input is taken as the camera number and --output is\n"\
        "\t\ttaken as the display window name.";
    std::string mode_help =
        "\t\tMode of output. Default is alpha.\n"\
        "\t\talpah: The output is the mask (alpha).\n"\
        "\t\tmerge: The output is an RGB image of the foreground.";

    std::cout << "Awesome Portrait Matting (APM) - by 2103216" << std::endl
        << "usage: \t.\\apm.exe [options]" << std::endl
        << "e.g.: \t.\\apm.exe -i ../test_video/TEST_01.mp4" << std::endl
        << "\t.\\apm.exe -i ..\\test_image\\TEST_01.jpg\n" << std::endl
        << "optional arguments:" << std::endl
        << "--help, -h \tShow this hlep message." << std::endl
        << "--install \tInstall APM, create model cache." << std::endl
        << install_help << std::endl
        << "--input INPUT_FILE, -i INPUT_FILE" << std::endl
        << input_file_help << std::endl
        << "--input INPUT_DIR, -i INPUT_DIR" << std::endl
        << input_dir_help << std::endl
        << "--output OUTPUT_DIR, -o OUTPUT_DIR" << std::endl
        << output_dir_help << std::endl
        << "--camera, -c \tUse camera as input." << std::endl
        << camera_help << std::endl
        << "--mode [alpha, merge], -m [alpha, merge]" << std::endl
        << mode_help << std::endl;
}

void help_callback()
{
    help_info();
    exit(0);
}

void awesome_portrait_matting(PortraitMatting& apm,
    const std::filesystem::path& _input_path,
    const std::filesystem::path& _output_dir,
    const std::string& mode)
{
    // 图片扩展名
    std::unordered_set<std::string> image_format = {
        ".jpg", ".jpeg", ".jpe", ".jp2", // JPEG files
        ".png", // Portable Network Graphics
        ".bmp", ".dib", // Windows bitmatps
        ".tiff", ".tif", // TIFF files
        ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", // Portable image format
        ".hdr", ".pic" // Radiance HDR
    };
    // 完善输入输出路径
    std::filesystem::path output_dir = _output_dir;
    std::string input_path = _input_path.generic_string();
    std::string output_path = output_dir
        .append(_input_path.filename().generic_string())
        .generic_string();
    while (output_path.back() != '.') output_path.pop_back();
    output_path.pop_back();
    output_path.append("_result");
    // 分辨输入是图片还是视频
    if (image_format.find(_input_path.extension().generic_string()) != image_format.end()) {
        std::cout << "[INFO] Input is image: " << input_path << std::endl; // 输入是图片
        apm.ImageMatting(input_path, output_path.append(".jpg"), mode);
    }
    else {
        std::cout << "[INFO] Input is video: " << input_path << std::endl; // 输入是视频
        apm.VideoMatting(input_path, output_path.append(".mp4"), mode);
    }
}

int main(int argc, char* argv[])
{
    // 将前处理整合到模型中并生成新的模型
    //PortraitMatting::IntegrateModel("model/original/awesome_portrait_matting.xml",
    //    "model/awesome_portrait_matting");

    std::filesystem::path input_path, output_dir;
    bool camera = false, install = false;
    std::string mode = "alpha";

    // ========  Step 0: 准备输入参数 =========
    juzzlin::Argengine ae(argc, argv, false);
    ae.setHelpText("Awesome Portrait Matting - by 2103216");
    ae.addHelp({ "-h", "--help" }, help_callback);
    ae.addOption({ "--install" }, [&install]() {
        install = true;
        });
    ae.addOption({ "-i", "--input" }, [&input_path](std::string _input_path) {
        input_path = _input_path;
        });
    ae.addOption({ "-o", "--output" }, [&output_dir](std::string _output_dir) {
        output_dir = _output_dir;
        });
    ae.addOption({ "-c", "--camera" }, [&camera]() {
        camera = true;
        });
    ae.addOption({ "-m", "--mode" }, [&mode](std::string _mode) {
        mode = _mode;
        });
    try {
        ae.parse();
    }
    catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl << std::endl;
        help_info();
        return EXIT_FAILURE;
    }

    // 编译模型并导出缓存 cache
    // 同一台设备只需要 install 一次即可！
    if (install) {
        std::cout << "[INFO] Compiling and loading model into device..." << std::endl
            << "[INFO] If this is first time, it may take a while...";
        ov::Core core;
        core.set_property(ov::cache_dir("cl_cache"));
        std::shared_ptr<ov::Model> model = core.read_model("model/awesome_portrait_matting.xml");
        ov::CompiledModel compiled_model = core.compile_model(model, "AUTO:GPU,CPU",
            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
        std::cout << "done!" << std::endl;
        std::cout << "[INFO] Successful installation!" << std::endl;
        return 0;
    }

    // ========  Step 1: 对错误参数输入处理 =========
    // 必须指定输入
    if (!camera && input_path.empty()) {
        std::cerr << "[ERROR] Path to video or image is required: use --input.\n" << std::endl;
        help_info();
        return EXIT_FAILURE;
    }
    // 没有指定输出目录，则与输入目录相同
    if (!camera && output_dir.empty()) {
        std::cout << "[INFO] Output directory is empty and will be the same as input directory." << std::endl;
        output_dir = input_path.has_extension() ? input_path.parent_path() : input_path;
    }
    // 不应该指定输出文件名，文件名将和输入相同，但以 _result 结尾
    else if (!camera && output_dir.has_extension()) {
        std::cout << "[WARNING] Only the output directory is required, no filename should be specified." << std::endl;
        output_dir = output_dir.parent_path();
    }
    // 错误的输出模式
    if (mode != "alpha" && mode != "merge") {
        std::cerr << "[ERROR] Wrong output mode, mode must be alpha or merge." << std::endl;
        help_info();
        return EXIT_FAILURE;
    }

    // ========  Step 2: 创建 matting 类 =========
    std::string model_path("model/awesome_portrait_matting.xml");
    PortraitMatting matte(model_path);

    // ========  Step 3: 处理输入 =========
    // 指定了 -camera 选项，则从相机读取输入
    if (camera) {
        std::string camera_id = input_path.generic_string();
        std::string output_name = output_dir.empty() ? "CameraMatting" : output_dir.generic_string();
        // 输入为空，默认从 0 号相机读取输入
        if (camera_id.empty()) {
            camera_id = "0";
        }
        else if (camera_id.size() > 1 || !isdigit(camera_id.at(0))) {
            std::cerr << "[ERROR] Wrong camera id." << std::endl;
            return EXIT_FAILURE;
        }
        matte.CameraMatting(stoi(camera_id), output_name, mode);
    }
    // 输入没有扩展名，即为目录，处理目录中所有文件
    if (!camera && !input_path.has_extension()) {
        std::cout << "[INFO] Input is directory, which will process all files in the directory." << std::endl;
        // 遍历目录中的文件
        std::filesystem::directory_iterator input_it(input_path), end;
        // 递归遍历目录及其子目录
        //std::filesystem::recursive_directory_iterator input_it(input_path), end; 
        if (input_it == end) {
            std::cout << "[INFO] Directory is empty, exit." << std::endl;
        }

        for (int i = 0; input_it != end; ++input_it) {
            if (!input_it->path().has_extension()) continue; // 跳过子目录
            std::cout << "\n=====> The " << ++i << "-th file in directory: " << input_path << std::endl;
            awesome_portrait_matting(matte, input_it->path(), output_dir, mode);
            std::cout << std::endl;
        }
    }
    // 输入有扩展名，即为文件，单独处理指定文件
    else if (!camera && input_path.has_extension()) {
        awesome_portrait_matting(matte, input_path, output_dir, mode);
    }

    return 0;
}