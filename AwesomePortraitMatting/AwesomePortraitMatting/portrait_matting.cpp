#include <cmath>
#include <chrono>

#include "portrait_matting.h"


PortraitMatting::PortraitMatting(const std::string& model_path)
{
    // ========  Step 1: 创建 OpenVINO Runime Core =========
    core.set_property(ov::cache_dir("cl_cache"));
    // ========  Step 2: 编译模型到设备 =========
    model = core.read_model(model_path);
    std::cout << "[INFO] Compiling and loading model into device..." << std::endl
        << "[INFO] If this is first time, it may take a while...";
    ov::CompiledModel compiled_model = core.compile_model(model, "AUTO:GPU,CPU",
        ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    std::cout << "done!" << std::endl;
    // ========  Step 3: 创建推理请求 =========
    for (const auto& name : input_to_output) {
        input_port[name.first] = compiled_model.input(name.first);
    }
    infer_request = compiled_model.create_infer_request();
}

void PortraitMatting::IntegrateModel(const std::string& original_model,
    const std::string& integrated_model)
{
    // ========  Step 1: read original model =========
    ov::Core core;
    std::shared_ptr<ov::Model> model =
        core.read_model(original_model);

    // ======== Step 2: Preprocessing ================
    ov::preprocess::PrePostProcessor ppp(model);
    // Declare section of desired application's input format
    ppp.input("img").tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    // Specify actual model layout
    ppp.input("img").model()
        .set_layout("NHWC");
    // Explicit preprocessing steps. Layout conversion will be done automatically as last step
    ppp.input("img").preprocess()
        .convert_element_type()
        .convert_color(ov::preprocess::ColorFormat::RGB)
        //.resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
        .scale(255.0f);
    model = ppp.build();

    // ======== Step 3: Save the model ================
    std::string xml(integrated_model + ".xml");
    std::string bin(integrated_model + ".bin");
    ov::pass::Serialize(xml, bin).run_on_model(model);
}

inline
void PortraitMatting::init_hide_status()
{
    for (const auto& name : input_to_output) {
        if (name.first == "img") continue;
        infer_request.set_tensor(name.first, ov::Tensor(input_port.at(name.first).get_element_type(),
            input_port.at(name.first).get_shape(), init_status_handler.at(name.first).data()));
    }
}

inline
void PortraitMatting::set_input_img(cv::Mat& mat)
{
    // ========  Step 1: 检查输入大小 =========
    //cv::cvtColor(img_mat, img_mat, cv::COLOR_BGR2RGB);
    //img_mat.convertTo(img_mat, CV_32FC3, 1.0f / 255.0f);
    int model_width = input_port.at("img").get_shape().at(2),
        model_height = input_port.at("img").get_shape().at(1);
    if (input_height != model_height || input_width != model_width) {
        cv::resize(mat, mat, cv::Size(model_width, model_height));
    }
    // ========  Step 2: 设置 img 输入 =========
    infer_request.set_tensor("img", ov::Tensor(input_port.at("img").get_element_type(),
        input_port.at("img").get_shape(), mat.data));
}

inline
void PortraitMatting::set_input_status()
{
    for (const auto& name : input_to_output) {
        if (name.first == "img") continue;
        infer_request.set_tensor(name.first, infer_request.get_tensor(name.second));
    }
}

inline
cv::Mat PortraitMatting::generate_matting(ov::Tensor& alp_tensor,
    cv::Mat& original_mat,
    const bool merge_mode)
{
    // ========  Step 1: 从输出 tensor 获取 alpha 结果 =========
    float* alp_ptr = alp_tensor.data<float>();
    cv::Mat alp_mat(input_port.at("img").get_shape().at(1),
        input_port.at("img").get_shape().at(2), CV_32FC1, alp_ptr);
    cv::resize(alp_mat, alp_mat, cv::Size(input_width, input_height));
    // ========  Step 2: [可选] 将前景通过 alpha 融合到黑色背景 =========
    if (merge_mode) {
        cv::Mat alp3_mat;
        std::vector<cv::Mat> alp_vec = { alp_mat, alp_mat, alp_mat };
        cv::merge(alp_vec, alp3_mat);
        original_mat.convertTo(original_mat, CV_32FC3);
        cv::multiply(original_mat, alp3_mat, original_mat);
        original_mat.convertTo(original_mat, CV_8UC3);
        return original_mat;
    }
    // ========  Step 3: 保存 alpha 结果 =========
    alp_mat.convertTo(alp_mat, CV_8UC1, 255);
    return alp_mat;
}



void PortraitMatting::ImageMatting(const std::string& image_path,
    const std::string& output_path,
    const std::string& mode)
{
    // ========  Step 1: 读取输入图片 =========
    cv::Mat mat = cv::imread(image_path);
    if (mat.empty()) {
        std::cerr << "[ERROR] Can not read image from: " << image_path << std::endl;
        return;
    }
    input_height = mat.rows;
    input_width = mat.cols;

    // ========  Step 2: 前处理+推理+后处理 =========
    std::cout << "[INFO] Processing image: " << image_path << std::endl;
    auto start = std::chrono::system_clock::now();
    // 初始化隐藏状态
    this->init_hide_status();
    // 前处理
    this->set_input_img(mat);
    // 推理
    infer_request.start_async();
    infer_request.wait();
    // 后处理
    ov::Tensor alp_tensor = infer_request.get_tensor("alp");
    cv::Mat result = this->generate_matting(alp_tensor, mat, mode == "merge");
    // 推理时间计算
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start; // 前处理 + 推理 + 后处理 耗时（ms）
    std::cout << "[INFO] Pre-processing + Inference + Post-processing time: "
        << elapsed.count() << std::endl;

    // ========  Step 3: 保存推理结果 =========
    try {
        if (!cv::imwrite(output_path, result)) {
            std::cerr << "[ERROR] Can not save image to: " << output_path
                << "  Check if directory exists." << std::endl;
            return;
        }
    }
    catch (const cv::Exception& ex) {
        std::cerr << "[ERROR] Exception save image to JPG format: " << ex.what() << std::endl;
        return;
    }
    std::cout << "[INFO] Successful!" << std::endl
        << "[INFO] Output: " << output_path << std::endl;
}

void PortraitMatting::VideoMatting(const std::string& video_path,
    const std::string& output_path,
    const std::string& mode,
    double writer_fps)
{
    bool merge_mode = mode == "merge"; // 输出模式是否为融合图

    // ========  Step 1: 创建一个从输入视频捕获帧的 capture =========
    cv::VideoCapture capture = cv::VideoCapture(video_path);
    if (!capture.isOpened()) {
        std::cerr << "[ERROR] Can not open video from: " << video_path << std::endl;
        return;
    }
    // ========  Step 2: 获取输入相关信息 =========
    input_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    input_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (writer_fps == NULL)
        writer_fps = capture.get(cv::CAP_PROP_FPS);
    double frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
    //int ex = static_cast<int>(capture.get(cv::CAP_PROP_FOURCC));
    int ex = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    // ========  Step 3: 创建一个保存抠图结果 alpha 的 writer =========
    cv::VideoWriter alpha_writer = cv::VideoWriter(output_path, ex, writer_fps,
        cv::Size(input_width, input_height), merge_mode);
    if (!alpha_writer.isOpened()) {
        std::cerr << "[ERROR] Can not save video to: " << output_path
            << "  Check if directory exists." << std::endl;
        return;
    }

    // ========  Step 4: matting loop 处理视频流 =========
    cv::Mat mat, result;

    // 累计 前处理 + 推理 + 后处理 耗时（ms)
    double progress = 0, diff = 100.0 / frame_count;
    std::cout << "[INFO] Processing video: " << video_path << " [  0%]";
    auto start = std::chrono::system_clock::now();
    // ========  Step 4-0: 初始化隐藏状态 =========
    this->init_hide_status();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    while (capture.read(mat)) {
        if (mat.empty()) break;
        start = std::chrono::system_clock::now();

        // ========  Step 4-1: 前处理 =========
        this->set_input_img(mat); // 前处理，设置输入 Tensor
        // ========  Step 4-2: 推理 =========
        infer_request.start_async();
        infer_request.wait();
        // ========  Step 4-3: 后处理 =========
        ov::Tensor alp_tensor = infer_request.get_tensor("alp");
        result = this->generate_matting(alp_tensor, mat, merge_mode);
        // ========  Step 4-4: 设置下一次推理的隐藏状态 =========
        this->set_input_status();

        // 累加耗时
        end = std::chrono::system_clock::now();
        elapsed += end - start;
        progress += diff;
        printf("\b\b\b\b\b\b[%3.0f%%]", progress);

        // ========  Step 4-5: 写入输出 =========
        alpha_writer.write(result);
    }
    std::cout << "\n[INFO] Pre-processing + Inference + Post-processing time: " << elapsed.count() << "ms" << std::endl;
    std::cout << "[INFO] Total frame count: " << frame_count << "   Each frame cost: " << elapsed.count() / frame_count << "ms" << std::endl;

    // ========  Step 5: Release =========
    capture.release();
    alpha_writer.release();
    std::cout << "[INFO] Successful!" << std::endl
        << "[INFO] Output: " << output_path << std::endl;
}

void PortraitMatting::CameraMatting(const int camera_id,
    const std::string& window_name,
    const std::string& mode)
{
    // ========  Step 1: 创建一个从输入视频捕获帧的 capture =========
    cv::VideoCapture capture(camera_id);
    capture.set(3, 1920);
    capture.set(4, 1080);
    if (!capture.isOpened()) {
        std::cerr << "[ERROR] Can not open video camer: " << camera_id << std::endl;
        return;
    }
    // ========  Step 2: 获取输入相关信息 =========
    input_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    input_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    double frame_count = 0;
    // ========  Step 3: 创建一个展示抠图结果 merger 的窗口 =========
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    // ========  Step 4: matting loop 处理视频流 =========
    cv::Mat mat, result;
    const bool merge_mode = mode == "merge"; // 输出模式是否为融合图

    // 累计 前处理 + 推理 + 后处理 耗时（ms)
    auto start = std::chrono::system_clock::now();
    // ========  Step 4-0: 初始化隐藏状态 =========
    this->init_hide_status();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "[INFO] Processing video from camera. Press ESC to quit!" << std::endl;

    while (capture.read(mat)) {
        if (mat.empty()) break;
        ++frame_count;
        start = std::chrono::system_clock::now();

        // ========  Step 4-1: 前处理 =========
        cv::Mat img_mat = mat.clone();
        this->set_input_img(img_mat); // 前处理，设置输入 Tensor
        // ========  Step 4-2: 推理 =========
        infer_request.start_async();
        infer_request.wait();
        // ========  Step 4-3: 后处理 =========
        ov::Tensor alp_tensor = infer_request.get_tensor("alp");
        result = this->generate_matting(alp_tensor, mat, merge_mode);
        // ========  Step 4-4: 设置下一次推理的隐藏状态 =========
        this->set_input_status();

        // 累加耗时
        end = std::chrono::system_clock::now();
        elapsed += end - start;

        // ========  Step 4-5: 写入输出 =========
        cv::imshow(window_name, result);
        if (cv::waitKey(1) == 27) {
            break;
        }
    }
    std::cout << "\n[INFO] Pre-processing + Inference + Post-processing time: " << elapsed.count() << "ms" << std::endl;
    std::cout << "[INFO] Total frame count: " << frame_count << "   Each frame cost: " << elapsed.count() / frame_count << "ms" << std::endl;

    // ========  Step 5: Release =========
    capture.release();
}