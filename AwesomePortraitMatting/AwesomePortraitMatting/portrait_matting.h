#pragma once

#ifndef PORTRAIT_MATTING_H
#define PORTRAIT_MATTING_H

#include <string>
#include <vector>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <openvino/pass/serialize.hpp>

/**
 * @brief 该类实现对图片、视频以及相机的人像抠图。
 * PortraitMatting 类主要提供三个对外接口：
 *  * ImageMatting：对图片进行人像抠图。
 *  * VideoMatting：对视频流进行人像抠图。
 *  * CameraMatting：从摄像头捕获视频流进行人像抠图。
 */
class PortraitMatting
{
public:
    /**
     * @brief 从指定模型构造用于人像抠图的对象。
     * @param model_path 指向 IR 模型的路径，是 .xml 文件的路径而不是 .bin 。
     *
     * @note 路径中最好不要有非 ASCII 字符。
     */
    __declspec(dllexport) explicit PortraitMatting(const std::string& model_path);

    /**
     * @brief 将前处理嵌入模型。
     * @param original_model 原始模型的路径。
     * @param integrated_model 将前处理整合后模型的输出路径。
     *
     * @note 路径中最好不要有非 ASCII 字符。
     */
    __declspec(dllexport) static void IntegrateModel(const std::string& original_model,
        const std::string& integrated_model);

    /**
     * @brief 对图片进行人像抠图。
     * @param image_path 需要抠图的图片路径。
     * @param output_path 抠图结果的输出路径。
     * @param mode 抠图模式，决定了输出的类型：
     * * alpha：输出为 mask(alpha)；
     * * merge：输出为 使用 mask 从原图中抠出的主体（叠加在黑色背景上）。
     *
     * @note 路径中最好不要有非 ASCII 字符。
     */
    __declspec(dllexport) void ImageMatting(const std::string& image_path,
        const std::string& output_path,
        const std::string& mode);

    /**
     * @brief 对视频进行人像抠图。
     * @param video_path 需要抠图的视频路径。
     * @param output_path 抠图结果的输出路径。
     * @param mode 抠图模式，决定了输出的类型：
     * * alpha：输出为 mask(alpha)；
     * * merge：输出为 使用 mask 从原图中抠出的主体（叠加在黑色背景上）。
     * @param writer_fps 输出结果写入文件的 fps，若不指定则与输入保持一致。
     *
     * @note 路径中最好不要有非 ASCII 字符。
     */
    __declspec(dllexport) void VideoMatting(const std::string& video_path,
        const std::string& output_path,
        const std::string& mode,
        double writer_fps = NULL);

    /**
     * @brief 从摄像头捕获视频流进行人像抠图，并将结果以窗口实时展示。
     * @param camera_id 摄像头 ID，指定从哪个摄像头捕获视频流。
     * @param window_name 展示抠图结果的窗口名。
     * @param mode 抠图模式，决定了输出的类型：
     * * alpha：输出为 mask(alpha)；
     * * merge：输出为 使用 mask 从原图中抠出的主体（叠加在黑色背景上）。
     */
    __declspec(dllexport) void CameraMatting(const int camera_id,
        const std::string& window_name,
        const std::string& mode);

private:
    /**
     * @brief 初始化模型的四个隐藏状态。
     *
     * @note 和训练时保持一致，初始状态以全 0 填充。
     */
    void init_hide_status();

    /**
     * @brief 设置模型的 img 输入。
     * @param img_mat 图片或者视频的一帧，喂给模型的 img 输入。
     *
     * @note 对实参的非 const 引用，进行前处理时将对实参产生更改。
     * @note 由于 OpenVINO 设置输入时不是深拷贝，因此必须确保推理时实参没有被销毁。
     */
    void set_input_img(cv::Mat& img_mat);

    /**
     * @brief 设置模型四个隐藏状态输入。在经过一次推理后，直接获取输出张量来设置输入。
     */
    void set_input_status();

    /**
     * @brief 生成抠图结果。
     * @param alp_tensor 从 OpenVINO 推理请求获取到的 alp 输出张量。
     * @param original_mat 原始输入图像，如果要叠加到背景将直接作用在原图上。
     * @param merge_mode 输出结果的类型：
     * * 0：输出为 mask(alpha)；
     * * 1：输出为 使用 mask 从原图中抠出的主体（叠加在黑色背景上）。
     *
     * @return 返回抠图结果。
     */
    cv::Mat generate_matting(ov::Tensor& alp_tensor,
        cv::Mat& original_mat,
        const bool merge_mode);

protected:
    PortraitMatting(const PortraitMatting&) = delete;
    PortraitMatting(PortraitMatting&&) = delete;
    PortraitMatting& operator=(const PortraitMatting&) = delete;
    PortraitMatting& operator=(PortraitMatting&&) = delete;

private:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::InferRequest infer_request;

    //! 全 0 填充的数组，用于初始化模型隐藏状态的 handler
    std::unordered_map<std::string, std::vector<float>> init_status_handler = {
        {"s1i", std::vector<float>(518400, 0)},
        {"s2i", std::vector<float>(163200, 0)},
        {"s3i", std::vector<float>(81600, 0)},
        {"s4i", std::vector<float>(32640, 0)}
    };

    //! 模型五个输入和五个输出名称的硬编码
    std::unordered_map<std::string, std::string> input_to_output = {
        {"img", "alp"},
        {"s1i", "s1o"},
        {"s2i", "s2o"},
        {"s3i", "s3o"},
        {"s4i", "s4o"}
    };

    //! 模型输入端口
    std::unordered_map<std::string, ov::Output<const ov::Node>> input_port;
    //! 模型输入支持的高
    int input_height = 1080;
    //! 模型输入支持的宽
    int input_width = 1920;
};

#endif // PORTRAIT_MATTING_H