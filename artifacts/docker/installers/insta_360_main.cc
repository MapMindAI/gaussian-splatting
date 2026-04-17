#include <iostream>
#include <ins_stitcher.h>

#include <iostream>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <vector>
#include <sstream>

#ifdef WIN32
#include <direct.h>
#include <Windows.h>
#endif // WIN32

using namespace std::chrono;
using namespace ins;

const std::string helpstr =
"{-help                   | default               | print this message                  }\n"
"{-model_root_dir         | ./models/             | the dir of models                   }\n"
"{-inputs                 | None                  | input files                         }\n"
"{-output                 | None                  | out path                            }\n"
"{-stitch_type            | template              | template                            }\n"
"{                                                | optflow                             }\n"
"{                                                | dynamicstitch                       }\n"
"{                                                | aistitch                            }\n"
"{-bitrate                | same as input video   | the bitrate of ouput file           }\n"
"{-enable_flowstate       | OFF                   | enable flowstate                    }\n"
"{-enable_directionlock   | OFF                   | enable directionlock                }\n"
"{-output_size            | 1920x960              | the resolution of output            }\n"
"{-enable_stitchfusion    | OFF                   | stitch_fusion                       }\n"
"{-enable_denoise         | OFF                   | enable denoise                      }\n"
"{-enable_colorplus       | OFF                   | enable colorplus                    }\n"
"{-enable_coolingshell    | OFF                   | enable Cooling Shell Detection      }\n"
"{-enable_deflicker       | OFF                   | enable deflicker                    }\n"
"{-enable_defringe        | OFF                   | enable defringe                     }\n"
"{-image_sequence_dir     | None                  | the output dir of image sequence    }\n"
"{-image_type             | jpg                   | jpg                                 }\n"
"{                                                | png                                 }\n"
"{-export_frame_index     |                       | Derived frame number sequence       }\n"
"{                                                | example: 20-50-30                   }\n"
"{-camera_accessory_type  | default 0             | refer to 'common.h'                 }\n"
"{-enable_h265_encoder    | h264                  | encode format                       }\n"
"{-disable_cuda           | true                  | disable cuda                        }\n"
"{-enable_soft_encode     | false                 | use soft encoder                    }\n"
"{-enable_soft_decode     | false                 | use soft decoder                    }\n"
"{-image_processing_accel | auto                  | image_processing_accel              }\n"
"{                                                | auto                                }\n"
"{                                                | cpu                                 }\n"
"{-exposure               | 0                     | use exposure[-100,100]              }\n"
"{-highlights             | 0                     | use highlights[-100,100]            }\n"
"{-shadows                | 0                     | use shadows[-100,100]               }\n"
"{-contrast               | 0                     | use contrast[-100,100]              }\n"
"{-brightness             | 0                     | use brightness[-100,100]            }\n"
"{-blackpoint             | 0                     | use blackpoint[-100,100]            }\n"
"{-saturation             | 0                     | use saturation[-100,100]            }\n"
"{-vibrance               | 0                     | use vibrance[-100,100]              }\n"
"{-warmth                 | 0                     | use warmth[-100,100]                }\n"
"{-tint                   | 0                     | use tint[-100,100]                  }\n"
"{-definition             | 0                     | use definition[0,100]               }\n"
"{-enable_debug_info      | OFF                   | print debug info                    }\n";

static std::string stringToUtf8(const std::string& original_str) {
#ifdef WIN32
    const char* std_origin_str = original_str.c_str();
    const int std_origin_str_len = static_cast<int>(original_str.length());

    int nwLen = MultiByteToWideChar(CP_ACP, 0, std_origin_str, -1, NULL, 0);

    wchar_t* pwBuf = new wchar_t[nwLen + 1];
    ZeroMemory(pwBuf, nwLen * 2 + 2);

    MultiByteToWideChar(CP_ACP, 0, std_origin_str, std_origin_str_len, pwBuf, nwLen);

    int nLen = WideCharToMultiByte(CP_UTF8, 0, pwBuf, -1, NULL, NULL, NULL, NULL);

    char* pBuf = new char[nLen + 1];
    ZeroMemory(pBuf, nLen + 1);

    WideCharToMultiByte(CP_UTF8, 0, pwBuf, nwLen, pBuf, nLen, NULL, NULL);

    std::string ret_str(pBuf);

    delete[] pwBuf;
    delete[] pBuf;

    pwBuf = NULL;
    pBuf = NULL;

    return ret_str;
#else
    return original_str;
#endif
}

template<typename T>
T clamp(const T& val, const T& min_val, const T& max_val) {
    return std::min(std::max(val, min_val), max_val);
}

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);

    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

#ifdef _WIN32
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif

std::string getExecutablePath() {
    std::string dir;
#ifdef WIN32
    char buffer[1024];
    GetModuleFileNameA(NULL, buffer, sizeof(buffer));
    std::string path = buffer;
    dir = path.substr(0, path.find_last_of("\\") + 1);
#endif
    return dir;
}

bool endsWithSlash(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    return path.back() == '/' || path.back() == '\\';
}

int main(int argc, char* argv[]) {
    ins::SetLogLevel(ins::InsLogLevel::ERR);

    ins::InitEnv();
     std::string model_root_dir;
#ifdef WIN32
    model_root_dir = getExecutablePath() + std::string("models\\");
#endif 
    
    std::vector<std::string> input_paths;
    std::string output_path;
    std::string image_sequence_dir;
    std::string exported_frame_number_sequence;

    STITCH_TYPE stitch_type = STITCH_TYPE::OPTFLOW;
    IMAGE_TYPE image_type = IMAGE_TYPE::JPEG;
    CameraAccessoryType accessory_type = CameraAccessoryType::kNormal;
    ImageProcessingAccel  image_processing_accel = ImageProcessingAccel::kAuto;
    int output_width = 1920;
    int output_height = 960;
    int output_bitrate = 0;

    bool enable_flowstate = false;
    bool enable_cuda = true;
    bool enable_soft_encode = false;
    bool enable_soft_decode = false;
    bool enalbe_stitchfusion = false;
    bool enable_colorplus = false;
    bool enable_directionlock = false;
    bool enable_sequence_denoise = false;
    bool enable_H265_encoder = false;
    bool enable_deflicker = false;
    bool enable_defringe = false;
    bool enable_cooling_shell_detection = false;

    int exposure = 0;
    int highlights = 0;
    int shadows = 0;
    int contrast = 0;
    int brightness = 0;
    int blackpoint = 0;
    int saturation = 0;
    int vibrance = 0;
    int warmth = 0;
    int tint = 0;
    int definition = 0;

    for (int i = 1; i < argc; i++) {
        if (std::string("-inputs") == std::string(argv[i])) {
            std::string input_path = argv[++i];
            while (input_path[0] != '-') {
                input_paths.push_back(stringToUtf8(input_path));
                input_path = argv[++i];
            }
        }
        if (std::string("-output") == std::string(argv[i])) {
            output_path = stringToUtf8(argv[++i]);
        }
        else if (std::string("-model_root_dir") == std::string(argv[i])) {
            model_root_dir = stringToUtf8(argv[++i]);
        }
        else if (std::string("-enable_coolingshell") == std::string(argv[i])) {
            enable_cooling_shell_detection = true;
        }
        else if (std::string("-stitch_type") == std::string(argv[i])) {
            std::string stitchType = argv[++i];
            if (stitchType == std::string("optflow")) {
                stitch_type = STITCH_TYPE::OPTFLOW;
            }
            else if (stitchType == std::string("dynamicstitch")) {
                stitch_type = STITCH_TYPE::DYNAMICSTITCH;
            }
            else if (stitchType == std::string("aistitch")) {
                stitch_type = STITCH_TYPE::AIFLOW;
            }
        }
        else if (std::string("-image_processing_accel") == std::string(argv[i])) {
            std::string imageProcessingAccel = argv[++i];
            if (imageProcessingAccel == std::string("auto")) {
                image_processing_accel = ImageProcessingAccel::kAuto;
            }
            else if (imageProcessingAccel == std::string("cpu")) {
                image_processing_accel = ImageProcessingAccel::kCPU;
            }
        }
        else if (std::string("-enable_flowstate") == std::string(argv[i])) {
            enable_flowstate = true;
        }
        else if (std::string("-disable_cuda") == std::string(argv[i])) {
            enable_cuda = false;
        }
        else if (std::string("-enable_stitchfusion") == std::string(argv[i])) {
            enalbe_stitchfusion = true;
        }
        else if (std::string("-enable_denoise") == std::string(argv[i])) {
            enable_sequence_denoise = true;
        }
        else if (std::string("-enable_colorplus") == std::string(argv[i])) {
            enable_colorplus = true;
        }
        else if (std::string("-enable_directionlock") == std::string(argv[i])) {
            enable_directionlock = true;
        }
        else if (std::string("-enable_h265_encoder") == std::string(argv[i])) {
            enable_H265_encoder = true;
        }
        else if (std::string("-bitrate") == std::string(argv[i])) {
            output_bitrate = atoi(argv[++i]);
        }
        else if (std::string("-output_size") == std::string(argv[i])) {
            auto res = split(std::string(argv[++i]), 'x');
            if (res.size() == 2) {
                output_width = std::atoi(res[0].c_str());
                output_height = std::atoi(res[1].c_str());
            }
        }
        else if (std::string("-image_sequence_dir") == std::string(argv[i])) {
            image_sequence_dir = std::string(argv[++i]);
        }
        else if (std::string("-image_type") == std::string(argv[i])) {
            std::string type = argv[++i];
            if (type == std::string("jpg")) {
                image_type = IMAGE_TYPE::JPEG;
            }
            else if (type == std::string("png")) {
                image_type = IMAGE_TYPE::PNG;
            }
        }
        else if (std::string("-camera_accessory_type") == std::string(argv[i])) {
            accessory_type = static_cast<CameraAccessoryType>(std::atoi(argv[++i]));
        }
        else if (std::string("-export_frame_index") == std::string(argv[i])) {
            exported_frame_number_sequence = argv[++i];
        }
        else if (std::string("-enable_deflicker") == std::string(argv[i])) {
            enable_deflicker = true;
        }
        else if (std::string("-enable_defringe") == std::string(argv[i])) {
            enable_defringe = true;
        }
        else if (std::string("-enable_soft_encode") == std::string(argv[i])) {
            enable_soft_encode = true;
        }
        else if (std::string("-enable_soft_decode") == std::string(argv[i])) {
            enable_soft_decode = true;
        }
        else if (std::string("-exposure") == std::string(argv[i])) {
            exposure = atoi(argv[++i]);
            exposure = clamp<int>(exposure, -100, 100);
        }
        else if (std::string("-highlights") == std::string(argv[i])) {
            highlights = atoi(argv[++i]);
            highlights = clamp<int>(highlights, -100, 100);
        }
        else if (std::string("-shadows") == std::string(argv[i])) {
            shadows = atoi(argv[++i]);
            shadows = clamp<int>(shadows, -100, 100);
        }
        else if (std::string("-contrast") == std::string(argv[i])) {
            contrast = atoi(argv[++i]);
            contrast = clamp<int>(contrast, -100, 100);
        }
        else if (std::string("-brightness") == std::string(argv[i])) {
            brightness = atoi(argv[++i]);
            brightness = clamp<int>(brightness, -100, 100);
        }
        else if (std::string("-blackpoint") == std::string(argv[i])) {
            blackpoint = atoi(argv[++i]);
            blackpoint = clamp<int>(blackpoint, -100, 100);
        }
        else if (std::string("-saturation") == std::string(argv[i])) {
            saturation = atoi(argv[++i]);
            saturation = clamp<int>(saturation, -100, 100);
        }
        else if (std::string("-vibrance") == std::string(argv[i])) {
            vibrance = atoi(argv[++i]);
            vibrance = clamp<int>(vibrance, -100, 100);
        }
        else if (std::string("-warmth") == std::string(argv[i])) {
            warmth = atoi(argv[++i]);
            warmth = clamp<int>(warmth, -100, 100);
        }
        else if (std::string("-tint") == std::string(argv[i])) {
            tint = atoi(argv[++i]);
            tint = clamp<int>(tint, -100, 100);
        }
        else if (std::string("-definition") == std::string(argv[i])) {
            definition = atoi(argv[++i]);
            definition = clamp<int>(definition, 0, 100);
        }
        else if (std::string("-enable_debug_info") == std::string(argv[i])) {
            ins::SetLogLevel(ins::InsLogLevel::VERBOSE);
        }
        else if (std::string("-help") == std::string(argv[i])) {
            std::cout << helpstr << std::endl;
        }
    }

    if (input_paths.empty()) {
        std::cout << "can not find input_file" << std::endl;
        std::cout << helpstr << std::endl;
        return -1;
    }

    if (output_path.empty() && image_sequence_dir.empty()) {
        std::cout << "can not find output_file" << std::endl;
        std::cout << helpstr << std::endl;
        return -1;
    }

    std::vector<uint64_t> export_frame_nums;
    if (!image_sequence_dir.empty()) {
        auto frame_index_vec = split(exported_frame_number_sequence, '-');
        for (auto& frame_index : frame_index_vec) {
            int index = atoi(frame_index.c_str());
            export_frame_nums.push_back(index);
        }
    }

    ins::SetModelFileRootDir(model_root_dir);

    int count = 1;
    while (count--) {
        std::mutex mutex;
        std::condition_variable cond;
        bool is_finished = false;
        bool has_error = false;
        int stitch_progress = 0;
        std::string suffix = input_paths[0].substr(input_paths[0].find_last_of(".") + 1);
        std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);
        if (suffix == "insp" || suffix == "jpg") {
            auto image_stitcher = std::make_shared<ImageStitcher>();
            image_stitcher->SetInputPath(input_paths);
            image_stitcher->SetStitchType(stitch_type);
            image_stitcher->SetOutputPath(output_path);
            image_stitcher->SetOutputSize(output_width, output_height);
            image_stitcher->EnableFlowState(enable_flowstate);
            image_stitcher->EnableDenoise(enable_sequence_denoise);
            image_stitcher->EnableCuda(enable_cuda);
            image_stitcher->EnableStitchFusion(enalbe_stitchfusion);
            image_stitcher->SetCameraAccessoryType(accessory_type);
            image_stitcher->EnableColorPlus(enable_colorplus);
            image_stitcher->SetImageProcessingAccelType(image_processing_accel);
            image_stitcher->EnableCoolingShellDetection(enable_cooling_shell_detection);
            image_stitcher->SetTint(tint);
            image_stitcher->SetWarmth(warmth);
            image_stitcher->SetShadows(shadows);
            image_stitcher->SetContrast(contrast);
            image_stitcher->SetExposure(exposure);
            image_stitcher->SetVibrance(vibrance);
            image_stitcher->SetHighlights(highlights);
            image_stitcher->SetDefinition(definition);
            image_stitcher->SetBrightness(brightness);
            image_stitcher->SetBlackpoint(blackpoint);
            image_stitcher->SetSaturation(saturation);
            image_stitcher->Stitch();
        }
        else if (suffix == "mp4" || suffix == "insv" || suffix == "lrv") {
            auto start_time = steady_clock::now();
            auto video_stitcher = std::make_shared<VideoStitcher>();
            video_stitcher->SetInputPath(input_paths);
            if (image_sequence_dir.empty()) {
                video_stitcher->SetOutputPath(output_path);
            }
            else {
                if (!export_frame_nums.empty()) {
                    video_stitcher->SetExportFrameSequence(export_frame_nums);
                }

                video_stitcher->SetImageSequenceInfo(image_sequence_dir, image_type);
            }
            video_stitcher->SetStitchType(stitch_type);
            video_stitcher->EnableCuda(enable_cuda);
            video_stitcher->EnableStitchFusion(enalbe_stitchfusion);
            video_stitcher->EnableColorPlus(enable_colorplus);
            video_stitcher->SetOutputSize(output_width, output_height);
            video_stitcher->SetOutputBitRate(output_bitrate);
            video_stitcher->EnableFlowState(enable_flowstate);
            video_stitcher->EnableDenoise(enable_sequence_denoise);
            video_stitcher->EnableDirectionLock(enable_directionlock);
            video_stitcher->SetCameraAccessoryType(accessory_type);
            video_stitcher->SetSoftwareCodecUsage(enable_soft_encode, enable_soft_decode);
            if (enable_H265_encoder) {
                video_stitcher->EnableH265Encoder();
            }
            video_stitcher->EnableDefringe(enable_defringe);
            video_stitcher->EnableDeflicker(enable_deflicker);
            video_stitcher->EnableCoolingShellDetection(enable_cooling_shell_detection);
            video_stitcher->SetImageProcessingAccelType(image_processing_accel);
            video_stitcher->SetTint(tint);
            video_stitcher->SetWarmth(warmth);
            video_stitcher->SetShadows(shadows);
            video_stitcher->SetContrast(contrast);
            video_stitcher->SetExposure(exposure);
            video_stitcher->SetVibrance(vibrance);
            video_stitcher->SetHighlights(highlights);
            video_stitcher->SetDefinition(definition);
            video_stitcher->SetBrightness(brightness);
            video_stitcher->SetBlackpoint(blackpoint);
            video_stitcher->SetSaturation(saturation);
            video_stitcher->SetStitchProgressCallback([&](int process, int error) {
                if (stitch_progress != process) {
                    const std::string process_desc = "process = " + std::to_string(process) + std::string("%");
                    std::cout << "\r" << process_desc << std::flush;
                    stitch_progress = process;
                }

                if (stitch_progress == 100) {
                    std::cout << std::endl;
                    std::unique_lock<std::mutex> lck(mutex);
                    cond.notify_one();
                    is_finished = true;
                }
            });

            video_stitcher->SetStitchStateCallback([&](int error, const char* err_info) {
                std::cout << "error: " << err_info << std::endl;
                has_error = true;
                cond.notify_one();
            });

            std::cout << "start stitch " << std::endl;
            video_stitcher->StartStitch();

            std::unique_lock<std::mutex> lck(mutex);
            cond.wait(lck, [&] {
                std::cout << "progress: " << video_stitcher->GetStitchProgress() << "; finished: " << is_finished << std::endl;
                return is_finished || has_error;
            });

            std::cout << "end stitch " << std::endl;

            auto end_time = steady_clock::now();
            std::cout << "cost = " << duration_cast<duration<double>>(end_time - start_time).count() << std::endl;
        }
    }
    return 0;
}