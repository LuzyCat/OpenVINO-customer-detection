// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <limits>

#include <gflags/gflags.h>
#include <utils/images_capture.h>
#include <monitors/presenter.h>

#include "detectors.hpp"
#include "face.hpp"
#include "visualizer.hpp"
#include "Session.hpp"

#include <Windows.h>
#include <direct.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>
#include <list>
#include <set>
#include <ctime>
#include <string>
#include <thread>
#include <mutex>
#include <utils/ocv_common.hpp>

namespace {
constexpr char h_msg[] = "show the help message and exit";
DEFINE_bool(h, false, h_msg);

constexpr char m_msg[] = "path to an .xml file with a trained Face Detection model";
DEFINE_string(m, "Model\\face-detection-adas-0001.xml", m_msg);

constexpr char i_msg[] = "an input to process. The input must be a single image, a folder of images, video file or camera id. Default is 0";
DEFINE_string(i, "0", i_msg);

constexpr char bb_enlarge_coef_msg[] = "coefficient to enlarge/reduce the size of the bounding box around the detected face. Default is 1.2";
DEFINE_double(bb_enlarge_coef, 1.2, bb_enlarge_coef_msg);

constexpr char d_msg[] =
    "specify a device to infer on (the list of available devices is shown below). "
    "Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. "
    "Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. "
    "Default is CPU";
DEFINE_string(d, "CPU", d_msg);

constexpr char dx_coef_msg[] = "coefficient to shift the bounding box around the detected face along the Ox axis";
DEFINE_double(dx_coef, 1, dx_coef_msg);

constexpr char dy_coef_msg[] = "coefficient to shift the bounding box around the detected face along the Oy axis";
DEFINE_double(dy_coef, 1, dy_coef_msg);

constexpr char fps_msg[] = "maximum FPS for playing video";
DEFINE_double(fps, -std::numeric_limits<double>::infinity(), fps_msg);

constexpr char lim_msg[] = "number of frames to store in output. If 0 is set, all frames are stored. Default is 1000";
DEFINE_uint32(lim, 1000, lim_msg);

constexpr char loop_msg[] = "enable reading the input in a loop";
DEFINE_bool(loop, false, loop_msg);

constexpr char mag_msg[] = "path to an .xml file with a trained Age/Gender Recognition model";
DEFINE_string(mag, "Model\\age-gender-recognition-retail-0013.xml", mag_msg);

constexpr char mam_msg[] = "path to an .xml file with a trained Antispoofing Classification model";
DEFINE_string(mam, "Model\\anti-spoof-mn3.xml", mam_msg);

constexpr char mem_msg[] = "path to an .xml file with a trained Emotions Recognition model";
DEFINE_string(mem, "Model\\emotions-recognition-retail-0003.xml", mem_msg);

constexpr char mhp_msg[] = "path to an .xml file with a trained Head Pose Estimation model";
DEFINE_string(mhp, "Model\\head-pose-estimation-adas-0001.xml", mhp_msg);

constexpr char mlm_msg[] = "path to an .xml file with a trained Facial Landmarks Estimation model";
DEFINE_string(mlm, "Model\\facial-landmarks-35-adas-0002.xml", mlm_msg);

constexpr char mreid_msg[] = "path to an .xml file with a trained Face Reidentification Retail model";
DEFINE_string(mreid, "Model\\face-reidentification-retail-0095.xml", mreid_msg);

//constexpr char mes_msg[] = "path to an .xml file with a trained Open Closed Eye model";
//DEFINE_string(mes, "Model\\face-reidentification-retail-0095.xml", mes_msg);

constexpr char o_msg[] = "name of the output file(s) to save";
DEFINE_string(o, "", o_msg);

constexpr char r_msg[] = "output inference results as raw values";
DEFINE_bool(r, false, r_msg);

constexpr char show_msg[] = "(don't) show output";
DEFINE_bool(show, true, show_msg);

constexpr char show_emotion_bar_msg[] = "(don't) show emotion bar";
DEFINE_bool(show_emotion_bar, false, show_emotion_bar_msg);

constexpr char smooth_msg[] = "(don't) smooth person attributes";
DEFINE_bool(smooth, true, smooth_msg);

constexpr char t_msg[] = "probability threshold for detections. Default is 0.5";
DEFINE_double(t, 0.5, t_msg);

constexpr char u_msg[] = "resource utilization graphs. Default is cdm. "
    "c - average CPU load, d - load distribution over cores, m - memory usage, h - hide";
DEFINE_string(u, "h", u_msg);

constexpr char pop_msg[] = "Optional. Text file of pop-up images.";
DEFINE_string(img, "pop_images.txt", pop_msg);

constexpr char ip_msg[] = "Optional. IP address 192.XXX.X.X.";
DEFINE_string(ip, "192.168.1.101", ip_msg);

constexpr char port_msg[] = "Optional. The number of port (default: 8986).";
DEFINE_uint32(port, 8986, port_msg);

constexpr char debug_msg[] = "Optional. Show debug text.";
DEFINE_bool(debug, false, debug_msg);

constexpr char showtime_msg[] = "Optional. How long time to show(s).";
DEFINE_uint32(show_time, 7, showtime_msg);

constexpr char size_msg[] = "Optional. Minimum Face size.";
DEFINE_uint32(min_size, 30000, size_msg);

constexpr char reid_thresh_msg[] = "Optional. Face Reidentification Threshold.";
DEFINE_double(reid_th, 0.7f, reid_thresh_msg);


constexpr char crop_msg[] = "Optional. Save cropped face.";
DEFINE_bool(crop, true, crop_msg);

constexpr char angle_msg[] = "probability threshold for detections. Default is 20.0";
DEFINE_double(ang, 20.0f, angle_msg);


void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {
        std::cout <<   "\t[ -h]                                         " << h_msg
                  << "\n\t[--help]                                      print help on all arguments"
                  << "\n\t  -m <MODEL FILE>                             " << m_msg
                  << "\n\t[ -i <INPUT>]                                 " << i_msg
                  << "\n\t[--bb_enlarge_coef <NUMBER>]                  " << bb_enlarge_coef_msg
                  << "\n\t[ -d <DEVICE>]                                " << d_msg
                  << "\n\t[--dx_coef <NUMBER>]                          " << dx_coef_msg
                  << "\n\t[--dy_coef <NUMBER>]                          " << dy_coef_msg
                  << "\n\t[--fps <NUMBER>]                              " << fps_msg
                  << "\n\t[--lim <NUMBER>]                              " << lim_msg
                  << "\n\t[--loop]                                      " << loop_msg
                  << "\n\t[--mag <MODEL FILE>]                          " << mag_msg
                  << "\n\t[--mam <MODEL FILE>]                          " << mam_msg
                  << "\n\t[--mem <MODEL FILE>]                          " << mem_msg
                  << "\n\t[--mhp <MODEL FILE>]                          " << mhp_msg
                  << "\n\t[--mlm <MODEL FILE>]                          " << mlm_msg
                  //<< "\n\t[--mreid <MODEL FILE>]                        " << mreid_msg
                  //<< "\n\t[--mes <MODEL FILE>]                          " << mes_msg
                  << "\n\t[ -o <OUTPUT>]                                " << o_msg
                  << "\n\t[ -r]                                         " << r_msg
                  << "\n\t[--show] ([--noshow])                         " << show_msg
                  << "\n\t[--show_emotion_bar] ([--noshow_emotion_bar]) " << show_emotion_bar_msg
                  << "\n\t[--smooth] ([--nosmooth])                     " << smooth_msg
                  << "\n\t[ -t <NUMBER>]                                " << t_msg
                  << "\n\t[ -u <DEVICE>]                                " << u_msg
                  << "\n\t[ -img <IMAGE FILE>]                          " << pop_msg
                  << "\n\t[ -ip <NUMBER>]                               " << ip_msg
                  << "\n\t[ -port <NUMBER>]                             " << port_msg
                  << "\n\t[ -debug ]                                    " << debug_msg
                  << "\n\t[ -show_time <NUMBER>]                        " << showtime_msg
                  << "\n\t[ -min_size <NUMBER>]                         " << size_msg
                  << "\n\t[ -reid_th <NUMBER>]                         " << reid_thresh_msg
                  << "\n\t[ -crop]                                      " << crop_msg
                  << "\n\t[ -ang <NUMBER>]                              " << angle_msg
                  << "\n\tKey bindings:"
                     "\n\t\tQ, q, Esc - Quit"
                     "\n\t\tP, p, 0, spacebar - Pause"
                     "\n\t\tC - average CPU load, D - load distribution over cores, M - memory usage, H - hide\n";
        showAvailableDevices();
        std::cout << ov::get_openvino_version() << std::endl;
        exit(0);
    } if (FLAGS_i.empty()) {
        throw std::invalid_argument{"-i <INPUT> can't be empty"};
    } if (FLAGS_m.empty()) {
        throw std::invalid_argument{"-m <MODEL FILE> can't be empty"};
    }
    slog::info << ov::get_openvino_version() << slog::endl;
}
} // namespace

// mouse callback
cv::Point p1(0, 0);
cv::Point p2(0, 0);
bool IsROI = false;

void onMouse(int event, int x, int y, int flags, void* param)
{
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        p1.x = x;
        p1.y = y;
        IsROI = false;
        slog::info << "Button Down" << p1 << slog::endl;
        break;
    case cv::EVENT_LBUTTONUP:
        p2.x = x;
        p2.y = y;
        IsROI = true;
        slog::info << "Button Up" << p2 << slog::endl;
        break;
    default:
        break;
    }
}

/** Network **/
//int checkCode(int* a_code, int* g_code)
//{
//    int img_num = -1;
//
//    if (g_code == 1)
//    {
//        //line 1: man<20
//        if (a_code == 0)
//        {
//            img_num = 1;
//        }
//        else if (a_code > 5)
//        {
//            img_num = 5;
//        }
//        else
//        {
//            img_num = a_code;
//        }
//    }
//    else if (g_code == 2)
//    {
//        if (a_code == 0)
//        {
//            img_num = 5 + 1;
//        }
//        else if (a_code > 5)
//        {
//            img_num = 5 + 5;
//        }
//        else
//        {
//            img_num = 5 + a_code;
//        }
//    }
//
//    return img_num;
//}
//
//void BackgroundThread()
//{
//    cv::Mat Background;
//    Background = cv::imread("black.jpg");
//    cv::namedWindow("BACKGROUND", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
//    cv::setWindowProperty("BACKGROUND", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
//    cv::imshow("BACKGROUND", Background);
//    auto keyinput = cv::waitKey(0); // Show Image 5s
//    if (keyinput == 27) cv::destroyWindow("BACKGROUND");
//}
//
//void showImage(std::vector<cv::Mat> POPIMAGES, std::mutex* mu, std::condition_variable* controller, int* a_code, int* g_code)
//{
//    cv::Mat promo;
//
//    while (1)
//    {
//        // mu.lock();
//        Sleep(100);
//        // if (a_code != -1 && g_code != -1)
//        {
//            std::unique_lock<std::mutex> lock(mu);
//            controller.wait(lock);
//            // printf("CODE %d %d\n", a_code, g_code);
//
//            int img_num = -1;
//
//            if (POPIMAGES.size() > 1)
//            {
//                img_num = checkCode(a_code, g_code);
//            }
//            else
//                break;
//
//            if (img_num == -1)
//                continue;
//            else
//            {
//                // std::cout << "First Result: " << img_num << std::endl;
//                Sleep(2000);
//            }
//
//            lock.unlock();
//
//            img_num = checkCode();
//            // std::cout << "Last Result: " << img_num << std::endl;
//
//            if (img_num == -1)
//                continue;
//
//            // promo = cv::imread(popimage_list[img_num - 1]);
//            promo = POPIMAGES[img_num - 1];
//            // std::cout << "IMAGE " << popimage_list[img_num - 1] << " OPENED!" << std::endl;
//
//            if (promo.empty())
//            {
//                std::cout << "[ERROR] IMAGE CANNOT LOADED" << std::endl;
//                return;
//            }
//
//            // Sleep(500);// 현재는 0.5초 후에 pop-up 하기로 설정해놓음
//
//            // debug text
//            if (FLAGS_debug)
//            {
//                std::string AGE = std::to_string(a_code);
//                std::string GENDER = (g_code == 1 ? "M" : "F");
//                std::string CODE = AGE + "-" + GENDER;
//                cv::Size s = promo.size();
//                cv::Point textPoint(s.width - 200, s.height - 50);
//                cv::putText(promo, CODE, textPoint, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 2, false);
//            }
//
//            cv::namedWindow("Display", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
//            cv::setWindowProperty("Display", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
//            cv::setWindowProperty("Display", cv::WND_PROP_TOPMOST, 1);
//            cv::imshow("Display", promo);
//            auto keyinput = cv::waitKey(FLAGS_show_time * 1000); // Show Image 5s
//            if (keyinput == 27) cv::destroyAllWindows();
//            cv::destroyWindow("Display");
//            // std::cout << "Close" << std::endl;
//
//            a_code = -1;
//            g_code = -1;
//        }
//        // mu.unlock();
//
//    }
//}

int main(int argc, char *argv[]) {
    std::set_terminate(catcher);
    parse(argc, argv);
    PerformanceMetrics metrics;

    // --------------------------- 1. Loading Inference Engine -----------------------------
    ov::Core core;

    FaceDetection faceDetector(FLAGS_m, FLAGS_t, FLAGS_r,
                                static_cast<float>(FLAGS_bb_enlarge_coef), static_cast<float>(FLAGS_dx_coef), static_cast<float>(FLAGS_dy_coef));
    AgeGenderDetection ageGenderDetector(FLAGS_mag, FLAGS_r);
    HeadPoseDetection headPoseDetector(FLAGS_mhp, FLAGS_r);
    //EmotionsDetection emotionsDetector(FLAGS_mem, FLAGS_r);
    //FacialLandmarksDetection facialLandmarksDetector(FLAGS_mlm, FLAGS_r);
    AntispoofingClassifier antispoofingClassifier(FLAGS_mam, FLAGS_r);
    FaceReidentificator faceReidentificator(FLAGS_mreid, FLAGS_r);
    // ---------------------------------------------------------------------------------------------------

    // --------------------------- 2. Reading IR models and loading them to plugins ----------------------
    Load(faceDetector).into(core, FLAGS_d);
    Load(ageGenderDetector).into(core, FLAGS_d);
    Load(headPoseDetector).into(core, FLAGS_d);
    //Load(emotionsDetector).into(core, FLAGS_d);
    //Load(facialLandmarksDetector).into(core, FLAGS_d);
    Load(antispoofingClassifier).into(core, FLAGS_d);
    Load(faceReidentificator).into(core, FLAGS_d);
    // ----------------------------------------------------------------------------------------------------

    /*Network*/
    //WSASession Session;
    //UDPSocket Socket;
    //using std::thread;
    //std::mutex* mu;
    //std::condition_variable* controller;
    //int g_code = -1;
    //int a_code = -1;
    //std::string IP = FLAGS_ip;
    //unsigned short PORT = FLAGS_port;

    // ;pg output
    time_t now = time(0);
    struct tm tstruct = *localtime(&now);
    char buf[80];
    strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M", &tstruct);
    std::string path = ".\\" + std::string(buf) + "\\";
    if (mkdir(path.c_str()) == 0)
        slog::info << "Make Output Directory " << std::string(buf) << " Success!" << slog::endl;
    std::string log_path = path + "log_" + std::string(buf) + ".csv";
    std::ofstream log_file(log_path.c_str(), std::ios_base::out | std::ios_base::app);
    log_file << "datetime, frame#, faceID, gender, age, age2\n";


    Timer timer;
    std::ostringstream out;
    size_t framesCounter = 0;
    double msrate = 1000.0 / FLAGS_fps;
    std::list<Face::Ptr> faces;
    size_t id = 0;
    //main ID
    size_t main_id = 0;
    size_t max_size = 0;

    std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);

    auto startTime = std::chrono::steady_clock::now();
    cv::Mat frame = cap->read();
    if (!frame.data) {
        throw std::runtime_error("Can't read an image from the input");
    }

    Presenter presenter(FLAGS_u, 60, {frame.cols / 4, 60});

    Visualizer visualizer{frame.size()};
    //if (FLAGS_show_emotion_bar && emotionsDetector.enabled()) {
    //    visualizer.enableEmotionBar(emotionsDetector.emotionsVec);
    //}

    LazyVideoWriter videoWriter{FLAGS_o, FLAGS_fps > 0.0 ? FLAGS_fps : cap->fps(), FLAGS_lim};

    //// load image list text file
    //std::ifstream imagelist(FLAGS_img);
    //// std::vector<std::string> popimage_list;
    //std::vector<cv::Mat> POPIMAGES;

    //if (imagelist.is_open())
    //{
    //    std::string s;
    //    while (std::getline(imagelist, s))
    //    {
    //        POPIMAGES.push_back(cv::imread(s));
    //    }
    //    imagelist.close();
    //}
    //// Show Images
    //std::thread backThread(BackgroundThread);
    //std::thread imgThread(showImage, POPIMAGES);
    //imgThread.detach();

    // Detecting all faces on the first frame and reading the next one
    faceDetector.submitRequest(frame);

    auto startTimeNextFrame = std::chrono::steady_clock::now();
    cv::Mat nextFrame = cap->read();
    while (frame.data) {
        timer.start("total");
        const auto startTimePrevFrame = startTime;
        cv::Mat prevFrame = std::move(frame);
        startTime = startTimeNextFrame;
        frame = std::move(nextFrame);
        framesCounter++;

        // Retrieving face detection results for the previous frame
        std::vector<FaceDetection::Result> prev_detection_results = faceDetector.fetchResults();

        // No valid frame to infer if previous frame is the last
        if (frame.data) {
            if (frame.size() != prevFrame.size()) {
                throw std::runtime_error("Images of different size are not supported");
            }
            faceDetector.submitRequest(frame);
        }

        if (!prev_detection_results.empty())
        {
            //reset main face && get main face
            main_id = 0;
            max_size = 0;
            for (size_t i = 0; i < prev_detection_results.size(); i++)
            {
                auto& result = prev_detection_results[i];
                int new_size = result.location.width * result.location.height;
                if (max_size < new_size)
                {
                    max_size = new_size;
                    main_id = i;
                }
            }

            //check size
            FaceDetection::Result main_face_detection;
            memcpy(&main_face_detection, &prev_detection_results[main_id], sizeof(struct FaceDetection::Result));
            int face_size = main_face_detection.location.width * main_face_detection.location.height;
            int face_cx = main_face_detection.location.x + main_face_detection.location.width / 2;
            int face_cy = main_face_detection.location.y + main_face_detection.location.height / 2;
            prev_detection_results.clear();
            if (face_size > FLAGS_min_size && face_cx > 200 && face_cx < 800)
            {
                //if (IsROI)
                //{
                //    //slog::info << face_size << slog::endl;
                //    if (face_cx > p1.x && face_cx < p2.x && face_cy > p2.y && face_cy < p1.y)
                //    {
                //        prev_detection_results.push_back(main_face_detection);
                //    }
                //}

                prev_detection_results.push_back(main_face_detection);
            }
        }

        // Filling inputs of face analytics networks
        for (auto &&face : prev_detection_results) {
            cv::Rect clippedRect = face.location & cv::Rect({0, 0}, prevFrame.size());
            const cv::Mat& crop = prevFrame(clippedRect);
            ageGenderDetector.enqueue(crop);
            headPoseDetector.enqueue(crop);
            //emotionsDetector.enqueue(crop);
            //facialLandmarksDetector.enqueue(crop);
            antispoofingClassifier.enqueue(crop);
            faceReidentificator.enqueue(crop);
        }

        // Running Age/Gender Recognition, Head Pose Estimation, Emotions Recognition, Facial Landmarks Estimation and Antispoofing Classifier networks simultaneously
        ageGenderDetector.submitRequest();
        headPoseDetector.submitRequest();
        //emotionsDetector.submitRequest();
        //facialLandmarksDetector.submitRequest();
        antispoofingClassifier.submitRequest();
        faceReidentificator.submitRequest();

        // Read the next frame while waiting for inference results
        startTimeNextFrame = std::chrono::steady_clock::now();
        nextFrame = cap->read();

        //  Postprocessing
        std::list<Face::Ptr> prev_faces;

        if (FLAGS_smooth) {
            prev_faces.insert(prev_faces.begin(), faces.begin(), faces.end());
        }

        faces.clear();

        // For every detected face
        for (size_t i = 0; i < prev_detection_results.size(); i++) {
            auto& result = prev_detection_results[i];
            cv::Rect rect = result.location & cv::Rect({0, 0}, prevFrame.size());

            Face::Ptr face;
            if (FLAGS_smooth) {
                face = matchFace(rect, prev_faces);
                float intensity_mean = calcMean(prevFrame(rect));

                if ((face == nullptr) ||
                    (((std::abs(intensity_mean - face->_intensity_mean) / face->_intensity_mean) > 0.07f) &&
                    ((calcDistance(face->getReidFeatures(), faceReidentificator[i], 256) > FLAGS_reid_th)))
                    ) {
                    face = std::make_shared<Face>(id++, rect);
                } else {
                    prev_faces.remove(face);
                }

                face->_intensity_mean = intensity_mean;
                face->_location = rect;
            } else {
                face = std::make_shared<Face>(id++, rect);
            }

            face->ageGenderEnable(ageGenderDetector.enabled());
            if (face->isAgeGenderEnabled()) {
                AgeGenderDetection::Result ageGenderResult = ageGenderDetector[i];
                face->updateGender(ageGenderResult.maleProb);
                face->updateAge(ageGenderResult.age);
            }

            //face->emotionsEnable(emotionsDetector.enabled());
            //if (face->isEmotionsEnabled()) {
            //    face->updateEmotions(emotionsDetector[i]);
            //}

            face->headPoseEnable(headPoseDetector.enabled());
            if (face->isHeadPoseEnabled()) {
                face->updateHeadPose(headPoseDetector[i]);
            }

            //face->landmarksEnable(facialLandmarksDetector.enabled());
            //if (face->isLandmarksEnabled()) {
            //    face->updateLandmarks(facialLandmarksDetector[i]);
            //}

            face->antispoofingEnable(antispoofingClassifier.enabled());
            if (face->isAntispoofingEnabled()) {
                face->updateRealFaceConfidence(antispoofingClassifier[i]);
            }

            face->facereidEnable(faceReidentificator.enabled());
            if (face->isfaceReIDEnabled()) {
                face->updateFaceReidentification(faceReidentificator[i]);
            }

            faces.push_back(face);
        }

        // save_raw
        if (FLAGS_crop)
        {
            for (auto f : faces)
            {
                double yaw = f->getHeadPose().angle_y;
                double pitch = f->getHeadPose().angle_p;
                double roll = f->getHeadPose().angle_r;

                if (f->isReal() && yaw < FLAGS_ang && pitch < FLAGS_ang && roll < FLAGS_ang &&
                    yaw > -(FLAGS_ang) && pitch > -(FLAGS_ang) && roll > -(FLAGS_ang))
                {
                    // image capture per id
                    cv::Rect rect = f->_location;
                    char s[5];
                    std::sprintf(s, "%05d", f->getId());
                    std::string filename = "raw_" + std::string(s) + ".jpg";
                    std::ifstream filetest(path + filename);
                    if (!filetest)
                    {
                        cv::imwrite(path + filename, prevFrame(rect));
                    }
                }
            }
        }
        

        // drawing faces
        visualizer.draw(prevFrame, faces);

        presenter.drawGraphs(prevFrame);
        //metrics.update(startTimePrevFrame, prevFrame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);

        timer.finish("total");

        videoWriter.write(prevFrame);

        for (auto f : faces)
        {
            double yaw = f->getHeadPose().angle_y;
            double pitch = f->getHeadPose().angle_p;
            double roll = f->getHeadPose().angle_r;

            if (f->isReal() && yaw < FLAGS_ang && pitch < FLAGS_ang && roll < FLAGS_ang &&
                yaw > -(FLAGS_ang) && pitch > -(FLAGS_ang) && roll > -(FLAGS_ang))
            {
                // image capture per id
                std::string filename = std::to_string(f->getId()) + ".jpg";
                std::ifstream filetest(path + filename);
                if (!filetest)
                {
                    cv::imwrite(path + filename, prevFrame/*(rect)*/);
                }

                log_file << std::string(buf) << ", ";
                log_file << framesCounter;
                log_file << ", " << f->getId() << ", ";
                log_file << (f->isMale() ? "Male" : "Female");
                log_file << ", " << f->getAge() + 5 << ", " << int((f->getAge() + 5) / 10) << "\n";
            }
        }

        ////network
        //time_t cur_time;
        //time(&cur_time);
        //struct tm time_now = *localtime(&cur_time);
        //char buf[80];
        //strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", &time_now);
        //std::string msg = "";
        //if (faces.empty())
        //{
        //    Socket.SendTo(IP, PORT, msg.c_str(), msg.size());
        //    a_code = -1;
        //    g_code = -1;
        //}
        //else
        //{
        //    for (auto f : faces)
        //    {
        //        log_file << std::string(buf) << ", ";
        //        log_file << framesCounter;
        //        log_file << ", " << f->getId() << ", ";
        //        log_file << (f->isMale() ? "Male" : "Female");
        //        log_file << ", " << f->getAge() + 5 << ", " << int((f->getAge() + 5) / 10) << "\n";
        //        std::string msg = "";
        //        std::string ID = std::to_string(f->getId());
        //        int age_group = static_cast<int>(((f->getAge() + 5) / 10));
        //        std::string AGE = std::to_string(age_group);
        //        std::string GENDER = (f->isMale() ? "1" : "2");
        //        msg = ID + "-" + GENDER + "-" + AGE;
        //        a_code = age_group;
        //        g_code = std::stoi(GENDER);
        //        if (g_code != -1 && a_code != -1) controller.notify_one();
        //        Socket.SendTo(IP, PORT, msg.c_str(), msg.size());
        //    }
        //}

        int delay = std::max(1, static_cast<int>(msrate - timer["total"].getLastCallDuration()));
        if (FLAGS_show) {

            //cv::Mat detection_line = cv::imread("face.png", cv::IMREAD_UNCHANGED);
            //cv::resize(detection_line, detection_line, cv::Size(600, 600)); // size 고쳐야함*
            //cv::cvtColor(prev_frame, prev_frame, cv::COLOR_BGR2BGRA);

            //int nr = detection_line.rows;
            //int nc = detection_line.cols;
            //int pr = (prev_frame.rows - nr);
            //int pc = (prev_frame.cols - nc) / 2;
            //cv::Mat roi = prev_frame(cv::Rect(pc, pr, nc, nr)); // 위치 고쳐야함*

            //for (int i = 0; i < nr; i++)
            //{
            //    cv::Vec4b* linedata = detection_line.ptr<cv::Vec4b>(i);
            //    cv::Vec4b* roiData = roi.ptr<cv::Vec4b>(i);

            //    for (int j = 0; j < nc; j++) {
            //        if (linedata[j][3] > 0) {
            //            double bgAlpha = (255 - linedata[j][3]) / 255.0;
            //            double fgAlpha = (linedata[j][3]) / 255.0;

            //            if (fgAlpha > 0.f)
            //            {
            //                roiData[j][0] = roiData[j][0] * bgAlpha + linedata[j][0] * fgAlpha;
            //                roiData[j][1] = roiData[j][1] * bgAlpha + linedata[j][1] * fgAlpha;
            //                roiData[j][2] = roiData[j][2] * bgAlpha + linedata[j][2] * fgAlpha;
            //            }
            //        }
            //    }
            //}

            //cv::Mat show_frame;
            //cv::resize(prev_frame, show_frame, cv::Size(prev_frame.cols / 2, prev_frame.rows / 2));
            //cv::imshow(argv[0], show_frame);

            cv::imshow(argv[0], prevFrame);
            cv::setMouseCallback(argv[0], onMouse);
            int key = cv::waitKey(delay);
            if ('P' == key || 'p' == key || '0' == key || ' ' == key) {
                key = cv::waitKey(0);
            }
            if (27 == key || 'Q' == key || 'q' == key) {
                break;
            }
            presenter.handleKey(key);
        }
    }

    slog::info << "Metrics report:" << slog::endl;
    metrics.logTotal();
    slog::info << presenter.reportMeans() << slog::endl;

    return 0;
}
