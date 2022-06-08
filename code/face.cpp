// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <map>
#include <utility>
#include <list>
#include <vector>

#include "face.hpp"

Face::Face(size_t id, cv::Rect& location):
    _location(location), _intensity_mean(0.f), _id(id), _age(-1),
    _maleScore(0), _femaleScore(0), _headPose({0.f, 0.f, 0.f}), _realFaceConfidence(0),
    _isAgeGenderEnabled(false), _isEmotionsEnabled(false),
    _isHeadPoseEnabled(false), _isLandmarksEnabled(false), _isAntispoofingEnabled(false) {
}

void Face::updateAge(float value) {
    _age = (_age == -1) ? value : 0.95f * _age + 0.05f * value;
}

void Face::updateGender(float value) {
    if (value < 0)
        return;

    if (value > 0.5) {
        _maleScore += value - 0.5f;
    } else {
        _femaleScore += 0.5f - value;
    }
}

void Face::updateEmotions(std::map<std::string, float> values) {
    for (auto& kv : values) {
        if (_emotions.find(kv.first) == _emotions.end()) {
            _emotions[kv.first] = kv.second;
        } else {
            _emotions[kv.first] = 0.9f * _emotions[kv.first] + 0.1f * kv.second;
        }
    }
}

void Face::updateHeadPose(HeadPoseDetection::Results values) {
    _headPose = values;
}

void Face::updateLandmarks(std::vector<float> values) {
    _landmarks = std::move(values);
}

void Face::updateRealFaceConfidence(float value) {
    _realFaceConfidence = value;
}

void Face::updateFaceReidentification(std::vector<float> values) {
    _facereidfeatures = std::move(values);
}

int Face::getAge() {
    return static_cast<int>(std::floor(_age + 0.5f));
}

bool Face::isMale() {
    return _maleScore > _femaleScore;
}

bool Face::isReal() {
    return _realFaceConfidence > 50.;
}

std::map<std::string, float> Face::getEmotions() {
    return _emotions;
}

std::pair<std::string, float> Face::getMainEmotion() {
    auto x = std::max_element(_emotions.begin(), _emotions.end(),
        [](const std::pair<std::string, float>& p1, const std::pair<std::string, float>& p2) {
            return p1.second < p2.second; });

    return std::make_pair(x->first, x->second);
}

HeadPoseDetection::Results Face::getHeadPose() {
    return _headPose;
}

const std::vector<float>& Face::getLandmarks() {
    return _landmarks;
}

std::vector<float>& Face::getReidFeatures() {
    return _facereidfeatures;
}


size_t Face::getId() {
    return _id;
}

void Face::ageGenderEnable(bool value) {
    _isAgeGenderEnabled = value;
}
void Face::emotionsEnable(bool value) {
    _isEmotionsEnabled = value;
}
void Face::headPoseEnable(bool value) {
    _isHeadPoseEnabled = value;
}
void Face::landmarksEnable(bool value) {
    _isLandmarksEnabled = value;
}
void Face::antispoofingEnable(bool value) {
    _isAntispoofingEnabled = value;
}

void Face::facereidEnable(bool value) {
    _isfaceReIDEnabled = value;
}

bool Face::isAgeGenderEnabled() {
    return _isAgeGenderEnabled;
}
bool Face::isEmotionsEnabled() {
    return _isEmotionsEnabled;
}
bool Face::isHeadPoseEnabled() {
    return _isHeadPoseEnabled;
}
bool Face::isLandmarksEnabled() {
    return _isLandmarksEnabled;
}
bool Face::isAntispoofingEnabled() {
    return _isAntispoofingEnabled;
}
bool Face::isfaceReIDEnabled() {
    return _isfaceReIDEnabled;
}

float calcIoU(cv::Rect& src, cv::Rect& dst) {
    cv::Rect i = src & dst;
    cv::Rect u = src | dst;

    return static_cast<float>(i.area()) / static_cast<float>(u.area());
}

float calcMean(const cv::Mat& src) {
    cv::Mat tmp;
    cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
    cv::Scalar mean = cv::mean(tmp);

    return static_cast<float>(mean[0]);
}

Face::Ptr matchFace(cv::Rect rect, std::list<Face::Ptr>& faces) {
    Face::Ptr face(nullptr);
    float maxIoU = 0.4f;
    for (auto&& f : faces) {
        float iou = calcIoU(rect, f->_location);
        if (iou > maxIoU) {
            //slog::info << "MATCH" << slog::endl;
            face = f;
            maxIoU = iou;
        }
    }

    return face;
}

float calcDistance(std::vector<float> A, std::vector<float> B, unsigned int length) {
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (unsigned int i = 0u; i < length; ++i) {
        dot += A[i] * B[i];
        denom_a += A[i] * A[i];
        denom_b += B[i] * B[i];
    }
    return dot / (sqrt(denom_a) * sqrt(denom_b));
}

