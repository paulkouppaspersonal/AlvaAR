#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <sophus/se3.hpp>

#include "ibow_lcd/lcdetector.h"
#include "state.hpp"
#include "frame.hpp"
#include "map_manager.hpp"

class Relocalizer
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Relocalizer(std::shared_ptr<State> state,
                std::shared_ptr<Frame> currFrame,
                std::shared_ptr<MapManager> mapManager);

    // Feed a new keyframe's descriptors into the visual vocabulary
    void addKeyframe(int keyframeId,
                     const std::vector<cv::KeyPoint> &kps,
                     const cv::Mat &descs);

    // Attempt relocalization using the current image.
    // Returns true if pose was recovered and set on currFrame_.
    bool attempt(const cv::Mat &image);

    void reset();

private:
    std::shared_ptr<State> state_;
    std::shared_ptr<Frame> currFrame_;
    std::shared_ptr<MapManager> mapManager_;

    std::unique_ptr<ibow_lcd::LCDetector> lcd_;
    cv::Ptr<cv::ORB> orb_;

    // Maps iBoW sequential image ID -> actual keyframe ID
    std::vector<int> ibowToKeyframeId_;
    int nextIbowId_ = 0;

    static constexpr float NNDR_THRESHOLD = 0.75f;
    static constexpr int MIN_PNP_POINTS = 5;
};
