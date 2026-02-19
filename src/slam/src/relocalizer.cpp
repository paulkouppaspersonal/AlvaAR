#include "relocalizer.hpp"
#include "multi_view_geometry.hpp"

Relocalizer::Relocalizer(std::shared_ptr<State> state,
                         std::shared_ptr<Frame> currFrame,
                         std::shared_ptr<MapManager> mapManager)
    : state_(state), currFrame_(currFrame), mapManager_(mapManager)
{
    // Configure iBoW-LCD for relocalization (relaxed vs loop closure defaults)
    ibow_lcd::LCDetectorParams params;
    params.p = 5;                     // Allow matching recent keyframes (not skip 100)
    params.min_score = 0.2;           // Lower threshold for viewpoint changes
    params.min_inliers = 10;          // PnP does the real geometric validation
    params.nframes_after_lc = 1;      // Query every frame while lost
    params.min_consecutive_loops = 1; // Don't wait for confirmation
    params.purge_descriptors = false; // Keep all descriptors for maximum recall

    lcd_ = std::make_unique<ibow_lcd::LCDetector>(params);
    orb_ = cv::ORB::create(500, 1.2f, 8);
}

void Relocalizer::addKeyframe(int keyframeId,
                               const std::vector<cv::KeyPoint> &kps,
                               const cv::Mat &descs)
{
    if (kps.empty() || descs.empty())
    {
        return;
    }

    ibow_lcd::LCDetectorResult result;
    lcd_->process(nextIbowId_, kps, descs, &result);
    ibowToKeyframeId_.push_back(keyframeId);
    nextIbowId_++;
}

bool Relocalizer::attempt(const cv::Mat &image)
{
    if (image.empty() || nextIbowId_ == 0)
    {
        return false;
    }

    // 1. Detect and describe ORB features in the current (lost) image
    std::vector<cv::KeyPoint> queryKps;
    cv::Mat queryDescs;
    orb_->detectAndCompute(image, cv::noArray(), queryKps, queryDescs);

    if ((int)queryKps.size() < MIN_PNP_POINTS)
    {
        return false;
    }

    // 2. Query iBoW-LCD for the best matching keyframe (search only, no add)
    ibow_lcd::LCDetectorResult result;
    lcd_->query(queryKps, queryDescs, &result);

    if (result.status != ibow_lcd::LC_DETECTED)
    {
        return false;
    }

    // 3. Map the iBoW train_id back to actual keyframe ID
    if (result.train_id >= ibowToKeyframeId_.size())
    {
        return false;
    }

    int matchedKfId = ibowToKeyframeId_[result.train_id];

    auto keyframe = mapManager_->getKeyframe(matchedKfId);

    if (!keyframe)
    {
        return false;
    }

    // 4. Collect the keyframe's 3D keypoints and their ORB descriptors
    auto kfKeypoints = keyframe->getKeypoints3d();

    if ((int)kfKeypoints.size() < MIN_PNP_POINTS)
    {
        return false;
    }

    cv::Mat kfDescs;
    std::vector<int> kfKpIds;

    for (const auto &kp : kfKeypoints)
    {
        if (!kp.desc_.empty())
        {
            kfDescs.push_back(kp.desc_);
            kfKpIds.push_back(kp.keypointId_);
        }
    }

    if (kfDescs.rows < MIN_PNP_POINTS)
    {
        return false;
    }

    // 5. Brute-force match query descriptors against keyframe descriptors
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(queryDescs, kfDescs, knnMatches, 2);

    // Ratio test
    std::vector<cv::DMatch> goodMatches;

    for (const auto &m : knnMatches)
    {
        if (m.size() >= 2 && m[0].distance < NNDR_THRESHOLD * m[1].distance)
        {
            goodMatches.push_back(m[0]);
        }
    }

    if ((int)goodMatches.size() < MIN_PNP_POINTS)
    {
        return false;
    }

    // 6. Build 2D-3D correspondences (query bearing vectors <-> keyframe map points)
    auto camCalib = currFrame_->cameraCalibration_;
    float fx = camCalib->fx_;
    float fy = camCalib->fy_;
    float cx = camCalib->cx_;
    float cy = camCalib->cy_;

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> observations;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> worldPoints;

    for (const auto &m : goodMatches)
    {
        int kfKpId = kfKpIds[m.trainIdx];

        auto mapPoint = mapManager_->getMapPoint(kfKpId);

        if (!mapPoint || !mapPoint->is3d_)
        {
            continue;
        }

        Eigen::Vector3d wpt = mapPoint->getPoint();

        // Convert query keypoint pixel position to bearing vector
        cv::Point2f rawPx = queryKps[m.queryIdx].pt;
        cv::Point2f undistPx = camCalib->undistortImagePoint(rawPx);

        Eigen::Vector3d bv(
            (undistPx.x - cx) / fx,
            (undistPx.y - cy) / fy,
            1.0
        );
        bv.normalize();

        observations.push_back(bv);
        worldPoints.push_back(wpt);
    }

    if ((int)observations.size() < MIN_PNP_POINTS)
    {
        return false;
    }

    // 7. P3P RANSAC for pose estimation
    Sophus::SE3d Twc;
    std::vector<int> outliers;

    bool success = MultiViewGeometry::p3pRansac(
        observations, worldPoints,
        200,    // maxIterations
        3.0f,   // errorThreshold (pixels)
        true,   // optimize
        true,   // doRandom
        fx, fy,
        Twc, outliers
    );

    if (!success)
    {
        return false;
    }

    int inliers = (int)observations.size() - (int)outliers.size();

    if (inliers < MIN_PNP_POINTS)
    {
        return false;
    }

    // 8. Set the recovered pose on the current frame
    currFrame_->setTwc(Twc);

    return true;
}

void Relocalizer::reset()
{
    // Recreate iBoW-LCD detector (no clear/reset method available)
    ibow_lcd::LCDetectorParams params;
    params.p = 5;
    params.min_score = 0.2;
    params.min_inliers = 10;
    params.nframes_after_lc = 1;
    params.min_consecutive_loops = 1;
    params.purge_descriptors = false;

    lcd_ = std::make_unique<ibow_lcd::LCDetector>(params);
    ibowToKeyframeId_.clear();
    nextIbowId_ = 0;
}
