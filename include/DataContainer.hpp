#pragma once

#include <vector>
#include <array>

// Maximum sliding window size.  The keyframe factor logic triggers
// when window_list grows beyond this, so the arrays never overflow.
#define MAX_WINDOW 8

struct DataContainer
{
    bool initialized;

    std::vector<cv::Mat> window_list;
    std::vector<cv::Mat> window_list_cart;
    std::vector<cv::Mat> window_list_cart_f;

    std::vector<cv::Mat> keyf_list;
    std::vector<cv::Mat> keyf_list_cart;
    std::vector<cv::Mat> keyf_list_cart_f;

    std::vector<ros::Time> stamp_list;
    std::vector<ros::Time> keyf_stamp_list;

    std::array<std::array<double, 3>, MAX_WINDOW> del_list;
    std::array<std::array<double, 3>, MAX_WINDOW> odom_list;
};