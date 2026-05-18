/**
 * oxford_radar_publisher.cpp
 *
 * Reads Oxford Radar RobotCar PNG files sequentially and publishes them
 * as sensor_msgs::Image on /radar/image_polar, mimicking what MulRan's
 * file_player does for the PhaRaO node.
 *
 * Oxford radar PNG format (same per-row layout as MulRan):
 *   Rows = 400 (azimuths)
 *   Cols = 11 (metadata) + range_bins (FFT data)
 *   Bytes 0-7: timestamp (int64, microseconds)
 *   Bytes 8-9: encoder angle (uint16, / 5600 * 2π)
 *   Byte 10:   valid flag (255 = valid)
 *   Bytes 11+: FFT power data (uint8)
 *
 * For PhaRaO we publish the RAW polar image (400 × (11 + range_bins))
 * as mono8, and PhaRaO's callback strips the 11-byte header internally
 * via img.t() and the range_bin parameter.
 *
 * WAIT — PhaRaO actually does NOT strip the header. It just does:
 *   img = img.t()    → transposed: cols=400 (angles), rows=full_width
 *   Then it takes img(Rect(0,0, length, 400)) where length = range_bin/scale
 *
 * So PhaRaO works on the TRANSPOSED image, treating rows as range and
 * cols as angle. The first 11 rows would be metadata garbage, but since
 * range_bin is set to 3360 (or 3768 for Oxford) and scale_factor=10,
 * width_ = 336 (or 376), height_ = 336 (or 376). The code takes
 * img(Rect(0,0, width_, p_height_)) which starts from row 0 — this
 * includes the 11 metadata bytes!
 *
 * For MulRan this works because their bag stores ONLY the FFT data
 * (no metadata header). Looking at PhaRaO README: they use file_player
 * which likely strips the header.
 *
 * Therefore, for Oxford, we should publish ONLY the FFT portion
 * (rows=400, cols=range_bins) to match what PhaRaO expects.
 */

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <algorithm>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "oxford_radar_publisher");
    ros::NodeHandle nh("~");

    std::string data_dir, topic;
    int start_index = 0, end_index = -1;
    double rate_hz = 4.0;
    bool strip_header = true;
    int header_bytes = 11;

    nh.param<std::string>("data_dir", data_dir,
        "/home/mcw/Desktop/MAY16PhD/2019-01-10-12-32-52-radar-oxford-10k/radar/");
    nh.param<std::string>("topic", topic, "/radar/image_polar");
    nh.param<int>("start_index", start_index, 0);
    nh.param<int>("end_index", end_index, -1);
    nh.param<double>("rate_hz", rate_hz, 4.0);
    nh.param<bool>("strip_header", strip_header, true);
    nh.param<int>("header_bytes", header_bytes, 11);

    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise(topic, 1);

    // Collect PNG files
    std::vector<std::string> files;
    DIR* dirp = opendir(data_dir.c_str());
    if (!dirp) {
        ROS_FATAL("Cannot open directory: %s", data_dir.c_str());
        return 1;
    }
    struct dirent* dp;
    while ((dp = readdir(dirp)) != nullptr) {
        std::string name(dp->d_name);
        if (name.size() > 4 && name.substr(name.size()-4) == ".png") {
            files.push_back(name);
        }
    }
    closedir(dirp);
    std::sort(files.begin(), files.end());
    ROS_INFO("Found %zu radar PNG files in %s", files.size(), data_dir.c_str());

    if (files.empty()) {
        ROS_FATAL("No PNG files found!");
        return 1;
    }

    ros::Rate loop_rate(rate_hz);
    int idx = 0;
    for (const auto& fname : files) {
        if (!ros::ok()) break;
        if (idx < start_index) { ++idx; continue; }
        if (end_index > 0 && idx >= end_index) {
            ROS_INFO("Reached end_index=%d, stopping publisher.", end_index);
            break;
        }

        std::string path = data_dir + "/" + fname;
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            ROS_WARN("Failed to read: %s", path.c_str());
            ++idx; continue;
        }

        cv::Mat pub_img;
        if (strip_header && img.cols > header_bytes) {
            // Strip 11-byte metadata header — publish only FFT data
            cv::Mat fft_only = img.colRange(header_bytes, img.cols).clone();
            // Transpose to match MulRan bag format expected by PhaRaO:
            // MulRan publishes as rows=range_bins, cols=azimuths
            // (PhaRaO then does img.t() to get rows=azimuths, cols=range)
            // Oxford PNG is rows=400(azimuths), cols=3768(range)
            // So we transpose: rows=3768(range), cols=400(azimuths)
            cv::transpose(fft_only, pub_img);
        } else {
            pub_img = img;
        }

        // Extract timestamp from first row's bytes 0-7
        int64_t ts_us = 0;
        for (int b = 0; b < 8; ++b)
            ts_us |= static_cast<int64_t>(img.at<uchar>(0, b)) << (b * 8);

        std_msgs::Header hdr;
        hdr.stamp = ros::Time(static_cast<double>(ts_us) * 1e-6);
        hdr.frame_id = "radar";
        hdr.seq = idx;

        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(hdr, "mono8", pub_img).toImageMsg();
        pub.publish(msg);

        if (idx % 100 == 0) {
            ROS_INFO("Published frame %d / %zu  (cols=%d)", idx, files.size(), pub_img.cols);
        }

        ++idx;
        loop_rate.sleep();
    }

    ROS_INFO("Oxford radar publisher finished. Published %d frames.", idx - start_index);
    // Stay alive so roslaunch doesn't kill PhaRaO while it's still processing
    ROS_INFO("Publisher idle — waiting for PhaRaO to finish processing...");
    ros::spin();  // blocks until roslaunch shuts down
    return 0;
}
