/**
 * main.cpp — Offline direct-call loop with GT visualization & per-frame ATE
 *
 * Pipeline (PW-NDT style — no pub/sub overhead):
 *   sort PNGs → build sensor_msgs::Image → pr.callback() → compare with GT
 *
 * Features:
 *   - Zero frame drops (sequential, no ROS transport queue)
 *   - GT path (red /pharao/gt_path) + estimated path (green /pharao/est_path)
 *     published for RViz — both normalized to start at (0,0,0)
 *   - Per-frame ATE (Absolute Trajectory Error) printed every 50 frames
 *   - odom.txt truncated on first write (no multi-run append contamination)
 *   - Final mean/max ATE summary printed at end
 */

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_datatypes.h>
#include <opencv2/opencv.hpp>
#include <boost/make_shared.hpp>
#include <Eigen/Dense>

#include <dirent.h>
#include <algorithm>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>

#include <PhaRaO.hpp>

using namespace std;

// ── GT loading (KITTI format: 12 floats = 3×4 row-major) ──────────────────
static vector<Eigen::Matrix4f> loadGTPoses(const string& fname)
{
	vector<Eigen::Matrix4f> poses;
	FILE* fp = fopen(fname.c_str(), "r");
	if (!fp) { ROS_WARN("[PhaRaO] Cannot open GT: %s", fname.c_str()); return poses; }
	while (!feof(fp)) {
		Eigen::Matrix4f P = Eigen::Matrix4f::Identity();
		if (fscanf(fp, "%f %f %f %f %f %f %f %f %f %f %f %f",
		           &P(0,0),&P(0,1),&P(0,2),&P(0,3),
		           &P(1,0),&P(1,1),&P(1,2),&P(1,3),
		           &P(2,0),&P(2,1),&P(2,2),&P(2,3)) == 12)
			poses.push_back(P);
	}
	fclose(fp);
	ROS_INFO("[PhaRaO] Loaded %zu GT poses.", poses.size());
	return poses;
}

// ── Matrix4f → PoseStamped ────────────────────────────────────────────────
static geometry_msgs::PoseStamped mat4ToPose(const Eigen::Matrix4f& P,
                                              const ros::Time& t)
{
	tf::Matrix3x3 rot(P(0,0),P(0,1),P(0,2), P(1,0),P(1,1),P(1,2), P(2,0),P(2,1),P(2,2));
	tf::Quaternion q; rot.getRotation(q);
	geometry_msgs::PoseStamped ps;
	ps.header.stamp = t; ps.header.frame_id = "odom";
	ps.pose.position.x = P(0,3); ps.pose.position.y = P(1,3); ps.pose.position.z = 0;
	ps.pose.orientation.w = q.w(); ps.pose.orientation.x = q.x();
	ps.pose.orientation.y = q.y(); ps.pose.orientation.z = q.z();
	return ps;
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "pharao_oxford_offline");
	ros::NodeHandle nh("~");

	// ── Parameters ─────────────────────────────────────────────────────────
	std::string data_dir =
	    "/home/mcw/Desktop/MAY16PhD/2019-01-10-12-32-52-radar-oxford-10k/radar/";
	std::string gt_file  =
	    "/home/mcw/Desktop/MAY16PhD/PW-ndt/catkin_ws/src/poses/"
	    "2019-01-10-12-32-52-radar-oxford-10k_gt.txt";
	int start_index = 0, end_index = -1;
	nh.param<std::string>("data_dir",    data_dir, data_dir);
	nh.param<std::string>("gt_file",     gt_file,  gt_file);
	nh.param<int>("start_index", start_index, 0);
	nh.param<int>("end_index",   end_index,  -1);

	// ── Publishers ─────────────────────────────────────────────────────────
	ros::Publisher gt_path_pub  = nh.advertise<nav_msgs::Path>("gt_path",  10);
	ros::Publisher est_path_pub = nh.advertise<nav_msgs::Path>("est_path", 10);
	nav_msgs::Path gt_msg, est_msg;
	gt_msg.header.frame_id  = "odom";
	est_msg.header.frame_id = "odom";

	// ── Load GT ─────────────────────────────────────────────────────────────
	vector<Eigen::Matrix4f> gt = loadGTPoses(gt_file);
	// Reference frame: GT pose at start_index (normalise both to (0,0,0))
	Eigen::Matrix4f gt_ref = Eigen::Matrix4f::Identity();
	if (!gt.empty()) {
		int ref_idx = std::min(start_index, (int)gt.size()-1);
		gt_ref = gt[ref_idx];
	}

	// ── PhaRaO algorithm object ─────────────────────────────────────────────
	PhaRaO pr(nh);

	// ── Enumerate and sort PNG files ────────────────────────────────────────
	std::vector<std::string> files;
	DIR* dirp = opendir(data_dir.c_str());
	if (!dirp) { ROS_FATAL("Cannot open: %s", data_dir.c_str()); return 1; }
	struct dirent* dp;
	while ((dp = readdir(dirp)) != nullptr) {
		std::string n(dp->d_name);
		if (n.size() > 4 && n.substr(n.size()-4) == ".png") files.push_back(n);
	}
	closedir(dirp);
	std::sort(files.begin(), files.end());
	ROS_INFO("[PhaRaO] %zu PNGs | frames [%d, %s) | GT %zu",
	         files.size(), start_index,
	         end_index<0?"end":std::to_string(end_index).c_str(), gt.size());

	// ── ATE statistics ──────────────────────────────────────────────────────
	double sum_te=0, sum_re=0, max_te=0;
	int    ate_n=0;
	bool   first_frame = true;

	// ── Offline loop ────────────────────────────────────────────────────────
	int idx=0, processed=0;
	for (const auto& fname : files) {
		if (!ros::ok()) break;
		if (idx < start_index) { ++idx; continue; }
		if (end_index>0 && idx>=end_index) {
			ROS_INFO("[PhaRaO] Reached end_index=%d.", end_index); break;
		}

		// Read raw PNG
		cv::Mat raw = cv::imread(data_dir+"/"+fname, cv::IMREAD_GRAYSCALE);
		if (raw.empty()) { ++idx; continue; }

		// Timestamp from bytes 0-7
		int64_t ts_us = 0;
		for (int b=0; b<8; ++b)
			ts_us |= static_cast<int64_t>(raw.at<uchar>(0,b)) << (b*8);
		ros::Time stamp(static_cast<double>(ts_us)*1e-6);

		// Strip 11-byte header + transpose for PhaRaO
		if (raw.cols <= 11) { ++idx; continue; }
		cv::Mat trans;
		cv::transpose(raw.colRange(11, raw.cols).clone(), trans);

		// Build sensor_msgs::Image in memory
		auto msg = boost::make_shared<sensor_msgs::Image>();
		msg->header.stamp=stamp; msg->header.frame_id="radar";
		msg->header.seq=idx; msg->encoding="mono8";
		msg->height=trans.rows; msg->width=trans.cols; msg->step=trans.cols;
		msg->is_bigendian=0;
		msg->data.assign(trans.data, trans.data+trans.total());

		// ── Direct algorithm call ───────────────────────────────────────────
		pr.callback(msg);
		Eigen::Vector3d est = pr.getLatestPose();  // (x, y, theta)

		// ── Per-frame ATE vs GT ─────────────────────────────────────────────
		if (!gt.empty() && idx>0 && idx<(int)gt.size() && !first_frame) {
			Eigen::Matrix4f gt_rel = gt_ref.inverse() * gt[idx];
			double gx  = gt_rel(0,3), gy = gt_rel(1,3);
			double gth = std::atan2((double)gt_rel(1,0), (double)gt_rel(0,0));

			double te = std::sqrt((est(0)-gx)*(est(0)-gx)+(est(1)-gy)*(est(1)-gy));
			double re = std::abs(est(2)-gth)*180.0/M_PI;
			if (re > 180.0) re = std::abs(re-360.0);

			sum_te += te; sum_re += re;
			if (te > max_te) max_te = te;
			++ate_n;

			// Print every 50 accepted frames
			if (processed % 50 == 0)
				ROS_INFO("[ATE idx=%d] trans=%.3fm  rot=%.2fdeg  "
				         "est(%.1f,%.1f,%.3f) gt(%.1f,%.1f,%.3f)",
				         idx, te, re,
				         est(0),est(1),est(2), gx,gy,gth);

			// GT path (normalized to start)
			gt_msg.header.stamp = stamp;
			gt_msg.poses.push_back(mat4ToPose(gt_rel, stamp));
			gt_path_pub.publish(gt_msg);
		}
		first_frame = false;

		// Estimated path
		{
			tf::Quaternion qe; qe.setRPY(0,0,est(2));
			geometry_msgs::PoseStamped ps;
			ps.header.stamp=stamp; ps.header.frame_id="odom";
			ps.pose.position.x=est(0); ps.pose.position.y=est(1); ps.pose.position.z=0;
			ps.pose.orientation.w=qe.w(); ps.pose.orientation.x=qe.x();
			ps.pose.orientation.y=qe.y(); ps.pose.orientation.z=qe.z();
			est_msg.header.stamp=stamp;
			est_msg.poses.push_back(ps);
			est_path_pub.publish(est_msg);
		}

		ros::spinOnce();
		++idx; ++processed;
		if (processed % 100 == 0)
			ROS_INFO("[PhaRaO] Processed %d frames (idx=%d)", processed, idx);
	}

	// ── Summary ─────────────────────────────────────────────────────────────
	ROS_INFO("══════════════════════════════════════════");
	ROS_INFO("[PhaRaO] Done — %d frames processed.", processed);
	if (ate_n > 0) {
		ROS_INFO("[ATE] Mean trans=%.3f m  Max trans=%.3f m  Mean rot=%.3f deg",
		         sum_te/ate_n, max_te, sum_re/ate_n);
	}
	ROS_INFO("══════════════════════════════════════════");
	return 0;
}
