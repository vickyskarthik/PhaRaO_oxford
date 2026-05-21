#include <factor/GraphOptimizer.hpp>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/nonlinear/Values.h>
#include <map>
#include <fstream>

GraphOptimizer::GraphOptimizer(ros::NodeHandle nh, DataContainer* dc, double resol)
							: nh_(nh), dc_(dc), FactorConstructor(dc)
{
	RESOL = resol;

	// Also propagate RESOL to the ImageTF member used for fine phase correlation
	itf.setResolution(resol);

	// Read coarse scale factor and propagate it
	double scale = 10.0;
	nh.getParam("coarse_scale_factor", scale);
	ratio = scale;
	itf.setScaleFactor(scale);

	// ROS
	nh_.getParam("odom_factor_cost_threshold", odom_threshold_);
	nh_.getParam("keyframe_factor_cost_threshold", keyf_threshold_);
	nh_.getParam("max_velocity_threshold", vel_threshold_);
	nh_.getParam("max_angular_velocity_threshold", angvel_threshold_);

	nh_.getParam("save_results_flag", save_flag_);
	nh_.getParam("path_filename_odom", filename_odom_);
	nh_.getParam("path_filename_optimized_odom", filename_optodom_);

	nh_.getParam("odom_noise_model_variance_x", odom_var_x_);
	nh_.getParam("odom_noise_model_variance_y", odom_var_y_);
	nh_.getParam("odom_noise_model_variance_theta", odom_var_theta_);
	nh_.getParam("keyframe_noise_model_variance", keyf_var_);
	nh_.getParam("velocity_direction_noise_model_variance", vel_dir_var_);

	pub_opt_odom_ 	= nh.advertise<nav_msgs::Odometry>("/opt_odom", 1000);
	pub_odom_ 		= nh.advertise<nav_msgs::Odometry>("/odom", 1000);

	// GTSAM
	parameters.relinearizeThreshold = 0.01;
	parameters.relinearizeSkip = 1;
	// Use QR factorization: more numerically stable than default Cholesky for
	// ill-conditioned systems (e.g., near-singular rotation-only factors)
	parameters.factorization = ISAM2Params::QR;
	isam2 = new ISAM2(parameters);

	Eigen::AngleAxisd rollAngle(0.0, Eigen::Vector3d::UnitX());   //M_PI
	Eigen::AngleAxisd pitchAngle(0.0, Eigen::Vector3d::UnitY());
	Eigen::AngleAxisd yawAngle(0.0, Eigen::Vector3d::UnitZ());
	Eigen::Quaternion<double> init_q = yawAngle * pitchAngle * rollAngle;

	Pose2 prior_pose = gtsam::Pose2(Rot2(0), gtsam::Point2(.0,.0));

	initial_values.insert(X(pose_count), prior_pose);

	prior_noise_model_ = noiseModel::Diagonal::Sigmas((Vector(3) << 1e-4, 1e-4, 1e-4).finished());
	loose_prior_noise_model_ = noiseModel::Diagonal::Sigmas((Vector(3) << 1, 1, 1).finished());
	odom_noise_model_ = noiseModel::Diagonal::Sigmas((Vector(3) << odom_var_x_, odom_var_y_, odom_var_theta_).finished());
	key_noise_model_ = noiseModel::Diagonal::Sigmas((Vector(1) << keyf_var_).finished());
	vel_dir_noise_model_ = noiseModel::Diagonal::Sigmas((Vector(1) << vel_dir_var_).finished());

	// Add prior factor to the graph.
	poseGraph->addPrior(X(pose_count), prior_pose, prior_noise_model_);
}

GraphOptimizer::~GraphOptimizer()
{

}

void
GraphOptimizer::calibrateInitVal()
{
	// Self-correlation calibration (per paper Section II.B):
	// Correlate the first frame's preprocessed images with THEMSELVES
	// to find the zero-displacement peak position.  This is deterministic
	// and noise-free — the peak of an auto-correlation is always at the
	// exact center, giving us the true zero baseline.
	//
	// Any error in init_val becomes a constant per-frame bias that
	// accumulates linearly over thousands of frames.

	if (dc_->window_list.empty() || dc_->window_list_cart.empty() ||
	    dc_->window_list_cart_f.empty()) {
		ROS_WARN("[PhaRaO] calibrateInitVal: no frames available yet.");
		return;
	}

	// Use the first frame's preprocessed images
	cv::Mat& logpolar  = dc_->window_list.back();
	cv::Mat& cart_c    = dc_->window_list_cart.back();
	cv::Mat& cart_f    = dc_->window_list_cart_f.back();

	// ── Coarse self-correlation ──
	// Log-polar: correlate with self → rotation baseline
	if (logpolar_hann_.empty() || logpolar_hann_.size() != logpolar.size()) {
		itf.createHanningWindow(logpolar_hann_, logpolar.size(), CV_32F);
	}
	cv::Mat lp_win = logpolar.mul(logpolar_hann_);
	double phaseCorr;
	cv::Point2d peakLoc_r = cv::phaseCorrelate(lp_win, lp_win, cv::noArray(), &phaseCorr);
	init_val[2] = peakLoc_r.y;  // rotation zero-baseline

	// Cartesian coarse: correlate with self → translation baseline
	cv::Point2d peakLoc_c = cv::phaseCorrelate(cart_c, cart_c, cv::noArray(), &phaseCorr);
	init_val[0] = peakLoc_c.x;  // x zero-baseline
	init_val[1] = peakLoc_c.y;  // y zero-baseline

	// ── Fine self-correlation ──
	// Fine cartesian: correlate with self → fine translation baseline
	cv::Point2d peakLoc_f = cv::phaseCorrelate(cart_f, cart_f, cv::noArray(), &phaseCorr);
	init_val_f[0] = peakLoc_f.x;  // fine x zero-baseline
	init_val_f[1] = peakLoc_f.y;  // fine y zero-baseline
	init_val_f[2] = 0.0;

	ROS_INFO("[PhaRaO] init_val calibrated via self-correlation:");
	ROS_INFO("  coarse: x=%.4f, y=%.4f, theta=%.4f",
	         init_val[0], init_val[1], init_val[2]);
	ROS_INFO("  fine:   x=%.4f, y=%.4f",
	         init_val_f[0], init_val_f[1]);
}

void
GraphOptimizer::optimize()
{
	// Guard display behind parameter (avoids crashes in headless mode)
	if (nh_.param("display_enabled", false)) {
		imshow("Coarse cart.",*(dc_->window_list_cart.end()-1));
		waitKey(1);
	}

	// initialize: calibrate init_val via self-correlation on the first frame.
	// Per the paper (Section II.B), init_val captures the zero-displacement
	// baseline — the phase correlation peak when correlating an image with
	// itself (Δx=Δy=Δθ=0).  Any error here becomes a constant per-frame bias.
	if(dc_->initialized == false){
		calibrateInitVal();
		dc_->initialized = true;
	}
	else
	{
		cout << "=================" << endl;

		bool valid = generateOdomFactor();


		if(valid){			
			generateKeyfFactor();
		}


	}

}

bool
GraphOptimizer::generateOdomFactor()
{
	// Phase correlation between the last two frames in the window (consecutive pair only).
	// The original code tried multi-frame matching (ii=1..num) which could skip
	// intermediate nodes, leaving a disconnected factor graph → GTSAM crash.
	// Now we ONLY match the consecutive pair (num-1, num) to guarantee a connected chain.
	int num = dc_->window_list.size() - 1;
	if (num < 1) return false;

	// Safety: cap window to prevent overflow of fixed arrays
	if (num >= MAX_WINDOW) {
		ROS_WARN("[PhaRaO] Window size %d >= MAX_WINDOW=%d, forcing keyframe.", num+1, MAX_WINDOW);
		// Still compute the odom factor for the latest pair, then let keyframing trigger
	}

	static double bias = .0;
	static int bias_cnt = 0;
	static int consecutive_static = 0;  // count consecutive near-zero frames

	cout << "---- Odometry Factor ----" << endl;

	// Always compute phase correlation for the consecutive pair
	int begin = num - 1;
	int end = num;
	auto odom_result = factorGeneration(begin, end);

	// Store in odom_list (safe: num-1 < MAX_WINDOW due to keyframing)
	if (num - 1 < MAX_WINDOW)
		dc_->odom_list[num-1] = odom_result;

	// Cost calculation
	double o_yx = atan2(odom_result[1], odom_result[0]);
	double o_theta;
	if (bias_cnt == 0)
		o_theta = odom_result[2];
	else
		o_theta = odom_result[2] - bias/(double)bias_cnt;
	double cost = exp(-abs(o_yx + o_theta));
	double norm = sqrt(odom_result[0]*odom_result[0] + odom_result[1]*odom_result[1]);

	cout << "[" << pose_count << " & " << pose_count+1 << "] Cost: " << cost
		<< ", o_theta: " << o_theta << ", o_yx: " << o_yx
		<< ", norm: " << norm << endl;

	// Reject near-zero motion frames with low cost, but accept every Nth
	// static frame to keep advancing through stationary sections.
	if (norm < 0.5 && cost < odom_threshold_ && consecutive_static < 5) {
		bias_cnt++;
		bias += odom_result[2];
		consecutive_static++;

		ROS_WARN("Poor Measurement (static, %d consecutive).", consecutive_static);
		dc_->window_list.erase(dc_->window_list.end()-1);
		dc_->window_list_cart.erase(dc_->window_list_cart.end()-1);
		dc_->window_list_cart_f.erase(dc_->window_list_cart_f.end()-1);
		dc_->stamp_list.erase(dc_->stamp_list.end()-1);
		return false;
	}
	consecutive_static = 0;  // reset on acceptance

	// Accept frame — compute odometry delta using nonholonomic decomposition.
	// x = forward distance (norm), y = lateral motion from heading change.
	// This acts as a nonholonomic constraint: ground vehicles move primarily
	// forward, so lateral motion is determined by heading, not raw phase corr y.
	double x = norm;
	double theta = (bias_cnt > 0) ?
		odom_result[2] - bias/(double)bias_cnt :
		odom_result[2];

	// Zero-velocity clamping: when displacement is below the phase correlation
	// noise floor (~0.1 pixel * RESOL * ratio ≈ 0.04 m), clamp to zero.
	// This prevents noisy micro-displacements during stationary periods from
	// accumulating into significant drift.
	const double ZERO_VEL_THRESHOLD = 0.1;  // metres
	if (norm < ZERO_VEL_THRESHOLD) {
		x = 0.0;
		theta = 0.0;
	}

	// Clamp theta to prevent tan() overflow
	const double MAX_THETA_RAD = 0.5;  // ~28.6°
	double theta_safe = std::max(-MAX_THETA_RAD, std::min(MAX_THETA_RAD, theta));
	double y = x * std::tan(theta_safe);

	// Guard against NaN/Inf
	if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(theta)) {
		ROS_WARN("[PhaRaO] Non-finite odom (x=%.2f y=%.2f θ=%.4f) — skipping.", x, y, theta);
		dc_->window_list.erase(dc_->window_list.end()-1);
		dc_->window_list_cart.erase(dc_->window_list_cart.end()-1);
		dc_->window_list_cart_f.erase(dc_->window_list_cart_f.end()-1);
		dc_->stamp_list.erase(dc_->stamp_list.end()-1);
		return false;
	}

	// Hard motion limit: reject physically implausible per-frame displacements
	const double MAX_TRANS = 8.0;  // metres per frame
	if (norm > MAX_TRANS) {
		ROS_WARN("[PhaRaO] Implausible translation %.2f m — rejecting.", norm);
		dc_->window_list.erase(dc_->window_list.end()-1);
		dc_->window_list_cart.erase(dc_->window_list_cart.end()-1);
		dc_->window_list_cart_f.erase(dc_->window_list_cart_f.end()-1);
		dc_->stamp_list.erase(dc_->stamp_list.end()-1);
		return false;
	}

	pose_count++;
	pose_node_nums.push_back(pose_count);
	if ((int)pose_values.size() == pose_count - 1) {
		pose_values.push_back(current_pose);
	}
	cout << "pose num : " << pose_values.size() << endl;
	base_pose = pose_values.at(pose_count - 1);

	Eigen::Vector2d odom_tr(base_pose(0), base_pose(1));
	Eigen::Rotation2D<double> odom_rot(base_pose(2));

	Eigen::Vector2d tr_delta(x, y);
	Eigen::Rotation2D<double> rot_delta(theta);

	odom_rot = odom_rot * rot_delta;
	odom_tr = odom_tr + odom_rot * tr_delta;

	current_pose << odom_tr(0), odom_tr(1), odom_rot.angle();
	Point2 prop_point = gtsam::Point2(current_pose(0), current_pose(1));
	Rot2 orien = Rot2(current_pose(2));
	Pose2 prop_pose = gtsam::Pose2(orien, prop_point);

	// Always connect consecutive nodes: X(pose_count-1) → X(pose_count)
	Pose2 odom_delta = gtsam::Pose2(x, y, theta_safe);

	// Paper Eq. 9: Cf-weighted noise model.
	// Cf scales the information (inverse covariance): high Cf → tight, low Cf → loose.
	// Base sigma ~0.05m (coarse sub-pixel accuracy). Scale by 1/sqrt(Cf) so that
	// low-confidence measurements get proportionally looser noise.
	double cf_safe = std::max(cost, 0.01);  // avoid division by zero
	double noise_scale = 1.0 / std::sqrt(cf_safe);  // Cf=1 → 1x, Cf=0.1 → 3.2x
	auto adaptive_odom_noise = noiseModel::Diagonal::Sigmas(
		(Vector(3) << odom_var_x_ * noise_scale,
		              odom_var_y_ * noise_scale,
		              odom_var_theta_ * noise_scale).finished());
	poseGraph->add(BetweenFactor<Pose2>(X(pose_count-1), X(pose_count), odom_delta, adaptive_odom_noise));
	initial_values.insert(X(pose_count), prop_pose);

	Eigen::AngleAxisd rollAngle(0.0, Eigen::Vector3d::UnitX());
	Eigen::AngleAxisd pitchAngle(0.0, Eigen::Vector3d::UnitY());
	Eigen::AngleAxisd yawAngle(odom_rot.angle(), Eigen::Vector3d::UnitZ());
	Eigen::Quaternion<double> gtsam_quat = yawAngle * pitchAngle * rollAngle;

	publishOdom(*(dc_->stamp_list.end()-1), prop_pose, gtsam_quat);
	if (save_flag_)
		saveToFile(filename_odom_, *(dc_->stamp_list.end()-1), prop_pose, gtsam_quat);

	cout << "Current pose number: " << pose_count << endl;
	cout << "Best Matching pair: " << pose_count-1 << " & " << pose_count << endl;
	cout << "x: " << odom_result[0] << ", y: " << odom_result[1]
		<< ", theta: " << theta << endl;

	return true;
}

void
GraphOptimizer::generateKeyfFactor()
{
	static int cnt = 0;
	int num = dc_->window_list.size()-1;

	// Safety: clamp index to prevent array overflow
	if (num < 1) return;
	if (num - 1 >= MAX_WINDOW) {
		ROS_WARN("[PhaRaO] Keyframe window %d exceeds MAX_WINDOW=%d, forcing trim.", num, MAX_WINDOW);
		num = MAX_WINDOW;
	}

	dc_->del_list[num-1] = factorGeneration(0,num);

	cout << "---- Keyframe Factor ----" << endl;

	/////////////////////////////////////////////////////////
	////// Cost (compared with a previous keyframe) calculation
	double d_yx = atan2(dc_->del_list[num-1][1],dc_->del_list[num-1][0]);
	double d_theta = dc_->del_list[num-1][2];
	norm_v[num-1]=sqrt(dc_->del_list[num-1][1]*dc_->del_list[num-1][1]+dc_->del_list[num-1][0]*dc_->del_list[num-1][0]);
	norm_w[num-1]=d_theta*180.0/M_PI;

	atv[num-1] = exp(-abs(d_yx + d_theta));

	// Heuristic constraints
	if(norm_v[num-1] > 1.0){
		// 1. Forward (x-axis in our sensor coordinate) motion is dominant at nonholonimc veheicle.
		if(abs(dc_->del_list[num-1][1]) > 2.0)	
			atv[num-1] = 0;
		// 2. Bounded angular motion.
		if(abs(norm_w[num-1]) > 90.0)
			atv[num-1] = 0;
	}

	/////////////////////////////////////////////////////////
	////// Sorting by costs (ascending)
    vector<int> y(num);
    size_t n(0);

    generate(std::begin(y), std::end(y), [&]{ return n++; });

    std::sort(  std::begin(y), 
                std::end(y),
                [&](int i1, int i2) { return atv[i1] < atv[i2]; } );

    int iter = 0;
    for (auto v : y) {
        //cout << "(" << v << ") " ;

        cost_idx[iter] = v;
        iter++;
    }
    //cout << endl;

	/////////////////////////////////////////////////////////	
	////// Sorting by delta_theta (ascending)
    n = 0;

    generate(std::begin(y), std::end(y), [&]{ return n++; });

    std::sort(  std::begin(y), 
                std::end(y),
                [&](int i1, int i2) { return abs(norm_w[i1]) < abs(norm_w[i2]); } );

    iter = 0;
    for (auto v : y) {
        cost_iter[iter] = v;
        iter++;
    }

	/////////////////////////////////////////////////////////	

    cout << "Cost : ";
    for (int ii = 0; ii < num; ii++)
    	cout << atv[ii] << " ";
    cout << endl;

    cout << "norm_v : ";
	for (int ii = 0; ii < num; ii++)
    	cout << norm_v[ii] << " ";
    cout << endl;

    cout << "norm_w : ";
	for (int ii = 0; ii < num; ii++)
    	cout << norm_w[ii] << " ";
    cout << endl;

    cout << "Indices sorted by cost : ";
	for (int ii = 0; ii < num; ii++)
    	cout << cost_idx[ii] << " ";
    cout << endl;

    cout << "Indices sorted by delta_theta : ";
	for (int ii = 0; ii < num; ii++)
    	cout << cost_iter[ii] << " ";
    cout << endl;

	////////////////////////////////////////////////////

	if(num > 1) {
		// Constraints to decide keyframe
		// Paper II.C.2)
		bool constraint1 = (atv[num-1] < atv[num-2]) && (atv[num-1] < keyf_threshold_*atv[cost_idx[num-1]]);
		bool constraint2 = (num >= MAX_WINDOW - 2);  // force keyframe before overflow
		bool constraint3 = (norm_v[0] > vel_threshold_);
		bool constraint4 = (norm_w[0] > angvel_threshold_);

		if( constraint1 || constraint2 || constraint3 || constraint4) {
			int i = num-2;
			int p_ind = num-2;

			for(int ii = num-1; ii >= 0; ii--){
				if(atv[cost_iter[ii]] > keyf_threshold_*atv[cost_idx[num-1]]){
					i = cost_iter[ii];
					break;
				}
			}
			if(i == -1)
				i = cost_idx[num-1];

			// p_ind : index of the new keyframe.
			p_ind = i;
			int new_key_node = pose_count-num+p_ind+1;
			cout << "$ Start keyframing (selected keyframe : " << new_key_node << ") $" << endl;

			/////////////////////////////////////////////////////////
			for(int ii = 0; ii < num; ii++) {
				cout << "- Adding PharaoRotFactor between " << key_node << " & " << pose_count-num+ii+1;
				if (p_ind == ii) {	// keyframe
					if (norm_v[0] > 0.5)
						poseGraph->add(PharaoRotFactor(X(key_node), X(new_key_node), dc_->del_list[p_ind][2], key_noise_model_));
					cout << " (keynode)";
				} else if(atv[ii] > keyf_threshold_*atv[cost_idx[num-1]]) {	// not keyframe but higher than threshold.
					if (norm_v[0] > 0.5)
						poseGraph->add(PharaoRotFactor(X(key_node), X(pose_count-num+ii+1), dc_->del_list[ii][2], vel_dir_noise_model_));
				}
				cout << endl;
			}
			

			int p_size = num;
			if(p_size > 2) {
				poseGraph->print();

				Values odom_result;
				bool opt_ok = false;
				try {
					isam2->update(*poseGraph, initial_values);
					isam2->update();
					odom_result = isam2->calculateEstimate();
					opt_ok = true;
				} catch (const gtsam::IndeterminantLinearSystemException& e) {
					ROS_WARN("[PhaRaO] GTSAM IndeterminantLinearSystem — skipping, continuing with raw odometry.");
				} catch (const gtsam::ValuesKeyAlreadyExists& e) {
					ROS_WARN("[PhaRaO] GTSAM duplicate key — skipping optimization.");
				} catch (const gtsam::ValuesKeyDoesNotExist& e) {
					ROS_WARN("[PhaRaO] GTSAM missing key — skipping optimization.");
				} catch (const std::exception& e) {
					ROS_WARN("[PhaRaO] GTSAM exception: %s — skipping optimization.", e.what());
				} catch (...) {
					ROS_WARN("[PhaRaO] Unknown GTSAM exception — skipping optimization.");
				}
				if (!opt_ok) {
					// Recovery: fully reset GTSAM and pose graph state.
					// CRITICAL: also reset pose_count to key_node so the next
					// BetweenFactor uses X(key_node) which exists in initial_values.
					// Without this, the next factor references X(pose_count-1) which
					// is NOT in initial_values after the reset → segfault.
					isam2 = new ISAM2(parameters);
					poseGraph = new NonlinearFactorGraph();
					gtsam::Values newVals;
					initial_values = newVals;

					// Re-anchor at the last known good keyframe pose
					Pose2 kf_pose(current_pose(0), current_pose(1), current_pose(2));
					poseGraph->addPrior(X(key_node), kf_pose, prior_noise_model_);
					initial_values.insert(X(key_node), kf_pose);

					// Reset pose_count so next BetweenFactor is (key_node → key_node+1)
					pose_count = key_node;

					// Clear window to just the last frame (so next frame has a valid reference)
					if (!dc_->window_list.empty()) {
						cv::Mat last_p  = dc_->window_list.back();
						cv::Mat last_c  = dc_->window_list_cart.back();
						cv::Mat last_cf = dc_->window_list_cart_f.back();
						ros::Time last_t = dc_->stamp_list.back();
						dc_->window_list.clear();
						dc_->window_list_cart.clear();
						dc_->window_list_cart_f.clear();
						dc_->stamp_list.clear();
						dc_->window_list.push_back(last_p);
						dc_->window_list_cart.push_back(last_c);
						dc_->window_list_cart_f.push_back(last_cf);
						dc_->stamp_list.push_back(last_t);
					}

					ROS_WARN("[PhaRaO] Recovery: reset to key_node=%d, pose_count=%d",
					         key_node, pose_count);
					return;
				}

				isam2 = new ISAM2(parameters);
				poseGraph = new NonlinearFactorGraph();
				gtsam::Values NewGraphValues;
				initial_values = NewGraphValues;

				/////////////////////////////////////////////////////////
				prev_pose = odom_result.at<Pose2>(X(new_key_node));

				// Update current_pose to the GTSAM-optimized value at the latest
				// node so that the next odom factor uses the corrected base pose,
				// not the un-optimized dead-reckoning estimate.
				{
					Pose2 latest_opt = odom_result.at<Pose2>(X(pose_count));
					current_pose << latest_opt.x(), latest_opt.y(), latest_opt.theta();
					// Also update pose_values for the latest poses in the window
					for (int ii = 0; ii < num; ii++) {
						int idx = pose_count - num + 1 + ii;
						if (idx >= 0 && idx < (int)pose_values.size()) {
							Pose2 popt = odom_result.at<Pose2>(X(idx));
							pose_values[idx] << popt.x(), popt.y(), popt.theta();
						}
					}
				}

				cout << "Last Pose value:\n     x:" << prev_pose.translation().x() 
					<< "     y:"<< prev_pose.translation().y() 
					<< "     theta:"<< prev_pose.rotation().theta()<<endl;
				Eigen::AngleAxisd rollAngle(0.0, Eigen::Vector3d::UnitX());   //M_PI
				Eigen::AngleAxisd pitchAngle(0.0, Eigen::Vector3d::UnitY());
				Eigen::AngleAxisd yawAngle(prev_pose.rotation().theta(), Eigen::Vector3d::UnitZ());
				Eigen::Quaternion<double> gtsam_quat = yawAngle * pitchAngle * rollAngle;

				publishOptOdom(*(dc_->stamp_list.begin() + 1 + p_ind), prev_pose, gtsam_quat);
				if (save_flag_)
					saveToFile(filename_optodom_, *(dc_->stamp_list.begin() + 1 + p_ind), prev_pose, gtsam_quat);

				key_node = new_key_node;
				window_loop += num;

				// DataContainer Rearranging
				std::vector<cv::Mat> temp_window_list;
				std::vector<cv::Mat> temp_window_list_cart;
				std::vector<cv::Mat> temp_window_list_cart_f;
				std::vector<ros::Time> temp_stamp_list;

				temp_window_list.resize((int)(dc_->window_list.size()));
				copy(dc_->window_list.begin(), dc_->window_list.end(), temp_window_list.begin());
				temp_window_list_cart.resize((int)(dc_->window_list_cart.size()));
				copy(dc_->window_list_cart.begin(), dc_->window_list_cart.end(), temp_window_list_cart.begin());
				temp_window_list_cart_f.resize((int)(dc_->window_list_cart_f.size()));
				copy(dc_->window_list_cart_f.begin(), dc_->window_list_cart_f.end(), temp_window_list_cart_f.begin());
				temp_stamp_list.resize((int)(dc_->stamp_list.size()));
				copy(dc_->stamp_list.begin(), dc_->stamp_list.end(), temp_stamp_list.begin());

				cv::Mat last_p,last_c,last_cf;
				ros::Time last_time;
				auto iter_p = temp_window_list.begin() + 1 + p_ind;
				last_p = *iter_p;
				auto iter_c = temp_window_list_cart.begin() + 1 + p_ind;
				last_c = *iter_c;
				auto iter_cf = temp_window_list_cart_f.begin() + 1 + p_ind;
				last_cf = *iter_cf;
				auto iter_time = temp_stamp_list.begin() + 1 + p_ind;
				last_time = *iter_time;

				dc_->window_list.clear();
				dc_->window_list_cart.clear();
				dc_->window_list_cart_f.clear();
				dc_->stamp_list.clear();

				// Replace old keyframe data (only keep latest keyframe to prevent OOM)
				dc_->keyf_list.clear();
				dc_->keyf_list_cart.clear();
				dc_->keyf_list_cart_f.clear();
				dc_->keyf_stamp_list.clear();
				dc_->keyf_list.push_back(last_p);
				dc_->keyf_list_cart.push_back(last_c);
				dc_->keyf_list_cart_f.push_back(last_cf);
				dc_->keyf_stamp_list.push_back(last_time);

				std::array<std::array<double, 3>, MAX_WINDOW> temp_odom_list;
				copy(dc_->odom_list.begin(), dc_->odom_list.end(), temp_odom_list.begin());
				dc_->odom_list.fill({});

				for (int ii = p_ind; ii < num; ii++)
				{
					int idx = pose_count-num+1+ii;
					Pose2 pose = odom_result.at<Pose2>(X(idx));

					if (ii == p_ind){
						poseGraph->addPrior(X(idx), pose, prior_noise_model_);
						initial_values.insert(X(idx), pose);
					}
					else
					{
						Pose2 prev_pose = odom_result.at<Pose2>(X(idx-1));
						Pose2 diff = prev_pose.inverse() * pose ;
						poseGraph->add(BetweenFactor<Pose2>(X(idx-1), X(idx), diff, odom_noise_model_));
						initial_values.insert(X(idx), pose);
					}
					
					dc_->window_list.push_back(*(iter_p + ii - p_ind));
					dc_->window_list_cart.push_back(*(iter_c + ii - p_ind));
					dc_->window_list_cart_f.push_back(*(iter_cf + ii - p_ind));
					dc_->stamp_list.push_back(*(iter_time + ii - p_ind));

				}

				cnt++;
			}
		
		}
	}

}

void
GraphOptimizer::publishOdom(ros::Time stamp, Pose2 pose, Eigen::Quaterniond quat)
{
	nav_msgs::Odometry odom;
	odom.header.stamp = stamp;
	odom.header.frame_id = "odom";
	odom.pose.pose.position.x = pose.translation().x();
	odom.pose.pose.position.y = pose.translation().y();
	odom.pose.pose.position.z = 0;
	odom.pose.pose.orientation.w = quat.w();
	odom.pose.pose.orientation.x = quat.x();
	odom.pose.pose.orientation.y = quat.y();
	odom.pose.pose.orientation.z = quat.z();
	pub_odom_.publish(odom);
}

void
GraphOptimizer::publishOptOdom(ros::Time stamp, Pose2 pose, Eigen::Quaterniond quat)
{
	nav_msgs::Odometry opt_odom;
	opt_odom.header.stamp = stamp;
	opt_odom.header.frame_id = "odom";
	opt_odom.pose.pose.position.x = pose.translation().x();
	opt_odom.pose.pose.position.y = pose.translation().y();
	opt_odom.pose.pose.position.z = 0;
	opt_odom.pose.pose.orientation.w = quat.w();
	opt_odom.pose.pose.orientation.x = quat.x();
	opt_odom.pose.pose.orientation.y = quat.y();
	opt_odom.pose.pose.orientation.z = quat.z();
	pub_opt_odom_.publish(opt_odom);
}

bool
GraphOptimizer::saveToFile(string filename, ros::Time stamp, 
							Pose2 pose, Eigen::Quaterniond quat)
{
	// Truncate on first write for this file path so multiple runs don't append
	static std::map<std::string, bool> first_write;
	auto mode = first_write[filename] ? (ios::app) : (ios::out | ios::trunc);
	first_write[filename] = true;

	ofstream writeFile;
	writeFile.open(filename, mode);
	if(writeFile.is_open()) {
		writeFile << stamp << ' '
				  << pose.translation().x() << ' '
				  << pose.translation().y() << " 0 "
				  << quat.w() << ' ' << quat.x() << ' ' << quat.y() << ' ' << quat.z() << endl;
		writeFile.close();

		return true;
	} else {
		ROS_ERROR("Cannot open the file!!!");

		return false;
	}
}