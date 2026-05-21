#include <PhaRaO.hpp>

PhaRaO::PhaRaO(ros::NodeHandle nh) : nh_(nh)
{
	nh_.getParam("is_polar_img", param_isPolarImg_);
	nh_.getParam("radar_range_bin", param_range_bin_);
	nh_.getParam("radar_angular_bin", param_ang_bin_);

	nh_.getParam("coarse_scale_factor", param_scale_);
	nh_.getParam("sub_img_size", param_sub_);
	nh_.getParam("radar_resolution", param_resol_);
	nh_.getParam("display_enabled", param_display_);
	nh_.getParam("use_log_magnitude", param_use_log_mag_);
	nh_.getParam("spectral_whitening", param_spectral_whitening_);
	nh_.getParam("whitening_sigma", param_whitening_sigma_);

	ROS_INFO("[PhaRaO] range_bin=%d, ang_bin=%d, scale=%d, sub=%d, resol=%.6f, "
	         "display=%d, log_mag=%d, spectral_whitening=%d (sigma=%.1f)",
	         param_range_bin_, param_ang_bin_, param_scale_, param_sub_, param_resol_,
	         (int)param_display_, (int)param_use_log_mag_,
	         (int)param_spectral_whitening_, param_whitening_sigma_);

	width_ 		= floor((double) param_range_bin_ / (double) param_scale_);
	height_ 	= width_;
	p_width_ 	= param_range_bin_;
	p_height_ 	= param_ang_bin_;

	ddc_.initialized = false;
	dc_ = &ddc_;
	go_ = new GraphOptimizer(nh_, dc_, param_resol_);
}

PhaRaO::~PhaRaO()
{

}

// Helper: normalize float mat to [0,255] uint8 and save
void
PhaRaO::saveDebugImg(const cv::Mat& img, const std::string& name)
{
	cv::Mat out;
	if (img.type() == CV_32FC1) {
		double mn, mx;
		cv::minMaxLoc(img, &mn, &mx);
		if (mx - mn > 1e-12)
			img.convertTo(out, CV_8UC1, 255.0/(mx-mn), -mn*255.0/(mx-mn));
		else
			out = cv::Mat::zeros(img.size(), CV_8UC1);
	} else {
		out = img;
	}
	cv::imwrite(debug_dir_ + name, out);
}

void
PhaRaO::callback(const sensor_msgs::ImageConstPtr& msg)
{
	ros::Time stamp = msg->header.stamp;
	dc_->stamp_list.push_back(stamp);
	if(dc_->initialized == false) {
		dc_->keyf_stamp_list.push_back(stamp);
	}

	// Image preprocessing for phase correlation
	cv::Mat img = cv_bridge::toCvShare(msg, "mono8")->image;
	
	// Convert to polar if input is cartesian
	if(!param_isPolarImg_)
		img = convertToPolar(img);

	img = img.t();

	boost::thread* thread_pc = new boost::thread(&PhaRaO::preprocess_coarse, this, img);
	boost::thread* thread_pcf = new boost::thread(&PhaRaO::preprocess_fine, this, img);

	thread_pc->join();
	thread_pcf->join();
	delete thread_pc;
	delete thread_pcf;

	frame_counter_++;

	go_->optimize();

	// Guard waitKey behind display flag — calling it headless can block/crash
	if (param_display_)
		waitKey(1);

}

void
PhaRaO::preprocess_coarse(cv::Mat img)
{
	cv::Mat radar_image_polar;
	cv::Mat radar_image_cart;

	bool save_debug = (frame_counter_ >= 38 && frame_counter_ <= 42);
	char prefix[64];
	if (save_debug)
		snprintf(prefix, sizeof(prefix), "f%04d_", frame_counter_ + 112);

	// Image Downsampling and Polar to Cartesian Module
	img.convertTo(radar_image_polar, CV_32FC1, 1.0/255.0);

	if (save_debug) {
		saveDebugImg(img, std::string(prefix) + "s0_polar_raw.png");
		saveDebugImg(radar_image_polar, std::string(prefix) + "s0_polar_float.png");
	}

	cv::Mat polar;
	cv::resize(radar_image_polar, polar, Size(width_, p_height_), 0, 0, CV_INTER_NN);	

	if (save_debug)
		saveDebugImg(polar, std::string(prefix) + "s1a_polar_ds.png");

	cv::Mat resize_cart;
	itf.warpPolar(polar, resize_cart, Size( width_*2,height_*2 ),
				Point2f( width_,height_ ), width_, CV_INTER_AREA | CV_WARP_INVERSE_MAP);

	if (save_debug)
		saveDebugImg(resize_cart, std::string(prefix) + "s1b_Dc_cart.png");

	////////////////////////////////////////////////////////////////////////////
	// Hanning window — reduces spectral leakage before DFT (standard for FMT)
	cv::Mat hann;
	itf.createHanningWindow(hann, resize_cart.size(), CV_32F);
	if (save_debug)
		saveDebugImg(hann, std::string(prefix) + "s1c_hanning.png");
	resize_cart = resize_cart.mul(hann);
	if (save_debug)
		saveDebugImg(resize_cart, std::string(prefix) + "s1c_Dc_hann.png");

	// FFT Module (OpenCV)
	cv::Mat padded;
	int m = getOptimalDFTSize( resize_cart.rows );
	int n = getOptimalDFTSize( resize_cart.cols );
	copyMakeBorder(resize_cart, padded, 0, m-resize_cart.rows, 0, n-resize_cart.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix

	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))

	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];

	if (param_use_log_mag_)
		log(magI, magI);  // log compression on DFT magnitude

	itf.fftShift(magI);

	if (save_debug)
		saveDebugImg(magI, std::string(prefix) + "s1d_dft_mag.png");

	////////////////////////////////////////////////////////////////////////////

	// HPF Module
	ArrayXXf filter = itf.highpassfilt(magI.size(), dc_->initialized);
	MatrixXf filter_mt = filter.matrix();
	cv::Mat filter_cv;
	eigen2cv(filter_mt, filter_cv);

	if (save_debug) {
		saveDebugImg(filter_cv, std::string(prefix) + "s1e_hpf.png");
	}

	cv::Mat filt_FFT = filter_cv.mul(magI);

	if (save_debug)
		saveDebugImg(filt_FFT, std::string(prefix) + "s1e_filt_mag.png");

	// ── Spectral Whitening ───────────────────────────────────────────────────
	// Local spectral whitening to remove systematic directional bias in the
	// DFT magnitude spectrum. The bias arises from non-uniform spectral energy
	// distribution in Oxford's CTS350 radar images (urban scene geometry,
	// antenna roll-off) which shifts the log-polar correlation peak by a
	// constant per-frame offset, accumulating into the observed heading drift.
	//
	// Method: divide each spectral component by its local neighbourhood average
	//   M_white(u,v) = M(u,v) / (G_σ * M(u,v) + ε)
	// making the spectrum locally flat in magnitude. The subsequent log-polar
	// phase correlation then responds only to the angular modulation pattern
	// (rotation-encoding structure), not to absolute spectral intensity bias.
	//
	// Reference: Tzimiropoulos et al., "Robust FFT-Based Scale-Invariant Image
	//   Registration with Image Gradients", IEEE Trans. Image Process., 2010.
	//   Equivalent to pre-whitening the matched filter in detection theory.
	if (param_spectral_whitening_) {
		cv::Mat local_avg;
		cv::GaussianBlur(filt_FFT, local_avg,
		                 cv::Size(0, 0), param_whitening_sigma_);
		local_avg += cv::Scalar(1e-6f);  // avoid division by zero
		cv::divide(filt_FFT, local_avg, filt_FFT);
		if (save_debug)
			saveDebugImg(filt_FFT, std::string(prefix) + "s1e_whitened.png");
	}
	// ─────────────────────────────────────────────────────────────────────────

	// Log-Polar Module
	cv::Mat resize_polar = log_polar(filt_FFT);	

	if (save_debug)
		saveDebugImg(resize_polar, std::string(prefix) + "s1f_Dp_logpolar.png");

	// Save preprocessed images
	dc_->window_list.push_back(resize_polar);
	dc_->window_list_cart.push_back(resize_cart);

	if(dc_->initialized == false) {
		dc_->keyf_list.push_back(resize_polar);
		dc_->keyf_list_cart.push_back(resize_cart);
	}

	////////////////////////////////////////////////////////////////////////////
}

void
PhaRaO::preprocess_fine(cv::Mat img)
{
	cv::Mat radar_image_polar;
	cv::Mat radar_image_cart;

	int length = param_sub_;

	bool save_debug = (frame_counter_ >= 38 && frame_counter_ <= 42);
	char prefix[64];
	if (save_debug)
		snprintf(prefix, sizeof(prefix), "f%04d_", frame_counter_ + 112);

	// Image Downsampling and Polar to Cartesian Module
	img(cv::Rect(0, 0, length, p_height_)).convertTo(radar_image_polar, CV_32FC1, 1.0/255.0);

	if (save_debug)
		saveDebugImg(radar_image_polar, std::string(prefix) + "s2_fine_polar.png");

	cv::Mat resize_cart;
	itf_f.warpPolar(radar_image_polar, resize_cart, Size( length*2,length*2 ),
				Point2f( length,length ), length, CV_INTER_AREA | CV_WARP_INVERSE_MAP);

	if (save_debug)
		saveDebugImg(resize_cart, std::string(prefix) + "s2_Fc_fine.png");

	// Save preprocessed images
	dc_->window_list_cart_f.push_back(resize_cart);
	
	if(dc_->initialized == false) {
		dc_->keyf_list_cart_f.push_back(resize_cart);
	}

}