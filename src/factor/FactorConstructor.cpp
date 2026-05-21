#include <factor/FactorConstructor.hpp>

FactorConstructor::FactorConstructor(DataContainer* dc)
				: dc_(dc)
{

}

FactorConstructor::~FactorConstructor()
{

}

std::array<double, 3>
FactorConstructor::factorGeneration(int src1, int src2)
{
    // Coarse Phase Correlation Module
    auto begin_iter = dc_->window_list.begin();
    auto begin_iter_cart = dc_->window_list_cart.begin();

    std::array<double, 3> state = {0,0,0};
    std::array<double, 3> cd_state = phaseCorr2D(*(begin_iter+src1), *(begin_iter+src2),
                                    *(begin_iter_cart+src1), *(begin_iter_cart+src2), true, state);

    // Fine Phase Correlation Module
    begin_iter_cart = dc_->window_list_cart_f.begin();

    std::array<double, 3> fd_state = phaseCorr2D(*(begin_iter+src1), *(begin_iter+src2),
                                    *(begin_iter_cart+src1), *(begin_iter_cart+src2), false, cd_state);

	std::array<double, 3> out_state;
    std::copy(fd_state.begin(), fd_state.end(), out_state.begin());

	return out_state;
}


std::array<double, 3>
FactorConstructor::phaseCorr2D(cv::Mat r_src1, cv::Mat r_src2, cv::Mat src1, cv::Mat src2,
					bool flag, std::array<double, 3> state)
{
	// flag == true : coarse, flag == false : fine
	double x,y,theta;
	double rotation;
	double phaseCorr;
	double factor;

	cv::Point2d peakLoc_r;

	if(flag == true) {
		// Rotation estimation: phase correlate log-polar images Dp.
		// Apply Hanning window to reduce spectral leakage in the log-polar
		// phase correlation. This stabilizes rotation estimation.
		if (logpolar_hann_.empty() || logpolar_hann_.size() != r_src1.size()) {
			itf.createHanningWindow(logpolar_hann_, r_src1.size(), CV_32F);
		}
		cv::Mat r1_win = r_src1.mul(logpolar_hann_);
		cv::Mat r2_win = r_src2.mul(logpolar_hann_);

		peakLoc_r = cv::phaseCorrelate(r1_win, r2_win, cv::noArray(), &phaseCorr);
		rotation = (peakLoc_r.y - init_val[2])*(360.0/(r_src1.rows));
		theta = rotation*(M_PI/180.0);
	} else {
		theta = state.at(2);
		rotation = theta*180.0/M_PI;
	}

	cv::Point2f center(src2.cols/2.0, src2.rows/2.0);
	cv::Mat rot = cv::getRotationMatrix2D(center, rotation, 1.0);
	cv::Mat derot_cart;
	cv::warpAffine(src2, derot_cart, rot, src2.size());

	cv::Point2d peakLoc;
	if(flag == true) {
		peakLoc = cv::phaseCorrelate(src1, derot_cart, cv::noArray(), &phaseCorr);
	
		factor = ratio;
		x = (init_val[0]-peakLoc.x)*RESOL*factor;
		y = (init_val[1]-peakLoc.y)*RESOL*factor;
	} else {
		// Fine refinement disabled — phaseCorrelateWindow produces
		// unreliable peaks that degrade trajectory (509m vs 69m baseline).
		// The coarse estimate is used directly.
		x = state.at(0);
		y = state.at(1);
	}

	// Note: init_val calibration is now done via self-correlation in
	// GraphOptimizer::calibrateInitVal(), not here.

	std::array<double, 3> d_state = {x, y, theta};
	return d_state;
}