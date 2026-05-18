#include <gtsam_custom/hs_rotation_factor.h>

namespace gtsam {

/* ************************************************************************* */
PharaoRotFactor::PharaoRotFactor(Key poseKey1, Key poseKey2, double mtheta,
    const SharedNoiseModel& model)
: Base(model, poseKey1, poseKey2), mtheta_(mtheta)
{
}

/* ************************************************************************* */
Vector PharaoRotFactor::evaluateError(const Pose2& pose1, const Pose2& pose2,
    boost::optional<Matrix&> H1, boost::optional<Matrix&> H2) const {

  // Fix: use atan2(sin,cos) for correct modular angle difference in all quadrants.
  // Original code used ±M_PI/2 thresholds which fired incorrectly for θ ∈ (π/2, π),
  // producing wrong residuals that made the GTSAM Hessian singular.
  double hx = std::atan2(std::sin(pose2.theta() - pose1.theta()),
                         std::cos(pose2.theta() - pose1.theta()));

  if (H1) {
    *H1 = Matrix::Zero(1,3);
    (*H1)(0, 2) = -1.0;
  }

  if (H2) {
    *H2 = Matrix::Zero(1,3);
    (*H2)(0, 2) = 1.0;
  }
  return (Vector(1) << hx - mtheta_).finished();
}

/* ************************************************************************* */
bool PharaoRotFactor::equals(const NonlinearFactor& expected, double tol) const {
  const This *e = dynamic_cast<const This*> (&expected);
  return e != nullptr && Base::equals(*e, tol) && std::abs(this->mtheta_ - e->mtheta_) < tol;
}

/* ************************************************************************* */
void PharaoRotFactor::print(const std::string& s, const KeyFormatter& keyFormatter) const {
  std::cout << s << "PharaoRotFactor, relative rotation = " << mtheta_ << std::endl;
  Base::print("", keyFormatter);
}
/* ************************************************************************* */

} // \namespace gtsam
