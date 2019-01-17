#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

class Tools {
public:
  Tools(); //Constructor
  virtual ~Tools(); //Destructor

  Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, const std::vector<Eigen::VectorXd> &ground_truth); //calculate RMSE
  Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd& x_state); //calculate Jacobians
};
#endif 