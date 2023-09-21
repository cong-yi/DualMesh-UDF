#pragma once
#include <Eigen/Dense>

class DualPoint;

class QEF
{
public:
    // Ax=b
    Eigen::MatrixXd A;
    Eigen::VectorXd b;

    // Default constructor
    QEF() {}
    // Copy Constructor
    QEF(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) : A(A), b(b) {}
    // Destructor
    ~QEF() {}

    // Solve QEF
    Eigen::VectorXd solve(double svd_threshold = 0.1);

    // Compute the residual of the input solution
    double residual(const Eigen::VectorXd& v);

    bool is_plane = false;
    bool is_line = false;
    Eigen::VectorXd dir;
    Eigen::VectorXd point;
};

DualPoint solve_point(QEF& qef, const Eigen::VectorXd& min_corner, const Eigen::VectorXd& max_corner, double svd_threshold, double corner_margin, double edge_margin, double plane_margin);