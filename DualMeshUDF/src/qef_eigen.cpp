#include <iostream>
#include <fstream>
#include <stdexcept>
#include <omp.h>
#include <functional>
#include <pybind11/eigen.h>
#include "octree.h"
#include "qef.h"
#include "mesh_utils.h"


Eigen::VectorXd QEF::solve(double svd_threshold)
{
	Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Eigen::MatrixXd Q = qr.householderQ();
    Eigen::VectorXd Qb = (Q.transpose() * b).topRows(3);
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
    R.conservativeResize(3, 3);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(R, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::MatrixXd diag_s_inv = Eigen::MatrixXd::Zero(3, 3);

    Eigen::VectorXd s = svd.singularValues();

    for(int i = 0; i < 3; ++i)
    {
	    if(s[i] > s[0] * svd_threshold)
	    {
	        diag_s_inv(i, i) = 1.0/s[i];
	    }
	    else
	    {
	        s[i] = 0;
	        if(i == 1)
	        {
	            is_plane = true;
	        }
	        if(i == 2 && !is_plane)
	        {
	            is_line = true;
	        }
	    }
    }

    Eigen::MatrixXd A_inv = svd.matrixV() * diag_s_inv * svd.matrixU().transpose();
    point = A_inv * Qb;

    if(is_plane)
    {
        // Normal constraint
        dir = svd.matrixV().topRows(3).col(0).stableNormalized();
    }
    else if(is_line)
    {
        // Direction constraint
        dir = svd.matrixV().topRows(3).col(2).stableNormalized();
    }

    if(point.array().isNaN().any())
    {
        std::cout << "NaN values in the QEF solution" << std::endl;
    }

    return point;
}


Eigen::Vector3d line_plane_intersection(Eigen::Vector3d line_dir, Eigen::Vector3d line_p, Eigen::Vector3d normal, const Eigen::Vector3d plane_p)
{
    line_dir.normalize();
    normal.normalize();

    const double d = normal.dot(plane_p);

    const double n_dot_dir = normal.dot(line_dir);

    // if the line is parallel to the plane
    if(abs(n_dot_dir) < 1e-4)
    {
        // return an empty point
        return Eigen::Vector3d(100, 100, 100);
    }

    const double t = (d - normal.dot(line_p)) / n_dot_dir;
    return line_p + line_dir * t;
}

bool check_plane_cell_intersection(Eigen::VectorXd normal, const Eigen::VectorXd& plane_p, const Eigen::VectorXd& min_corner, const Eigen::VectorXd& max_corner)
{
    Eigen::VectorXd center = (min_corner + max_corner) * 0.5;
    Eigen::VectorXd extent = max_corner - center;
    double r = normal.cwiseAbs().dot(extent);
    double s = abs(normal.dot(center) - normal.dot(plane_p));

    return s <= r;
}

DualPoint solve_point(QEF& qef, const Eigen::VectorXd& min_corner, const Eigen::VectorXd& max_corner, double svd_threshold, double corner_factor, double edge_factor, double plane_factor)
{
    // solve the qef
    Eigen::VectorXd v = qef.solve(svd_threshold);

    // check if it is inside without any offset
    bool is_point_inside = is_inside(v, min_corner, max_corner, 1e-5);

    // if the reconstructed point is a corner point
    if(!qef.is_line && !qef.is_plane)
    {
        // use corner factor
        if (is_inside(v, min_corner, max_corner, corner_factor))
        {
            return DualPoint(v, CORNER, Eigen::VectorXd(), is_point_inside);
        }
        else
        {
	        return DualPoint(Eigen::VectorXd(), NONETYPE, Eigen::VectorXd(), false);
        }
    }
    else
    {
        std::vector<Eigen::VectorXd> v_vec(6);
        std::vector<Eigen::VectorXd> v_candidates;

        if(qef.is_line)
        {
            v_vec[0] = line_plane_intersection(qef.dir, qef.point, Eigen::Vector3d(1, 0, 0), min_corner);
            v_vec[1] = line_plane_intersection(qef.dir, qef.point, Eigen::Vector3d(0, 1, 0), min_corner);
            v_vec[2] = line_plane_intersection(qef.dir, qef.point, Eigen::Vector3d(0, 0, 1), min_corner);
            v_vec[3] = line_plane_intersection(qef.dir, qef.point, Eigen::Vector3d(1, 0, 0), max_corner);
            v_vec[4] = line_plane_intersection(qef.dir, qef.point, Eigen::Vector3d(0, 1, 0), max_corner);
            v_vec[5] = line_plane_intersection(qef.dir, qef.point, Eigen::Vector3d(0, 0, 1), max_corner);

            for(size_t i = 0; i < v_vec.size(); ++i)
            {
                // check if the intersection point is inside the cell
                if(is_inside(v_vec[i], min_corner, max_corner, 1e-5))
                {
                    v_candidates.emplace_back(v_vec[i]);
                }
            }
            if (v_candidates.empty())
            {
                // if there is no intersection inside the cell, enlarge the cell accordingly
				for (size_t i = 0; i < v_vec.size(); ++i)
				{
					if (is_inside(v_vec[i], min_corner, max_corner, edge_factor))
					{
						v_candidates.push_back(v_vec[i]);
					}
				}
            }
        }
        else// then it must be a plane
        {
            v_vec.resize(12);

            v_vec[0] = line_plane_intersection(Eigen::Vector3d(1, 0, 0), min_corner, qef.dir, qef.point);
            v_vec[1] = line_plane_intersection(Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(min_corner[0], min_corner[1], max_corner[2]), qef.dir, qef.point);
            v_vec[2] = line_plane_intersection(Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(min_corner[0], max_corner[1], min_corner[2]), qef.dir, qef.point);
            v_vec[3] = line_plane_intersection(Eigen::Vector3d(1, 0, 0), max_corner, qef.dir, qef.point);
            
            v_vec[4] = line_plane_intersection(Eigen::Vector3d(0, 1, 0), min_corner, qef.dir, qef.point);
            v_vec[5] = line_plane_intersection(Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(min_corner[0], min_corner[1], max_corner[2]), qef.dir, qef.point);
            v_vec[6] = line_plane_intersection(Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(max_corner[0], min_corner[1], min_corner[2]), qef.dir, qef.point);
            v_vec[7] = line_plane_intersection(Eigen::Vector3d(0, 1, 0), max_corner, qef.dir, qef.point);
            
            v_vec[8] = line_plane_intersection(Eigen::Vector3d(0, 0, 1), min_corner, qef.dir, qef.point);
            v_vec[9] = line_plane_intersection(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(min_corner[0], max_corner[1], min_corner[2]), qef.dir, qef.point);
            v_vec[10] = line_plane_intersection(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(max_corner[0], min_corner[1], min_corner[2]), qef.dir, qef.point);
            v_vec[11] = line_plane_intersection(Eigen::Vector3d(0, 0, 1), max_corner, qef.dir, qef.point);

            for(size_t i = 0; i < v_vec.size(); ++i)
            {
                // check if the intersection point is inside the cell
                if(is_inside(v_vec[i], min_corner, max_corner, 1e-5))
                {
                    v_candidates.push_back(v_vec[i]);
                }
            }
            if(v_candidates.empty())
            {
                // if there is no intersection inside the cell, enlarge the cell accordingly
                for(size_t i = 0; i < v_vec.size(); ++i)
                {
                    if(is_inside(v_vec[i], min_corner, max_corner, plane_factor))
                    {
                        v_candidates.push_back(v_vec[i]);
                    }
                }
            }

        }

        if(v_candidates.empty())
        {
        	return DualPoint(Eigen::VectorXd(), NONETYPE, Eigen::VectorXd(), false);
        }
        else
        {
            Eigen::VectorXd best_v(3);
            best_v.setZero();
            for(const auto& v : v_candidates)
            {
                best_v += v;
            }
            best_v /= v_candidates.size();

            // check if the solution is inside the cell
            is_point_inside = is_inside(best_v, min_corner, max_corner, 1e-4);
            if(qef.is_line)
            {
                return DualPoint(best_v, EDGE, qef.dir, is_point_inside);
            }
            else
            {
                return DualPoint(best_v, PLANE, qef.dir, is_point_inside);
            }

        }
    }
}


// Sampling function
Eigen::MatrixXd sampling_on_each_triangle(const Eigen::MatrixXd& mesh_v, const Eigen::MatrixXi& mesh_f, int num_samples_per_face) {
    // Initialize the output matrix for sample points
    Eigen::MatrixXd sample_points(mesh_f.rows() * num_samples_per_face, 3);

    // Loop over each face
    for (int i = 0; i < mesh_f.rows(); ++i)
    {
        // Get the vertices of the current triangle
        Eigen::RowVector3d v0 = mesh_v.row(mesh_f(i, 0));
        Eigen::RowVector3d v1 = mesh_v.row(mesh_f(i, 1));
        Eigen::RowVector3d v2 = mesh_v.row(mesh_f(i, 2));

        sample_points.row(i * num_samples_per_face) = (v0 + v1) * 0.5;
        sample_points.row(i * num_samples_per_face + 1) = (v0 + v2) * 0.5;
        sample_points.row(i * num_samples_per_face + 2) = (v1 + v2) * 0.5;
        sample_points.row(i * num_samples_per_face + 3) = (v0 + v1 + v2) / 3.0;
        sample_points.row(i * num_samples_per_face + 4) = v0;
        sample_points.row(i * num_samples_per_face + 5) = v1;
        sample_points.row(i * num_samples_per_face + 6) = v2;

        // // Generate random samples using uniform distribution
        // std::uniform_real_distribution<double> unif_dist(0.0, 1.0);
        // std::default_random_engine random_engine;
        // for (int j = 0; j < num_samples_per_face; ++j)
        // {
        //     // Generate random barycentric coordinates
        //     double u = unif_dist(random_engine);
        //     double v = unif_dist(random_engine) * (1 - u);
        //     double w = 1 - u - v;
        //
        //     // Compute the sample point in the triangle
        //     Eigen::Vector3d sample_point = u * v0 + v * v1 + w * v2;
        //
        //     // Store the sample point in the output matrix
        //     sample_points.row(i * num_samples_per_face + j) = sample_point;
        // }
    }

    return sample_points;
}


double QEF::residual(const Eigen::VectorXd& v)
{
    return (A * v - b).norm() / b.rows();
}



