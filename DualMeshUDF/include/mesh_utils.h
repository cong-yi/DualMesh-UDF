#pragma once
#include <Eigen/Dense>
#include <vector>

const int CORNER = 0;
const int EDGE = 1;
const int PLANE = 2;
const int NONETYPE = 3;

class DualPoint
{
public:
    DualPoint() { m_is_inside = false; }
    DualPoint(const Eigen::VectorXd& point, int type, const Eigen::VectorXd& dir, bool is_inside) : m_point(point), m_type(type), m_dir(dir), m_is_inside(is_inside) {}
    DualPoint(const DualPoint& dual_point)
    {
        m_point = dual_point.m_point;
        m_type = dual_point.m_type;
        m_dir = dual_point.m_dir;
        m_is_inside = dual_point.m_is_inside;
    }
    Eigen::VectorXd m_point;
    int m_type; // 0 for CORNER, 1 for EDGE, 2 for PLANE
    Eigen::VectorXd m_dir; // line dir or plane normal
    bool m_is_inside;
};

// check if the point is inside the AABB with an offset factor
bool is_inside(const Eigen::VectorXd& point, Eigen::VectorXd min_corner, Eigen::VectorXd max_corner, double offset);

// check if points are coplanar
bool are_coplanar(const Eigen::MatrixXd& points, double threshold=0.25);

// get the normal vector of a set of planar points
Eigen::VectorXd get_normal_of_planar_points(const Eigen::MatrixXd& points);

// get the intersection point of a line and a plane
Eigen::Vector3d line_plane_intersection(Eigen::Vector3d line_dir, Eigen::Vector3d line_p, Eigen::Vector3d normal, const Eigen::Vector3d plane_p);

// triangulate quad faces
Eigen::MatrixXi triangulate_faces(const Eigen::MatrixXd& mesh_v, const Eigen::MatrixXi& mesh_f, const std::vector<int>& v_type, const Eigen::MatrixXd& mesh_v_dir);

// output the mesh
bool write_obj(const std::string& str, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);