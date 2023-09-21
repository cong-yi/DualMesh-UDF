#include "mesh_utils.h"
#include <fstream>

bool is_inside(const Eigen::VectorXd& point, Eigen::VectorXd min_corner, Eigen::VectorXd max_corner, double offset = 0.0)
{
    // compute the offset direction
    const Eigen::VectorXd offset_dir = max_corner - min_corner;

    // compute the extended min and max corners
    Eigen::VectorXd ext_min_corner = min_corner - offset_dir * offset;
    Eigen::VectorXd ext_max_corner = max_corner + offset_dir * offset;

    // check if the point is inside the extended AABB
    return (point.array() < ext_max_corner.array()).all() && (point.array() > ext_min_corner.array()).all();
}


bool are_coplanar(const Eigen::MatrixXd& points, double threshold)
{
    Eigen::MatrixXd centered_points = points.rowwise() - points.colwise().mean();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered_points, Eigen::ComputeFullV);
    return svd.singularValues()[2] < svd.singularValues()[0] * threshold;
}

Eigen::VectorXd get_normal_of_planar_points(const Eigen::MatrixXd& points)
{
    // Check that points is a 4x3 matrix
    if (points.rows() < 3 || points.cols() != 3) {
        throw std::invalid_argument("Input matrix must be 4x3");
    }

    // Compute the centroid of the points
    Eigen::RowVector3d centroid = points.colwise().mean();

    // Compute the "design matrix" A
    Eigen::MatrixXd A(points.rows(), 3);
    for (int i = 0; i < points.rows(); i++) {
        A.row(i) = points.row(i) - centroid;
    }

    // Compute the SVD of A
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Extract the nullspace of A (i.e., the vector that is orthogonal to the plane)
    Eigen::VectorXd normal = svd.matrixV().col(2);

    return normal;
}

Eigen::MatrixXi triangulate_faces(const Eigen::MatrixXd& mesh_v, const Eigen::MatrixXi& mesh_f, const std::vector<int>& v_type, const Eigen::MatrixXd& mesh_v_dir)
{
    Eigen::MatrixXi mesh_tri_f(mesh_f.rows() * 2, 3);

    int new_face_id = 0;
    for (int i = 0; i < mesh_f.rows(); i++)
    {
        const auto& f = mesh_f.row(i);

        Eigen::MatrixXd f_v(4, 3);
        Eigen::MatrixXd f_v_dir(4, 3);

        for (int j = 0; j < f.size(); ++j)
        {
            f_v.row(j) = mesh_v.row(f(j));
            f_v_dir.row(j) = mesh_v_dir.row(f(j));
        }
        
        bool is_quad_valid = true;
        if (are_coplanar(f_v))
        {
            // for planar points, compute the normal of the plane
            Eigen::VectorXd normal = get_normal_of_planar_points(f_v).normalized();
            for (int j = 0; j < f.size(); ++j)
            {
                if (v_type[f(j)] == PLANE)
                {
                    if (std::abs(f_v_dir.row(j) * normal) < 0.85)// theta > 30
                    {
                        is_quad_valid = false;
                        break;
                    }
                }
                else if (v_type[f(j)] == EDGE)
                {
                    if (std::abs(f_v_dir.row(j) * normal) > 0.5)// theta < 60
                    {
                        is_quad_valid = false;
                        break;
                    }
                }
            }
            if (is_quad_valid)
            {
                // arbitrary triangulation is valid
                mesh_tri_f.middleRows(new_face_id, 2) << f(0), f(1), f(2), f(2), f(3), f(0);
                new_face_id += 2;
            }
        }
        else
        {
            int t = -1;
            // check all points
            for (int j = 0; j < f.size(); ++j)
            {
                if (v_type[f(j)] == PLANE)
                {
                    Eigen::MatrixXd tri_f_v(3, 3);
                    tri_f_v.row(0) = f_v.row(j);
                    tri_f_v.row(1) = f_v.row((j + 1) % 4);
                    tri_f_v.row(2) = f_v.row((j + 3) % 4);
                    Eigen::VectorXd tri_normal = get_normal_of_planar_points(tri_f_v);
                    if (std::abs(f_v_dir.row(j) * tri_normal) < 0.85)
                    {
                        is_quad_valid = false;
                        t = -1;
                        break;
                    }
                    else
                    {
                        if (t == -1 || t + 2 == j)
                        {
                            t = j;
                        }
                        else
                        {
                            is_quad_valid = false;
                            t = -1;
                            break;
                        }
                    }
                }
            }
            if (is_quad_valid)
            {
                if (t != -1)
                {
                    mesh_tri_f.middleRows(new_face_id, 2) << f(t), f((t + 1) % 4), f((t + 3) % 4), f((t + 1) % 4), f((t + 2) % 4), f((t + 3) % 4);
                    new_face_id += 2;
                }
                else // no planar vertex
                {
                    if (v_type[f(0)] == EDGE && v_type[f(2)] == EDGE)
                    {
                        mesh_tri_f.middleRows(new_face_id, 2) << f(0), f(1), f(2), f(2), f(3), f(0);
                    }
                    else
                    {
                        //arbitrary triangulation is valid
                        mesh_tri_f.middleRows(new_face_id, 2) << f(0), f(1), f(3), f(1), f(2), f(3);
                    }
                    new_face_id += 2;
                }
            }
        }
    }
    mesh_tri_f.conservativeResize(new_face_id, 3);

    return mesh_tri_f;
}


bool write_obj(
    const std::string& str,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    using namespace std;
    using namespace Eigen;
    assert(V.cols() == 3 && "V should have 3 columns");
    ofstream s(str);
    if (!s.is_open())
    {
        fprintf(stderr, "IOError: writeOBJ() could not open %s\n", str.c_str());
        return false;
    }
    s <<
        V.format(IOFormat(FullPrecision, DontAlignCols, " ", "\n", "v ", "", "", "\n")) <<
        (F.array() + 1).format(IOFormat(FullPrecision, DontAlignCols, " ", "\n", "f ", "", "", "\n"));
    return true;
}
