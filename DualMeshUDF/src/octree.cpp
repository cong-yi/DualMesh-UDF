#include "octree.h"
#include "qef.h"
#include "mesh_utils.h"
#include <iostream>
#include <stdexcept>
#include <omp.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>


namespace py = pybind11;

Eigen::MatrixXi g_child_order = []
{
    Eigen::MatrixXi m(8, 3);
    m << 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1;
    return m;
}();

// From Dual Contouring
const int cellProcFaceMask[12][3] = { {0,4,0},{1,5,0},{2,6,0},{3,7,0},{0,2,1},{4,6,1},{1,3,1},{5,7,1},{0,1,2},{2,3,2},{4,5,2},{6,7,2} };

const int cellProcEdgeMask[6][5] = { {0,1,2,3,0},{4,5,6,7,0},{0,4,1,5,1},{2,6,3,7,1},{0,2,4,6,2},{1,3,5,7,2} };

const int faceProcFaceMask[3][4][3] = {
    {{4,0,0},{5,1,0},{6,2,0},{7,3,0}},
    {{2,0,1},{6,4,1},{3,1,1},{7,5,1}},
    {{1,0,2},{3,2,2},{5,4,2},{7,6,2}}
};

const int faceProcEdgeMask[3][4][6] = {
    {{1,4,0,5,1,1},{1,6,2,7,3,1},{0,4,6,0,2,2},{0,5,7,1,3,2}},
    {{0,2,3,0,1,0},{0,6,7,4,5,0},{1,2,0,6,4,2},{1,3,1,7,5,2}},
    {{1,1,0,3,2,0},{1,5,4,7,6,0},{0,1,5,0,4,1},{0,3,7,2,6,1}}
};

const int edgeProcEdgeMask[3][2][5] = {
    {{3,2,1,0,0},{7,6,5,4,0}},
    {{5,1,4,0,1},{7,3,6,2,1}},
    {{6,4,2,0,2},{7,5,3,1,2}},
};


OctreeNode::OctreeNode(Eigen::VectorXi min_corner_id, Eigen::VectorXi max_corner_id, OctreeNode* parent, int depth)
{
    m_parent = parent;
    m_children = std::vector<OctreeNode>();
    m_depth = depth;

    m_min_corner_id = min_corner_id;
    m_max_corner_id = max_corner_id;

    m_centroid_id = (m_min_corner_id + m_max_corner_id) / 2;

    m_edge_span = max_corner_id[0] - min_corner_id[0];
    
    m_vertex_idx = -1;
    m_node_type = "empty";
}


bool OctreeNode::subdivide()
{
    if (m_children.empty())
    {
        m_children.reserve(8);
        int child_edge_span = m_edge_span / 2;
        Eigen::VectorXi max_corner_id_offset = Eigen::VectorXi::Constant(3, child_edge_span);
        for (int i = 0; i < 8; ++i)
        {
            Eigen::VectorXi child_min_corner_id = m_min_corner_id + (g_child_order.row(i) * child_edge_span).transpose();

            Eigen::VectorXi child_max_corner_id = child_min_corner_id + max_corner_id_offset;

            m_children.emplace_back(child_min_corner_id, child_max_corner_id, this, m_depth + 1);
        }
        return true;
    }
    else
    {
        throw std::invalid_argument("Children are not empty!");
    }
}


inline Eigen::MatrixXi OctreeNode::generate_sample_grids(const Eigen::MatrixXi& sampling_pattern)
{
    return sampling_pattern.rowwise() + m_min_corner_id.transpose();
}


Octree::Octree(int max_depth, Eigen::VectorXd min_corner, Eigen::VectorXd max_corner, int sampling_depth)
{
    m_max_depth = max_depth;
    m_min_corner = min_corner;
    m_max_corner = max_corner;
    m_sampling_depth = sampling_depth;

    m_capacity_per_dim = std::pow(2, max_depth + sampling_depth) + 1;
    std::cout << "The resolution is " << (m_capacity_per_dim - 1) / std::pow(2, m_sampling_depth) << "." << std::endl;
    std::cout << "The number of samples per cell is " << std::pow(m_sampling_depth + 2, 3) << "." << std::endl;

    Eigen::VectorXi root_node_min_corner_id = Eigen::VectorXi::Zero(3);
    Eigen::VectorXi root_node_max_corner_id = Eigen::VectorXi::Constant(3, m_capacity_per_dim - 1);

    m_root_node = OctreeNode(root_node_min_corner_id, root_node_max_corner_id);

    // compute the unit edge length
    m_unit_edge_length = (max_corner[0] - min_corner[0]) / double(m_capacity_per_dim - 1);

    // initialize the nodes vector and push the root node into it
    m_leaf_nodes.clear();
    m_leaf_nodes.push_back(std::ref(m_root_node));

    // set all as false
    m_proj_validity.resize(m_capacity_per_dim * m_capacity_per_dim * m_capacity_per_dim, false);
    m_grid_existence.resize(m_capacity_per_dim * m_capacity_per_dim * m_capacity_per_dim, false);
    m_gid2sid.resize(m_capacity_per_dim * m_capacity_per_dim * m_capacity_per_dim, -1);

    // generate sampling pattern
    m_sampling_pattern = generate_sample_pattern(std::pow(2, sampling_depth) + 1);

    m_centroid_coords.clear();
    m_centroid_udfs.clear();
    m_centroid_grads.clear();

    m_mesh_f.clear();
}


Eigen::VectorXd Octree::convert_grid_to_coords(const Eigen::VectorXi& grid)
{
    return m_min_corner + grid.cast<double>() * m_unit_edge_length;
}


Eigen::MatrixXd Octree::get_centroids_of_new_nodes()
{
    Eigen::MatrixXd centroids(m_leaf_nodes.size(), 3);
    for (size_t i = 0; i < m_leaf_nodes.size(); ++i)
    {
        if (m_leaf_nodes[i].get().m_edge_span <= 1)
        {
            // insufficient spatial subdivision, should never happen
            centroids.row(i) = (convert_grid_to_coords(m_leaf_nodes[i].get().m_min_corner_id).transpose() + convert_grid_to_coords(m_leaf_nodes[i].get().m_max_corner_id).transpose()) * 0.5;

            throw std::invalid_argument("Insufficient pre-allocated depth!");
        }

        // get the centroid grid
        Eigen::VectorXi centroid_grid = m_leaf_nodes[i].get().m_centroid_id;
        centroids.row(i) = convert_grid_to_coords(centroid_grid).transpose();

        // convert to global id
        int gid = encode(centroid_grid);

        m_grid_existence[gid] = true;
        m_gid2sid[gid] = m_centroid_coords.size();

        // prepare data for computing qef
        m_centroid_coords.emplace_back(centroids.row(i).transpose());
        m_centroid_udfs.emplace_back(-1);
        m_centroid_grads.emplace_back(Eigen::Vector3d());

    }
    return centroids;
}


int Octree::encode(const Eigen::VectorXi& grid)
{
    return (grid[0] * m_capacity_per_dim + grid[1]) * m_capacity_per_dim + grid[2];
}


Eigen::VectorXi Octree::decode(int gid)
{
    Eigen::VectorXi grid = Eigen::VectorXi::Zero(3);
    grid[0] = gid / std::pow(m_capacity_per_dim, 2);
    grid[1] = (gid / m_capacity_per_dim) % m_capacity_per_dim;
    grid[2] = gid % m_capacity_per_dim;
    return grid;
}


py::tuple Octree::get_samples_of_new_nodes()
{
    std::vector<int> new_grid_indices_vec;
    new_grid_indices_vec.reserve(m_leaf_nodes.size() * m_sampling_pattern.rows());
    std::vector<Eigen::Vector3d> new_grid_coords_vec;
    new_grid_coords_vec.reserve(m_leaf_nodes.size() * m_sampling_pattern.rows());

	// projection to sampling id list
    m_proj_sid.clear();
    m_proj_sid.reserve(m_leaf_nodes.size() * m_sampling_pattern.rows());

    // projection to global id list
    m_proj_gid.clear();
    m_proj_gid.reserve(m_leaf_nodes.size() * m_sampling_pattern.rows());

    int id_counter = 0;
    for (size_t i = 0; i < m_leaf_nodes.size(); ++i)
    {
        m_leaf_nodes[i].get().m_sample_grids = m_leaf_nodes[i].get().generate_sample_grids(m_sampling_pattern);

        const Eigen::MatrixXi& sampling_grids = m_leaf_nodes[i].get().m_sample_grids;

        // batch convert 3D grids to global ids efficiently
        Eigen::VectorXi gids = (sampling_grids.col(0) * m_capacity_per_dim + sampling_grids.col(1)) * m_capacity_per_dim + sampling_grids.col(2);

        for (int j = 0; j < sampling_grids.rows(); ++j)
        {
            // those grids are not checked
            if (!m_grid_existence[gids[j]])
            {
                new_grid_indices_vec.emplace_back(gids[j]);

                // compute the coordinates of the samples
                new_grid_coords_vec.emplace_back(convert_grid_to_coords(sampling_grids.row(j)));

                // mark it as visited
                m_grid_existence[gids[j]] = true;

                m_gid2sid[gids[j]] = m_centroid_coords.size() + id_counter;
                ++id_counter;
            }
            if (!m_proj_validity[gids[j]])
            {
                // mark the projection validity as true
                m_proj_validity[gids[j]] = true;
                m_proj_sid.emplace_back(m_gid2sid[gids[j]]);
                m_proj_gid.emplace_back(gids[j]);
            }
        }
    }

    m_sample_coords.resize(m_centroid_coords.size() + id_counter, 3);
    m_sample_grads.resize(m_centroid_coords.size() + id_counter, 3);
    m_sample_udfs.resize(m_centroid_coords.size() + id_counter);

    for (int i = 0; i < m_centroid_coords.size(); ++i)
    {
        m_sample_coords.row(i) = m_centroid_coords[i];
        m_sample_grads.row(i) = m_centroid_grads[i];
        m_sample_udfs[i] = m_centroid_udfs[i];
    }

    Eigen::MatrixXd new_grid_coords(new_grid_coords_vec.size(), 3);
    for (int i = 0; i < new_grid_coords.rows(); ++i)
    {
        new_grid_coords.row(i) = new_grid_coords_vec[i].transpose();
    }
    m_sample_coords.bottomRows(new_grid_coords_vec.size()) = new_grid_coords;

    Eigen::VectorXi new_grid_indices = Eigen::Map<Eigen::VectorXi>(new_grid_indices_vec.data(), new_grid_indices_vec.size());

    return py::make_tuple(new_grid_indices, new_grid_coords);
}


void Octree::set_new_grid_data(const Eigen::VectorXi& new_grid_indices, const Eigen::VectorXf& new_grid_udfs, const Eigen::MatrixXf& new_grid_grads)
{
#pragma omp parallel for
    for (int i = 0; i < new_grid_indices.size(); ++i)
    {
        int data_id = m_gid2sid[new_grid_indices[i]];
        m_sample_udfs[data_id] = new_grid_udfs(i);
        m_sample_grads.row(data_id) = new_grid_grads.row(i).cast<double>();
    }
}


void Octree::adaptive_subdivide(const Eigen::VectorXf& centroid_udfs, const Eigen::MatrixXf& centroid_grads, double margin)
{
    // compute the diagonal length of the cell
    double node_edge_length = m_unit_edge_length * m_leaf_nodes[0].get().m_edge_span;
    double half_diagonal_length = std::sqrt(3) * node_edge_length * 0.5;

    // subdivide valid cells
    std::vector<std::reference_wrapper<OctreeNode> > new_nodes;
    for (size_t i = 0; i < m_leaf_nodes.size(); ++i)
    {
        if (m_leaf_nodes[i].get().m_edge_span <= 1)
        {
            throw std::invalid_argument("Insufficient pre-allocated depth!");
        }
        // set the udf and gradient values to the current leaf node
        Eigen::VectorXi centroid_grid = m_leaf_nodes[i].get().m_centroid_id;
        // convert the 3D centroid grid id to the global id, and then to the sampling id
        int sid = m_gid2sid[encode(centroid_grid)];
        m_centroid_udfs[sid] = centroid_udfs(i);
        m_centroid_grads[sid] = centroid_grads.row(i).cast<double>().transpose();

        // Our key criterion
        if (m_centroid_udfs[sid] < half_diagonal_length + margin)
        {
            // check if current depth reaches the max depth
            if (m_leaf_nodes[i].get().m_depth < m_max_depth)
            {
                m_leaf_nodes[i].get().subdivide();
                m_leaf_nodes[i].get().m_node_type = "interval";
                for (size_t j = 0; j < m_leaf_nodes[i].get().m_children.size(); ++j)
                {
                    new_nodes.emplace_back(std::ref(m_leaf_nodes[i].get().m_children[j]));
                }
            }
            else
            {
                // put valid cells into a new container
                new_nodes.emplace_back(m_leaf_nodes[i]);
            }
        }
    }

    m_leaf_nodes.clear();
    m_leaf_nodes = new_nodes;
}


void Octree::set_grid_validity(const Eigen::VectorXi& indices, const std::vector<bool>& validity_mask)
{
#pragma omp parallel for
    for (int i = 0; i < validity_mask.size(); ++i)
    {
        if (!validity_mask[i])
        {
            m_proj_validity[indices[i]] = false;
        }
    }
}


py::tuple Octree::get_projections_for_checking_validity()
{

    Eigen::MatrixXd projections = m_sample_coords(m_proj_sid, Eigen::all).array() - m_sample_grads(m_proj_sid, Eigen::all).array().colwise() * m_sample_udfs(m_proj_sid).array();
    return py::make_tuple(m_proj_gid, projections);
}


Eigen::MatrixXi Octree::generate_sample_pattern(int per_edge_sample_num)
{
    std::vector<Eigen::VectorXi> cartesian_pair(per_edge_sample_num);
    for (int i = 0; i < cartesian_pair.size(); ++i)
    {
        cartesian_pair[i] = Eigen::Vector3i(i, i, i);
    }

    Eigen::MatrixXi sample_grids(cartesian_pair.size() * cartesian_pair.size() * cartesian_pair.size(), 3);
    int row_idx = 0;
    for (auto a : cartesian_pair)
    {
        for (auto b : cartesian_pair)
        {
            for (auto c : cartesian_pair)
            {
                sample_grids(row_idx, 0) = a(0);
                sample_grids(row_idx, 1) = b(1);
                sample_grids(row_idx, 2) = c(2);
                ++row_idx;
            }
        }
    }
    return sample_grids;
}


void Octree::batch_solve(double sample_udf_threshold, double corner_margin, double edge_margin, double plane_margin, double singular_value_threshold)
{
    // store results in the leaf nodes
    std::vector<DualPoint> results(m_leaf_nodes.size());

    // clear mesh vertices
    m_mesh_v.clear();
    m_v_type.clear();

#pragma omp parallel for
    for (size_t i = 0; i < m_leaf_nodes.size(); ++i)
    {
        // get min and max corners
        Eigen::VectorXd min_corner = m_sample_coords.row(m_gid2sid[encode(m_leaf_nodes[i].get().m_min_corner_id)]);
        Eigen::VectorXd max_corner = m_sample_coords.row(m_gid2sid[encode(m_leaf_nodes[i].get().m_max_corner_id)]);

        int sample_num = m_leaf_nodes[i].get().m_sample_grids.rows();

        Eigen::MatrixXd coords(sample_num, 3);
        Eigen::MatrixXd grads(sample_num, 3);
        Eigen::VectorXd udfs(sample_num);
        int valid_num = 0;
        for (int j = 0; j < m_leaf_nodes[i].get().m_sample_grids.rows(); ++j)
        {
            int grid_index = encode(m_leaf_nodes[i].get().m_sample_grids.row(j));
            int sid = m_gid2sid[grid_index];
            double udf = m_sample_udfs[sid];

            if (udf > sample_udf_threshold && m_proj_validity[grid_index])
            {
                coords.row(valid_num) = m_sample_coords.row(sid);
                grads.row(valid_num) = m_sample_grads.row(sid);
                udfs(valid_num) = udf;
                valid_num++;
                
            }
        }

        coords.conservativeResize(valid_num, 3);
        grads.conservativeResize(valid_num, 3);
        udfs.conservativeResize(valid_num);

        Eigen::MatrixXd projections = coords - (grads.array().colwise() * udfs.array()).matrix();

        // set the center of projections to the original point
        int inside_num = 0;
        Eigen::Vector3d proj_center(0, 0, 0);
        for(int j = 0; j < projections.rows(); ++j)
        {
            if (is_inside(projections.row(j), min_corner, max_corner, 0.01))
            {
                proj_center += projections.row(j).transpose();
                inside_num++;
            }
        }
        if(inside_num > 0)
        {
            proj_center /= inside_num;
        }
        else
        {
            Eigen::VectorXi centroid_grid = m_leaf_nodes[i].get().m_centroid_id;
            Eigen::VectorXd centroid_coords = convert_grid_to_coords(centroid_grid);
            proj_center = centroid_coords;
        }

        min_corner -= proj_center;
        max_corner -= proj_center;

        coords = (coords.rowwise() - proj_center.transpose()).eval();
        projections = (projections.rowwise() - proj_center.transpose()).eval();

        // check if there are valid samples
        if (valid_num > 0)
        {
            QEF qef;
            if (valid_num > 2)
            {
                Eigen::VectorXd b = coords.cwiseProduct(grads).rowwise().sum() - udfs;
                qef.A = grads;
                qef.b = b;
                results[i] = solve_point(qef, min_corner, max_corner, singular_value_threshold, corner_margin, edge_margin, plane_margin);
            }
            if (results[i].m_point.size() == 0)
            {
				Eigen::MatrixXd filtered_coords(valid_num, 3);
				Eigen::MatrixXd filtered_grads(valid_num, 3);
				Eigen::VectorXd filtered_udfs(valid_num);
 
				valid_num = 0;
 
				for (int j = 0; j < projections.rows(); ++j)
				{
					if (is_inside(projections.row(j), min_corner, max_corner, 1e-4))
					{
						filtered_coords.row(valid_num) = coords.row(j);
						filtered_grads.row(valid_num) = grads.row(j);
						filtered_udfs(valid_num) = udfs(j);
						++valid_num;
					}
				}

                // gradually enlarge the cell the get adequate samples to solve the QEF
                double cell_offset = 0.02;
                while (valid_num <= 2 && cell_offset < 0.3)
                {
                    valid_num = 0;
                    for (int j = 0; j < projections.rows(); ++j)
                    {
                        if (is_inside(projections.row(j), min_corner, max_corner, cell_offset))
                        {
                            filtered_coords.row(valid_num) = coords.row(j);
                            filtered_grads.row(valid_num) = grads.row(j);
                            filtered_udfs(valid_num) = udfs(j);
                            ++valid_num;
                        }
                    }
                    cell_offset += 0.05;
                }

				if (valid_num > 2)
				{
                    filtered_coords.conservativeResize(valid_num, 3);
                    filtered_grads.conservativeResize(valid_num, 3);
                    filtered_udfs.conservativeResize(valid_num);

					Eigen::VectorXd filtered_b = filtered_coords.cwiseProduct(filtered_grads).rowwise().sum() - filtered_udfs;
					QEF qef2(filtered_grads, filtered_b);
                    DualPoint candidate = solve_point(qef2, min_corner, max_corner, singular_value_threshold, corner_margin, edge_margin, plane_margin);

                    if (candidate.m_is_inside && candidate.m_point.size() > results[i].m_point.size())
                    {
                        results[i] = candidate;
                    }
				}
            }

        }
        if (results[i].m_point.size() > 0)
        {
            results[i].m_point += proj_center;
        }
    }
    for (size_t i = 0; i < m_leaf_nodes.size(); ++i)
    {
        if (results[i].m_point.size() == 3)
        {
            m_leaf_nodes[i].get().m_node_type = "leaf";
            m_leaf_nodes[i].get().m_reconstruct_point = results[i].m_point;
            m_leaf_nodes[i].get().m_vertex_idx = m_mesh_v.size();
            m_mesh_v.emplace_back(results[i].m_point);
            m_v_type.emplace_back(results[i].m_type);
            
            if (results[i].m_dir.size() == 3)
            {
                m_mesh_v_dir.emplace_back(results[i].m_dir);
            }
            else
            {
                m_mesh_v_dir.emplace_back(Eigen::VectorXd::Zero(3));
            }
        }
    }
}


void Octree::generate_mesh()
{
    cellProcContour(m_root_node);
}


void Octree::cellProcContour(OctreeNode& node)
{
    if (node.m_node_type == "interval")
    {
        // 8 Cell calls
        for (int i = 0; i < 8; i++)
        {
            cellProcContour(node.m_children[i]);
        }

        // 12 face calls
        for (int i = 0; i < 12; i++)
        {
            int c[2] = { cellProcFaceMask[i][0], cellProcFaceMask[i][1] };
            std::vector<std::reference_wrapper<OctreeNode> > fcd{std::ref(node.m_children[c[0]]), std::ref(node.m_children[c[1]])};

            faceProcContour(fcd, cellProcFaceMask[i][2]);
        }

        // 6 edge calls

        for (int i = 0; i < 6; i++)
        {
            int c[4] = { cellProcEdgeMask[i][0], cellProcEdgeMask[i][1], cellProcEdgeMask[i][2], cellProcEdgeMask[i][3] };
            std::vector<std::reference_wrapper<OctreeNode> > ecd;
            for (int j = 0; j < 4; j++)
            {
                ecd.emplace_back(std::ref(node.m_children[c[j]]));
            }

            edgeProcContour(ecd, cellProcEdgeMask[i][4]);
        }
    }
}


void Octree::faceProcContour(std::vector<std::reference_wrapper<OctreeNode> >& node, int dir)
{
    if (node[0].get().m_node_type == "empty" || node[1].get().m_node_type == "empty")
    {
        return;
    }
    if (node[0].get().m_node_type == "interval" || node[1].get().m_node_type == "interval")
    {
        // 4 face calls
        for (int i = 0; i < 4; i++)
        {
            int c[2] = { faceProcFaceMask[dir][i][0], faceProcFaceMask[dir][i][1] };
            std::vector<std::reference_wrapper<OctreeNode> >  fcd;
            for (int j = 0; j < 2; j++)
            {
                if (node[j].get().m_node_type == "leaf")
                {
                    fcd.emplace_back(std::ref(node[j]));
                }
                else
                {
                    fcd.emplace_back(std::ref(node[j].get().m_children[c[j]]));
                }
            }
            faceProcContour(fcd, faceProcFaceMask[dir][i][2]);
        }

        // 4 edge calls
        int orders[2][4] = { { 0, 0, 1, 1 }, { 0, 1, 0, 1 } };

        for (int i = 0; i < 4; i++)
        {
            int c[4] = { faceProcEdgeMask[dir][i][1], faceProcEdgeMask[dir][i][2],
                         faceProcEdgeMask[dir][i][3], faceProcEdgeMask[dir][i][4] };
            int* order = orders[faceProcEdgeMask[dir][i][0]];
            std::vector<std::reference_wrapper<OctreeNode> > ecd;
            for (int j = 0; j < 4; j++)
            {
                if (node[order[j]].get().m_node_type == "leaf")
                {
                    ecd.emplace_back(node[order[j]]);
                }
                else
                {
                    ecd.emplace_back(std::ref(node[order[j]].get().m_children[c[j]]));
                }
            }
            edgeProcContour(ecd, faceProcEdgeMask[dir][i][5]);
        }
    }
}



void Octree::edgeProcContour(std::vector<std::reference_wrapper<OctreeNode> >& node, int dir)
{
    for (int i = 0; i < 4; ++i)
    {
        if (node[i].get().m_node_type == "empty")
        {
            return;
        }
    }

    if (node[0].get().m_node_type == "leaf" && node[1].get().m_node_type == "leaf" && node[2].get().m_node_type == "leaf" && node[3].get().m_node_type == "leaf")
    {
        processEdgeWrite(node);
    }
    else
    {
        // 2 edge calls
        for (int i = 0; i < 2; i++)
        {
            int c[4] = { edgeProcEdgeMask[dir][i][0],
                           edgeProcEdgeMask[dir][i][1],
                           edgeProcEdgeMask[dir][i][2],
                           edgeProcEdgeMask[dir][i][3] };
            std::vector<std::reference_wrapper<OctreeNode> >  ecd;
            for (int j = 0; j < 4; j++)
            {
                if (node[j].get().m_node_type == "leaf")
                {
                    ecd.emplace_back(node[j]);
                }
                else
                {
                    ecd.emplace_back(std::ref(node[j].get().m_children[c[j]]));
                }
            }

            edgeProcContour(ecd, edgeProcEdgeMask[dir][i][4]);
        }
    }
}


void Octree::processEdgeWrite(std::vector<std::reference_wrapper<OctreeNode> >& node)
{
    std::vector<int> ind(4);
    for (int i = 0; i < 4; i++)
    {
        ind[i] = node[i].get().m_vertex_idx;
    }

    if (ind[0] == ind[1])
    {
        m_mesh_f.push_back(std::vector<int>({ ind[0], ind[3], ind[2] }));
    }
    else if (ind[1] == ind[3])
    {
        m_mesh_f.push_back(std::vector<int>({ ind[0], ind[1], ind[2] }));
    }
    else if (ind[3] == ind[2])
    {
        m_mesh_f.push_back(std::vector<int>({ ind[0], ind[1], ind[3] }));
    }
    else if (ind[2] == ind[0])
    {
        m_mesh_f.push_back(std::vector<int>({ ind[1], ind[3], ind[2] }));
    }
    else
    {
        m_mesh_f.push_back(std::vector<int>({ ind[0], ind[1], ind[3], ind[2] }));
    }
}


py::tuple Octree::visualize_octree()
{
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::VectorXi> faces;

    std::vector<OctreeNode*> nodes_to_check;
    nodes_to_check.push_back(&m_root_node);

    Eigen::ArrayXXi face_pattern(6, 4);
    face_pattern << 0, 1, 3, 2,
        4, 5, 7, 6,
        0, 4, 5, 1,
        2, 6, 7, 3,
        0, 2, 6, 4,
        1, 3, 7, 5;

    int face_offset = 0;

    while(!nodes_to_check.empty())
    {
        std::vector<OctreeNode*> next_nodes;
	    for(int i = 0; i < nodes_to_check.size(); ++i)
	    {
            Eigen::MatrixXi node_corner_id = (g_child_order * int(std::pow(2, m_max_depth + m_sampling_depth - nodes_to_check[i]->m_depth))).rowwise() + nodes_to_check[i]->m_min_corner_id.transpose();
            Eigen::MatrixXd node_corners = (node_corner_id.cast<double>() * m_unit_edge_length).rowwise() + m_min_corner.transpose();

            for(int j = 0; j < 8; ++j)
            {
                points.push_back(node_corners.row(j));
            }

            for(int j = 0; j < 6; ++j)
            {
                faces.push_back(face_pattern.row(j) + face_offset);
            }
            face_offset += 8;
            for(int j = 0; j < nodes_to_check[i]->m_children.size(); ++j)
            {
                next_nodes.push_back(&nodes_to_check[i]->m_children[j]);
            }
	    }
        nodes_to_check = next_nodes;
    }
    Eigen::MatrixXd point_mat(points.size(), 3);
    Eigen::MatrixXi face_mat(faces.size(), 4);
    for(int i = 0; i < point_mat.rows(); ++i)
    {
        point_mat.row(i) = points[i];
    }
    for (int i = 0; i < face_mat.rows(); ++i)
    {

        face_mat.row(i) = faces[i];
    }
    return py::make_tuple(point_mat, face_mat);
}