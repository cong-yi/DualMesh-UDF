#pragma once
#include <iostream>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <vector>


class OctreeNode
{
public:
    // the parent node of the current node
    OctreeNode* m_parent;

    // the children nodes of the current node
    std::vector<OctreeNode> m_children;

    // the current depth
    int m_depth;

    // the grid id of the min and the max corners
    Eigen::VectorXi m_min_corner_id;
    Eigen::VectorXi m_max_corner_id;

    // the grid id of the centroid point
    Eigen::VectorXi m_centroid_id;
    
    // grid ids of the sampling data
    Eigen::MatrixXi m_sample_grids;

    // the edge length of the current node
    int m_edge_span;

    // vertex id in the reconstructed mesh
    int m_vertex_idx;

    // node type
    std::string m_node_type;

    // reconstruction point
    Eigen::VectorXd m_reconstruct_point;

    OctreeNode(Eigen::VectorXi min_corner_id, Eigen::VectorXi max_corner_id, OctreeNode* parent = nullptr, int depth = 0);
    OctreeNode(){}
    ~OctreeNode(){}

    // subdive the current node
    bool subdivide();

    inline Eigen::MatrixXi generate_sample_grids(const Eigen::MatrixXi& sampling_pattern);
};


class Octree
{
public:
    // mesh vertices
    std::vector<Eigen::VectorXd> m_mesh_v;

    // mesh faces (quad)
    std::vector<std::vector<int> > m_mesh_f;

    // the point types of mesh vertices
    std::vector<int> m_v_type;

    // the direction of mesh vertices
    std::vector<Eigen::VectorXd> m_mesh_v_dir;

    Octree(int max_depth, Eigen::VectorXd min_corner = Eigen::VectorXd::Constant(3, -1.), Eigen::VectorXd max_corner = Eigen::VectorXd::Constant(3, 1.), int sampling_depth=1);
    ~Octree(){}

    // get new samples
    pybind11::tuple get_samples_of_new_nodes();

    // get the number of the leaf nodes
    int get_num_of_leaf_nodes() { return m_leaf_nodes.size(); }

    // solve all QEFs
    void batch_solve(double sample_udf_threshold = 0.002, double corner_margin = 0.5, double edge_margin = 0.1, double plane_margin = 0.1, double singular_value_threshold = 0.01);

    // generate quad mesh
    void generate_mesh();
    
	// visualize the octree
    pybind11::tuple visualize_octree();

    // return centroid coordinates for query UDF
    Eigen::MatrixXd get_centroids_of_new_nodes();

    // subdivide the current leaf nodes according to the UDF values of their centroids
    void adaptive_subdivide(const Eigen::VectorXf& centroid_udfs, const Eigen::MatrixXf& centroid_grads, double margin);

    // compute the projections by the UDF and gradient of the samples
    pybind11::tuple get_projections_for_checking_validity();

    // set udf and grad data for new grids
    void set_new_grid_data(const Eigen::VectorXi& new_grid_indices, const Eigen::VectorXf& new_grid_udfs, const Eigen::MatrixXf& new_grid_grads);

    // set the grid validity according to the mask
    void set_grid_validity(const Eigen::VectorXi& indices, const std::vector<bool>& validity_mask);
    

private:
    // the root node
    OctreeNode m_root_node;

    // the max depth of the octree
    int m_max_depth;

    // the subdivision depth for sampling in a cell
    int m_sampling_depth;

    // the sampling resolution on each edge
    int m_capacity_per_dim;

    // the min corner
    Eigen::VectorXd m_min_corner;

    // the max corner
    Eigen::VectorXd m_max_corner;

    // the sampling pattern
    Eigen::MatrixXi m_sampling_pattern;

    // the unit edge length
    double m_unit_edge_length;

    // the current leaf nodes
    std::vector<std::reference_wrapper<OctreeNode> > m_leaf_nodes;

    // the existence of grids, indexed by the global id
    std::vector<bool> m_grid_existence;

    // the map from global id to sampling id
    std::vector<int> m_gid2sid;

    // the coordinate, udf, and gradient values of centroids
    std::vector<Eigen::Vector3d>    m_centroid_coords;
    std::vector<double>             m_centroid_udfs;
    std::vector<Eigen::Vector3d>    m_centroid_grads;

    // the coordinate, udf, and gradient values of all sample points
    Eigen::MatrixXd m_sample_coords;
    Eigen::MatrixXd m_sample_grads;
    Eigen::VectorXd m_sample_udfs;

    // validity of points, indexed by the global id
    std::vector<bool> m_proj_validity;

    // sampling ids of the projections
    std::vector<int> m_proj_sid;

    // global ids of the projections
    std::vector<int> m_proj_gid;

    // mesh generation
    void cellProcContour(OctreeNode& node);
    void faceProcContour ( std::vector<std::reference_wrapper<OctreeNode> >& node, int dir );
    void edgeProcContour ( std::vector<std::reference_wrapper<OctreeNode> >& node, int dir );
    void processEdgeWrite ( std::vector<std::reference_wrapper<OctreeNode> >& node );

    // convert a 3D grid tuple to a global index
    int encode(const Eigen::VectorXi& grid);

    // convert a global index to a 3D grid tuple
    Eigen::VectorXi decode(int gid);

    // pre-compute the sampling pattern
    Eigen::MatrixXi generate_sample_pattern(int per_edge_sample_num);

    // compute the coordinate of a grid
    Eigen::VectorXd convert_grid_to_coords(const Eigen::VectorXi& grid);
};