#include "octree.h"
#include "mesh_utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

 PYBIND11_MODULE(DualMeshUDF_core, m)
 {
     m.doc() = "DualMeshUDF_core";

     py::class_<OctreeNode>(m, "OctreeNode")
         .def(py::init<Eigen::VectorXi, Eigen::VectorXi>())
         .def(py::init<Eigen::VectorXi, Eigen::VectorXi, OctreeNode*, int>())
         .def("subdivide", &OctreeNode::subdivide, py::return_value_policy::reference);

     py::class_<Octree>(m, "Octree")
         .def(py::init<int, Eigen::VectorXd, Eigen::VectorXd, int>(), py::arg("max_depth") = 7, py::arg("min_corner") = Eigen::VectorXd::Constant(3, -1.), py::arg("max_corner") = Eigen::VectorXd::Constant(3, 1.), py::arg("sampling_depth") = 1)
         .def("batch_solve", &Octree::batch_solve, py::return_value_policy::copy)
         .def("centroids_of_new_nodes", &Octree::get_centroids_of_new_nodes)
         .def("adaptive_subdivide", &Octree::adaptive_subdivide)
         .def("set_new_grid_data", &Octree::set_new_grid_data)
         .def("get_projections_for_checking_validity", &Octree::get_projections_for_checking_validity)
         .def("get_samples_of_new_nodes", &Octree::get_samples_of_new_nodes)
         .def("set_grid_validity", &Octree::set_grid_validity)
         .def("visualize_octree", &Octree::visualize_octree)
         .def("get_num_of_leaf_nodes", &Octree::get_num_of_leaf_nodes)
         .def("generate_mesh", &Octree::generate_mesh)
         .def_readwrite("mesh_v", &Octree::m_mesh_v)
         .def_readwrite("mesh_f", &Octree::m_mesh_f)
         .def_readwrite("v_type", &Octree::m_v_type)
         .def_readwrite("mesh_v_dir", &Octree::m_mesh_v_dir);

     m.def("triangulate_faces", &triangulate_faces);
     m.def("write_obj", &write_obj);
 }
