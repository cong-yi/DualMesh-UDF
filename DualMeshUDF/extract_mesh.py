import numpy as np
from DualMeshUDF_core import OctreeNode, Octree, triangulate_faces
import igl


def extract_mesh(
        udf_func,
        udf_grad_func,
        batch_size = 150000,
        max_depth=7
):
    """
    Extract the mesh from a UDF
    Parameters
    ------------
    udf_func : udf
    udf_grad_func :
    batch_size: batch size for inferring the UDF network
    max_depth: the max depth of the octree, e.g., max_depth=7 stands for resolution of 128^3
    """

    octree = Octree(max_depth=max_depth,
                    min_corner=np.array([[-1.], [-1.], [-1.]]),
                    max_corner=np.array([[1.], [1.], [1.]]),
                    sampling_depth=1)

    cur_depth = 0

    while cur_depth <= max_depth:

        # get centroids of the nodes in the current depth
        centroids_coords = octree.centroids_of_new_nodes().astype(np.float32)

        # query udf values and gradients for the centroids
        centroid_udfs, centroid_grads = query_udf_and_grad(udf_grad_func, centroids_coords, batch_size)

        # adaptively subdivide the cells
        octree.adaptive_subdivide(centroid_udfs, centroid_grads, 0.002)

        cur_depth += 1

    new_grid_indices, new_grid_coords = octree.get_samples_of_new_nodes()

    # query udf values and gradients for the samples
    new_grid_udfs, new_grid_grads = query_udf_and_grad(udf_grad_func, new_grid_coords.astype(np.float32), batch_size)

    octree.set_new_grid_data(new_grid_indices, new_grid_udfs, new_grid_grads)

    # check if projections are reliable
    indices, projections = octree.get_projections_for_checking_validity()
    projection_udfs = query_udf(udf_func, projections, batch_size)
    octree.set_grid_validity(indices, projection_udfs < 0.002)

    # (reliable udf threshold, corner factor, edge factor, face factor, singular value threshold)
    octree.batch_solve(0.002, 1.0, 1.0, 0.15, 0.08)

    octree.generate_mesh()

    tri_faces = triangulate_faces(octree.mesh_v, octree.mesh_f, octree.v_type, octree.mesh_v_dir)

    v, f = igl.remove_duplicates(np.array(octree.mesh_v), tri_faces, 1e-7)

    v, f, _, _ = igl.remove_unreferenced(v, f)

    return v, f


def query_udf(udf_func, coords, max_batch_size=-1):
    '''
    coords should be N*M*3, where N is the batch size
    '''
    input_shape = list(coords.shape)
    query_points = coords.reshape(-1, 3)
    if max_batch_size > 0 and query_points.shape[0] > max_batch_size:
        batch_num = query_points.shape[0] / max_batch_size + 1
        pts_list = np.array_split(query_points, batch_num)
        d = []
        for q_per_batch in pts_list:
            temp_d = udf_func(q_per_batch)
            d.append(temp_d)
        d = np.vstack(d)
    else:
        d = udf_func(query_points)

    input_shape[-1] = 1
    d = d.reshape(input_shape)
    return d


def query_udf_and_grad(udf_grad_func, coords, batch_size=-1):
    '''
    coords should be N*M*3 or N*3, where N is the batch size
    '''
    input_shape = list(coords.shape)
    query_points = coords.reshape(-1, 3)
    if batch_size > 0 and query_points.shape[0] > batch_size:
        batch_num = query_points.shape[0] / batch_size + 1
        pts_list = np.array_split(query_points, batch_num)
        d = []
        g = []
        for q_per_batch in pts_list:
            temp_d, temp_g = udf_grad_func(q_per_batch)
            d.append(temp_d)
            g.append(temp_g)
        d = np.vstack(d)
        g = np.vstack(g)
    else:
        d, g = udf_grad_func(query_points)

    g = g.reshape(input_shape)
    input_shape[-1] = 1
    d = d.reshape(input_shape)
    return d, g

