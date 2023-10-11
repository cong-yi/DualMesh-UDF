import torch
from DualMeshUDF import extract_mesh

def extract_mesh_from_udf(
        net,
        device,
        batch_size = 150000,
        max_depth=7
):
    """
    Extract the mesh from a UDF network
    Parameters
    ------------
    net : the udf network
    device : the device of the net parameters
    batch_size: batch size for inferring the UDF network
    max_depth: the max depth of the octree, e.g., max_depth=7 stands for resolution of 128^3
    """

    # compose functions
    udf_func = udf_from_net(net, device)
    udf_grad_func = udf_grad_from_net(net, device)

    # get mesh
    mesh_v, mesh_f = extract_mesh(udf_func, udf_grad_func, batch_size, max_depth)

    return mesh_v, mesh_f


def normalize(v, dim=-1):
    norm = torch.linalg.norm(v, axis=dim, keepdims=True)
    norm[norm == 0] = 1
    return v / norm


def udf_from_net(net, device):
    def udf(pts, net=net, device=device):

        net.eval()

        target_shape = list(pts.shape)

        pts = pts.reshape(-1, 3)
        pts = torch.from_numpy(pts).to(device)

        input = pts.reshape(-1, pts.shape[-1]).float()
        udf_p = net(input)
        
        target_shape[-1] = 1
        udf_p = udf_p.reshape(target_shape).detach().cpu().numpy()

        return udf_p

    return udf


def udf_grad_from_net(net, device):

    def grad(pts, net=net, device=device):

        net.eval()
        
        target_shape = list(pts.shape)

        pts = pts.reshape(-1, 3)
        pts = torch.from_numpy(pts).to(device)
        pts.requires_grad = True

        input = pts.reshape(-1, pts.shape[-1]).float()
        udf_p = net(input)

        udf_p.sum().backward()
        grad_p = pts.grad.detach()
        grad_p = normalize(grad_p)

        grad_p = grad_p.reshape(target_shape).detach().cpu().numpy()
        target_shape[-1] = 1
        udf_p = udf_p.reshape(target_shape).detach().cpu().numpy()

        return udf_p, grad_p

    return grad