import torch


def normalize(v, dim=-1):
    norm = torch.linalg.norm(v, axis=dim, keepdims=True)
    norm[norm == 0] = 1
    return v / norm


def udf_from_mlp(net, device):
    def udf(pts, net=net, device=device):

        net.eval()

        target_shape = list(pts.shape)

        pts = pts.reshape(-1, 3)
        pts = torch.from_numpy(pts).to(device)
        pts.requires_grad = True

        input = pts.reshape(-1, pts.shape[-1]).float()
        udf_p = net(input)
        
        target_shape[-1] = 1
        udf_p = udf_p.reshape(target_shape).detach().cpu().numpy()

        return udf_p

    return udf


def udf_and_grad_from_mlp(net, device):

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