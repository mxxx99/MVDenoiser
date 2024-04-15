import torch

def warp2center_5frame(x,flow):
    out=torch.empty(x.shape, device=x.device)
    out[:,:,2]=x[:,:,2]
    out[:,:,0],_=warp(x[:,:,0],flow['forward'][:,0])#T=0向T=2对齐
    out[:,:,1],_=warp(x[:,:,1],flow['forward'][:,1])#T=1向T=2对齐
    out[:,:,3],_=warp(x[:,:,3],flow['backward'][:,1])#T=3向T=2对齐
    out[:,:,4],_=warp(x[:,:,4],flow['backward'][:,0])#T=4向T=2对齐
    return out

def warp(x, flo):
    '''
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] (flow)
    '''
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones((B, C, H, W), device=x.device))
    mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask, mask


# simply convert raw seq to rgb seq for computing optical flow
def demosaic(raw_seq):
    N, T, C, H, W = raw_seq.shape
    rgb_seq = torch.empty((N, T, 3, H, W), dtype=raw_seq.dtype, device=raw_seq.device)
    rgb_seq[:, :, 0] = raw_seq[:, :, 0]
    rgb_seq[:, :, 1] = (raw_seq[:, :, 1] + raw_seq[:, :, 2]) / 2
    rgb_seq[:, :, 2] = (raw_seq[:, :, 3])
    return rgb_seq