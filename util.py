import numpy as np
import torch
import matplotlib.pyplot as plt
def p(im):
    plt.imshow(im.detach().cpu(), cmap="Greys_r")
    #plt.colorbar()
def sampleLatent(param, c, stat=False):
    loss = torch.tensor(0.0, device=c.device)
    latent = []
    RMS_miu = []
    mean_var = []
    for i in range(c.depth):
        mean = param[i][:, : c.NLatent[i]]
        log_var = param[i][:, c.NLatent[i] :].mean(3).mean(2).unsqueeze(2).unsqueeze(3)
        var = torch.exp(log_var)
        latent.append(torch.normal(mean, torch.sqrt(var)))
        if param[i].nelement() != 0:
            loss += ((mean * mean).mean() + var.mean() - log_var.mean() - 1.0) / 2
        RMS_miu.append((mean * mean).mean(1).sqrt().mean().item())
        mean_var.append(var.mean().item())
    if stat:
        return latent, loss, RMS_miu, mean_var
    else:
        return latent, loss


def test(c, valset, structuralEncoder, latentEncoder, decoder,VGGLoss):
    structuralEncoder.eval()
    latentEncoder.eval()
    decoder.eval()
    result = {}
    RMS_miu = np.zeros(c.depth)
    mean_var = np.zeros(c.depth)
    n = 0
    L1 = 0
    L2 = 0
    VGG=0
    loader = torch.utils.data.DataLoader(valset, batch_size=c.batchSize)
    for _, data in enumerate(loader):
        n += 1
        X = data["X"].to(c.device)
        Y = data["Y"].to(c.device)

        structure = structuralEncoder(X)
        leout = latentEncoder(torch.cat([X, Y], dim=1))
        latent, latentLoss_, RMS_miu_, mean_var_ = sampleLatent(
            latentEncoder(torch.cat([X, Y], dim=1)), c, True
        )
        Y_pred = decoder(structure, latent)+X
        RMS_miu += np.array(RMS_miu_)
        mean_var += np.array(mean_var_)
        L1 += (Y - Y_pred).abs().mean().item()
        L2 += (Y - Y_pred).pow(2).mean().item()
        VGG+=VGGLoss(Y,Y_pred).mean().item()
    result.update(
        {
            "L1": L1 / n * 1000,
            "L2": (L2 / n)**0.5 * 1000,
            "VGGLoss":VGG,
            "RMS_miu": RMS_miu / n,
            "mean_var": mean_var / n,
            "sample": torch.cat((X, Y_pred, Y), dim=3),
        }
    )
    return result
import importlib
reload=importlib.reload

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)