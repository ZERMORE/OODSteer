"""USteer Eq. 2, 3, 6--8. Analytic gradients; no backward or training."""
import torch
import torch.nn.functional as F


def nci_gradient(x, gain, eps, target):
    # Work in fp32; do NOT normalize the unembedding vector (Eq. 2).
    x, gain, target = x.float(), gain.float(), target.float()
    r = (x.square().mean(-1, keepdim=True) + eps).sqrt()
    z = gain * x / r
    norm2 = z.square().sum(-1, keepdim=True).clamp_min(1e-20)
    dot = (target * z).sum(-1, keepdim=True)
    gz = 2 * dot / norm2 * (target - dot / norm2 * z)
    u = gain * gz
    # Exact RMSNorm Jacobian including epsilon. Eq. 7 omits epsilon and
    # a positive global scalar, canceled by the unit-gradient update.
    gx = u / r - x * (u * x).mean(-1, keepdim=True) / r.pow(3)
    return gx


class USteer:
    def __init__(self, model, layers, alpha, scope="last"):
        self.model, self.layers, self.alpha, self.scope = model, list(layers), alpha, scope
        self.handles = []
        self.calls = 0
        if scope not in ("last", "all"):
            raise ValueError(scope)
        backbone = model.model
        self.norm = backbone.norm
        self.head = model.get_output_embeddings()
        if not hasattr(self.norm, "variance_epsilon"):
            raise ValueError("Only RMSNorm Llama/Qwen backbones supported")
        if any(i < 0 or i >= len(backbone.layers) for i in self.layers):
            raise ValueError("Layer index out of range")

    @torch.no_grad()
    def hook(self, module, args, output):
        if self.alpha == 0:
            return output
        h = output[0] if isinstance(output, tuple) else output
        x = h[:, -1:, :] if self.scope == "last" else h
        xf = x.float()
        gain = self.norm.weight.float()
        eps = self.norm.variance_epsilon
        z = gain * xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        # Vocabulary projection in native precision to avoid fp32 V x H copy.
        logits = F.linear(z.to(self.head.weight.dtype), self.head.weight, self.head.bias)
        target = self.head.weight[logits.argmax(-1)].float()
        grad = nci_gradient(xf, gain, eps, target)
        updated = (xf + self.alpha * F.normalize(grad, dim=-1, eps=1e-20)).to(h.dtype)
        result = h.clone()
        if self.scope == "last":
            result[:, -1:, :] = updated
        else:
            result = updated
        self.calls += 1
        return (result,) + output[1:] if isinstance(output, tuple) else result

    def __enter__(self):
        for i in self.layers:
            self.handles.append(self.model.model.layers[i].register_forward_hook(self.hook))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def intermediate_layers(n, depth):
    if not 1 <= depth < n:
        raise ValueError("depth must be between 1 and n_layers-1")
    # Exclude the final decoder block. See README layer convention.
    return list(range(n - 1 - depth, n - 1))
