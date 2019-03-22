import torch 

def depth_to_space(x, r):   
    n, c, h, w = x.size() 
    x = x.view(n, r, r, c//r**2, h, w).permute(0, 3, 4, 1, 5, 2).reshape(n, c//r**2, r*h, r*w) 
    return x 

@torch._jit_internal.weak_script 
def normalize(input, p=2, dim=1, eps=1e-12, out=None): 
    if out is None: 
        denom = input.norm(p, dim, True).expand_as(input) 
        ret = input / (denom + eps) 
    else: 
        denom = input.norm(p, dim, True).expand_as(input) 
        ret = torch.div(input, denom+eps, out=torch.jit._unwrap_optional(out)) 
    return ret 












