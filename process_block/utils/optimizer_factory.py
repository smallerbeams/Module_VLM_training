import torch

def minigptv2_optimizer(lr_conf=dict, model=None):
    # TODO make optimizer class and configurations
    if model != None:
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        print("number of trainable parameters: %d" % num_parameters)
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": lr_conf['weight_decay'],
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
        beta2 = lr_conf['beta2']
        _optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(lr_conf['init_lr']),
            weight_decay=lr_conf['weight_decay'],
            betas=(0.9, beta2),
        )

        return _optimizer