import clip
import torch


from ds import prepare_coco_dataloaders, prepare_flickr_dataloaders, prepare_cub_dataloaders, prepare_flo_dataloaders

def load_model(device, model_path=None):
    # load zero-shot CLIP model
    model, _ = clip.load(name='ViT-B/32',
                         device=device,
                         loss_type='contrastive')
    if model_path is None:
        # Convert the dtype of parameters from float16 to float32
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    else:
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)
    return model

def load_vlm(device, model_name='CLIP', backbone = 'ViT-B/32', model_path=None):
    # load zero-shot CLIP model

    if model_name == 'CLIP':
        model, _ = clip.load(name=backbone,
                         device=device,
                         )
    else:
        model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
    if model_path is None:
        # Convert the dtype of parameters from float16 to float32
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    else:
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)
    return model

def load_data_loader(dataset_name, data_dir):
    prepare_loaders = {
        'coco': prepare_coco_dataloaders,
        'flickr': prepare_flickr_dataloaders,
        'CUB':prepare_cub_dataloaders,
        'FLO':prepare_flo_dataloaders
    }[dataset_name]
    if dataset_name == 'CUB':
        loaders = prepare_loaders(
            dataset_root=data_dir,
            caption_root=data_dir+'/text/text',
            )     
    elif dataset_name == 'FLO':
        loaders = prepare_loaders(
            dataset_root=data_dir,
            caption_root=data_dir+'/text_c10',)
    else:
        loaders = prepare_loaders(
            dataset_root=data_dir,
        )
    return loaders

'''
def kl_divergence_gaussians(mu_q, var_q, mu_p, var_p):
    """
    Compute KL divergence between two multivariate Gaussians with variances as vectors.

    Args:
        mu_q (torch.Tensor): Mean of the first Gaussian, shape (batch_size, d)
        var_q (torch.Tensor): Variance vector of the first Gaussian, shape (batch_size, d)
        mu_p (torch.Tensor): Mean of the second Gaussian, shape (batch_size, d)
        var_p (torch.Tensor): Variance vector of the second Gaussian, shape (batch_size, d)

    Returns:
        torch.Tensor: KL divergence for each pair of Gaussians in the batch, shape (batch_size,)
    """

    # Ensure var_q and var_p are variances (as vectors)
    # var_q and var_p are expected to be (batch_size, d)
    
    # Compute the dimension of the Gaussians
    k = mu_q.shape[1]

    # Compute log(det(sigma_p) / det(sigma_q)) = log(sigma_p) - log(sigma_q) for variances (vectors)
    log_det_ratio = torch.sum(torch.log(var_p) - torch.log(var_q), dim=1)

    # Compute trace of (Sigma_p^-1 Sigma_q) = sum(var_q / var_p) for variances (vectors)
    trace_term = torch.sum(var_q / var_p, dim=1)

    # Compute quadratic term (mu_p - mu_q)^T Sigma_p^-1 (mu_p - mu_q) for variances (vectors)
    diff = mu_p - mu_q
    quadratic_term = torch.sum((diff ** 2) / var_p, dim=1)

    # KL divergence for each Gaussian in the batch
    kl_div = 0.5 * (trace_term + quadratic_term - k + log_det_ratio)
    
    return kl_div
'''

def kl_divergence_gaussians(mu1, var1, mu2, var2):
    """
    Compute KL divergence between two diagonal Gaussian distributions.
    
    Args:
    - mu1: Mean of the first Gaussian (batch_size x d)
    - logvar1: Log variance of the first Gaussian (batch_size x d)
    - mu2: Mean of the second Gaussian (batch_size x d)
    - logvar2: Log variance of the second Gaussian (batch_size x d)
    
    Returns:
    - kl_div: KL divergence for each distribution in the batch (batch_size x d)
    """
    # Variance from log variance (since input is in log variance form)
    #var1 = torch.exp(logvar1)  # sigma1^2
    #var2 = torch.exp(logvar2)  # sigma2^2
    
    # Calculate each term in the KL divergence formula
    term1 = torch.log(var2) - torch.log(var1)  # log(σ2) - log(σ1)
    term2 = (var1 + (mu1 - mu2) ** 2) / (2 * var2)  # (σ1^2 + (μ1 - μ2)^2) / (2 * σ2^2)
    term3 = -0.5  # The constant term (-1/2)
    
    # Combine the terms
    kl_div = term1 + term2 + term3
    kl_div_mean = kl_div.mean(dim=-1)
    
    return kl_div_mean