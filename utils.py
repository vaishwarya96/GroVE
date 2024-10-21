import clip
import torch
from lavis.models import load_model_and_preprocess
from gpytorch.mlls import VariationalELBO
from torch import optim

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

def kl_divergence_gaussians(mu1, var1, mu2, var2):
    """
    Compute KL divergence between two diagonal Gaussian distributions.
    
    Args:
    - mu1: Mean of the first Gaussian (batch_size x d)
    - var1: variance of the first Gaussian (batch_size x d)
    - mu2: Mean of the second Gaussian (batch_size x d)
    - var2: variance of the second Gaussian (batch_size x d)
    
    Returns:
    - kl_div: KL divergence 
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
'''

def kl_divergence_gaussians(mu0, var0, mu1, var1):
    """
    KL divergence between two diagonal multivariate Gaussians.
    :param mu0: Mean of first Gaussian (batch_size, dim)
    :param var0: Variance of first Gaussian (batch_size, dim)
    :param mu1: Mean of second Gaussian (batch_size, dim)
    :param var1: Variance of second Gaussian (batch_size, dim)
    :return: KL divergence (batch_size,)
    """
    # Compute the log variance ratios and variances
    log_var_ratio = torch.log(var1) - torch.log(var0)
    var_ratio = var0 / var1
    
    # Mahalanobis distance term between the means
    mean_diff = mu1 - mu0
    mahalanobis_term = (mean_diff ** 2) / var1
    
    # Compute the KL divergence for each batch element
    kl_div = 0.5 * (torch.sum(log_var_ratio + var_ratio + mahalanobis_term - 1, dim=1))
    #print(kl_div.shape)

    return kl_div


# Function to infer Z given new observed X (text features)
def infer_prob_embeddings(model, likelihood, latent_dim, Z_new, num_epochs=20, lr=0.00001):
    model.eval()
    likelihood.eval()
    
    # Initialize new latent variables Z_new (requires_grad=True to optimize them)
    X_new = torch.randn(Z_new.size(0), latent_dim, requires_grad=True, device="cuda")
    
    # Optimizer for Z_new
    optimizer = optim.Adam([X_new], lr=lr)

    mll_infer = VariationalELBO(likelihood, model, num_data=X_new.size(0))

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass through the model to get p(Z_new | X_new)
        output = model(X_new)
        
        # Compute the ELBO loss to optimise the latent points
        loss_infer = -mll_infer(output, Z_new)
        loss_infer.backward(retain_graph=True)

        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss_infer.item()}')

    with torch.no_grad():
        prediction = likelihood(model(X_new))
        X_mean = prediction.mean
        X_variance = prediction.variance

    return X_mean, X_variance  # Return the mean and variance of the probabilisitc embedding

def get_recall(pred_ranks_all, recall_ks=(1,), q_classes_all=None, g_classes_all=None):
    recall_scores = []
    for recall_k in recall_ks:
        corr=0
        total = len(pred_ranks_all)
        for i in range(len(pred_ranks_all)):
            gt_class = q_classes_all[i]
            pred_classes = [g_classes_all[j] for j in pred_ranks_all[i][:recall_k]]
            if gt_class in pred_classes:
                corr+=1
        recall_scores.append(corr/total)

    return recall_scores

def geo_mean(tensor):
    log_tensor = torch.log(tensor)
    mean_log = torch.mean(log_tensor, dim=1)
    geometric_mean = torch.exp(mean_log)
    return geometric_mean

def get_expected_calibration_error_retrieval(pred_ranks_all, q_classes_all, g_classes_all, embedding_variances, bins):

    total = len(pred_ranks_all)
    correct = np.zeros((total,1))
    for i in range(len(pred_ranks_all)):
        gt_class = q_classes_all[i]
        pred_classes = [g_classes_all[j] for j in pred_ranks_all[i][:1]]
        if gt_class in pred_classes:
            correct[i] = 1
    
    unc = torch.cat(embedding_variances).cpu() 
    unc = geo_mean(unc)
    
    x, y = [], []
    for i in range(num_bins):
        bin_data = bins['bin{}'.format(i)]
        if len(bin_data) == 0:
            continue

        mask = [t[0] for t in bin_data]
        x.append(np.sum(correct[mask])/len(correct[mask]))
        y.append(i)
    
    print("Spearmann coefficient: ", spearmanr(y, x))
    
    #r2 score
    x = np.array(x) 
    y = np.array(y)

    uncertainty_levels_reshaped = y.reshape(-1, 1)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(uncertainty_levels_reshaped, x)

    # Predict R@1 performances
    predicted_r_at_1 = model.predict(uncertainty_levels_reshaped)

    # Calculate the R^2 score
    r2 = r2_score(x, predicted_r_at_1)
    print(f"R^2 score: {r2}")