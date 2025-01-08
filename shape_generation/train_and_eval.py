import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

import math
from itertools import combinations_with_replacement

from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics, EMD_CD
from tqdm import tqdm
import argparse
from torch.distributions import Normal
import pandas as pd

from utils.file_utils import *
from utils.visualize import *
import torch.distributed as dist
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from torchvision.utils import save_image
from copy import deepcopy
from collections import OrderedDict


from utils.misc import Evaluator

from tensorboardX import SummaryWriter
from socket import gethostname
from pytorch3d.ops import sample_farthest_points, knn_points


'''
some utils
'''
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        # # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        # if name.startswith('model.module'):
        #     name = name.replace('model.module.', 'model.')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate(vertices, faces):
    '''
    vertices: [numpoints, 3]
    '''
    M = rotation_matrix([0, 1, 0], np.pi / 2).transpose()
    N = rotation_matrix([1, 0, 0], -np.pi / 4).transpose()
    K = rotation_matrix([0, 0, 1], np.pi).transpose()

    v, f = vertices[:,[1,2,0]].dot(M).dot(N).dot(K), faces[:,[1,2,0]]
    return v, f

def norm(v, f):
    v = (v - v.min())/(v.max() - v.min()) - 0.5

    return v, f

def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
    return pNorm, gradNorm


def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)
        
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


'''
Moment GPT
'''
def generate_exponents(d, degree):
    """
    Generate all multi-indices with total degree up to 'degree' for d-dimensional points.
    
    Parameters:
        d (int): The dimension of the points.
        degree (int): The maximum degree of the monomials.
    
    Returns:
        ndarray: The multi-indices of shape (num_poly, d).
    """
    num_poly = math.comb(degree + d, d)
    exponents = torch.zeros(num_poly, d, dtype=int)
    i = 0
    for total_degree in range(degree + 1):
        for exps in combinations_with_replacement(range(d), total_degree):
            for var in exps:
                exponents[i, var] += 1
            i += 1
            
    return exponents[1:]

def generate_monomials_sequences_batch(X, exponents):
    """
    Generate monomials given a point cloud and multi-indices.

    Parameters:
        X (ndarray): An array of shape (B, N, d) representing the point cloud.
        exponents (ndarray): The multi-indices of shape (M, d).

    Returns:
        ndarray: Monomial sequences of shape (B, M).
    """
    B, N, d = X.shape
    device = X.device
    exponents = exponents.to(device)
    M = len(exponents)
    # print(f'Number of monomials: {M}') # Number of polynomials: n1 + n2 + ... + n_d = degree; degree + d choose d; d number of dividers for an array in space R^d.
    # monomials = torch.ones(B, N, M, device=device)
    # for i, exp in enumerate(exponents):
    #     monomials[:, :, i] = torch.prod(X ** exp, axis=2) # x1^exp1 * x2^exp2 * ... * xd^expd. e.g. x1^2 * x2^3 * x3^1 \in R^3
    monomials = X.unsqueeze(2).repeat(1, 1, M, 1) ** exponents.unsqueeze(0).unsqueeze(0) # (B, N, M, d) ** (1, 1, M, d) -> (B, N, M, d)
    monomials = monomials.prod(dim=-1) # (B, N, M)
    return monomials.sum(dim=1) / N # (B, N, M) -> (B, M)

def generate_chebyshev_polynomials_sequence_batch(X, exponents):
    """
    Generate Chebyshev polynomials given a point cloud and multi-indices.

    Parameters:
        X (ndarray): An array of shape (B, N, d) representing the d-dimensional point cloud.
        exponents (ndarray): The multi-indices of shape (M, d).

    Returns:
        ndarray: Chebyshev polynomial sequences of shape (B, M).
    """
    B, N, d = X.shape
    device = X.device
    exponents = exponents.to(device)
    cheby_polynomials = torch.cos(exponents.unsqueeze(0).unsqueeze(0) * torch.acos(X).unsqueeze(2)) # (B, N, M)
    cheby_polynomials = cheby_polynomials.prod(dim=-1) # (B, N)
    
    return cheby_polynomials.sum(dim=1) / N # (B, N, M) -> (B, M)

def poly_seq_batch(X, exponents, poly_type='monomial'):
    if poly_type == 'monomial':
        return generate_monomials_sequences_batch(X, exponents)
    elif poly_type == 'chebyshev':
        return generate_chebyshev_polynomials_sequence_batch(X, exponents)
    else:
        raise ValueError('Unknown polynomial type')


class Model(nn.Module):
    def __init__(self, args, poly_type='chebyshev', poly_degree=15):
        super(Model, self).__init__()
        self.self.exponts = generate_exponents(args.nc, poly_degree)
        self.model = DiT_models[args.model_type](input_size=args.n_c,
                                                 num_classes=args.num_classes)
    
    def get_loss_iter(self, data, noises=None, y=None):
        # data [B, N_C, N_P, 3]
        
        B, N_C, N_P, D = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises, y=y)
        assert losses.shape == t.shape == torch.Size([B])
        return losses, torch.mean(t.float())

    def gen_samples(self, shape, device, y, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False,
                    cfg=1.0):
        assert cfg >= 1.0
        if cfg != 1.0:
            denoise_fn_kwargs = {'cfg': cfg}
            return self.diffusion.p_sample_loop(self._denoise_with_cfg, shape=shape, device=device, y=y, noise_fn=noise_fn, 
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running, denoise_fn_kwargs=denoise_fn_kwargs)
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, y=y, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    def gen_sample_traj(self, shape, device, y, freq, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False, cfg=1.0):
        assert cfg >= 1.0
        if cfg != 1.0:
            denoise_fn_kwargs = {'cfg': cfg}
            return self.diffusion.p_sample_loop_trajectory(self._denoise_with_cfg, shape=shape, device=device, y=y, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running, denoise_fn_kwargs=denoise_fn_kwargs)
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, y=y, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)
        
    def parameters(self, recurse = True):
        return self.model.parameters(recurse)
    
    def state_dict(self, destination = None, prefix = '', keep_vars = False):
        return self.model.state_dict(destination, prefix, keep_vars)
    
    def load_state_dict(self, state_dict, strict = True):
        # only load the state dict of the model
        self.model.load_state_dict(state_dict, strict)


def get_dataset(dataroot, npoints,category):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=category.split(','), split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True)
    te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=category.split(','), split='val',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )
    return tr_dataset, te_dataset

def main():
    opt = parse_args()

    # ====== setup logging ======
    # current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    # output_dir = os.path.join(opt.model_dir, opt.experiment_name, current_time)
    ## current time could be different when running on multiple nodes
    output_dir = os.path.join(opt.model_dir, opt.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    copy_source(__file__, output_dir)
    
    logger = setup_logging(output_dir)

    if not opt.debug:
        if opt.use_tb:
            # tb writers
            tb_writer = SummaryWriter(output_dir)
    # ====== setup logging ======
    
    # ====== seed ======
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.manualSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    # ====== seed ======
    
    # ====== dataset ======
    train_dataset, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category)
    if opt.fps_points > 0:
        train_dataset.cache_fps_points(opt.fps_points)
        test_dataset.cache_fps_points(opt.fps_points)
    # ====== dataset ======
    
    should_diag = True
    
    # ====== model ======
    model = Model(opt, poly_type=opt.poly_type, poly_degree=opt.poly_degree)
    
    # ====== model ======

    # ================ check if distributed training ================
    try:
        opt.distributed = int(os.environ["WORLD_SIZE"]) > 1
    except:
        opt.distributed = False
    # ================ check if distributed training ================

    if opt.distributed:
        # ================ distributed training ================
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["SLURM_PROCID"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert gpus_per_node == torch.cuda.device_count()
        print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
            f" {gpus_per_node} allocated GPUs per node.", flush=True)
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(local_rank)
        print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank
            )
            
        should_diag = rank==0
        device = local_rank
        
        opt.bs = int(opt.bs / gpus_per_node)
        
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[device])
        model = model.to(device)
        model.multi_gpu_wrapper(_transform_)
        
        # Note that parameter initialization is done within the DiT constructor
        if opt.use_ema:
            ema = deepcopy(model.model).to(device)  # Create an EMA of the model for use after training
            requires_grad(ema, False)
        
        # ================ distributed training ================
    else:
        # ================ single GPU training ================
        train_sampler = None
        test_sampler = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        if opt.use_ema:
            ema = deepcopy(model.model).to(device)
            requires_grad(ema, False)
        # ================ single GPU training ================
        
    # ================ dataloaders ================
    opt.workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=opt.bs,
                                                   sampler=train_sampler,
                                                   shuffle=train_sampler is None, 
                                                   num_workers=opt.workers, 
                                                   drop_last=True)
    # print('train_dataloader size:', len(train_dataloader))

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                      batch_size=opt.bs,
                                                      sampler=test_sampler,
                                                      shuffle=False, 
                                                      num_workers=opt.workers, 
                                                      drop_last=False)
        # print('test_dataloader size:', len(test_dataloader))    
    else:
        test_dataloader = None
    # ================ dataloaders ================
    
    # ================ general model logging ================    
    if should_diag:
        
        logger.info(f"Random Seed: {opt.manualSeed}")
        logger.info(f'train_dataset size: {len(train_dataset)}')
        logger.info(f'test_dataset size: {len(test_dataset)}')
        logger.info(opt)
        
        logger.info("Model = %s" % str(model))
        total_params = sum(param.numel() for param in model.parameters())/1e6
        logger.info("Total_params = %s MB " % str(total_params))  
        
         
    # ================ general model logging ================
    if not opt.evaluate:
        train(opt, output_dir, logger, tb_writer, should_diag, model, train_sampler, ema, device, train_dataloader)
        opt.model = os.path.join(output_dir, 'checkpoint.pth')
        
    test(opt, output_dir, logger, should_diag, model, device, test_dataloader)
    if opt.distributed:
        dist.barrier()
        dist.destroy_process_group()

def test(opt, output_dir, logger, should_diag, model, device, test_dataloader):
    # logger.info('='*20)
    # logger.info('Start evaluation')
    # logger.info('='*20)
    
    model.eval()
    
    evaluator = Evaluator(results_dir=output_dir)
    
    assert opt.model != ''
    checkpoint = torch.load(opt.model)
    model_weights = checkpoint['ema' if opt.use_ema else 'model_state']
    # checkpoint_dict = {k.replace('model.', 'module.' if opt.distributed else ''): model_weights[k] for k in model_weights if k.startswith('model.')}
    checkpoint_dict = {k.replace('module.', ''): model_weights[k] for k in model_weights} if not opt.distributed else model_weights
    # checkpoint_dict = {'module.' + k: model_weights[k] for k in model_weights} if opt.distributed else model_weights
    model.load_state_dict(checkpoint_dict)
        
    for run_i in range(opt.nrepeats):
        outf_syn = os.path.join(output_dir, f'run_{run_i}', 'syn')
        os.makedirs(outf_syn, exist_ok=True)
        opt.eval_path = os.path.join(outf_syn, 'samples.pth')
        Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
        # stats = generate_eval(model, opt, device, outf_syn, evaluator)
        
        def new_y_chain(device, num_chain, num_classes):
            return torch.randint(low=0,high=num_classes,size=(num_chain,),device=device)
        
        with torch.no_grad():
            samples = []
            reference = []
            for i, data in enumerate(test_dataloader):
                if should_diag:
                    logger.info('Start generating: (%d/%d)' % (i, len(test_dataloader)))
                x = data['train_points']
                m, s = data['mean'].float(), data['std'].float()
                y = data['cate_idx']
                
                gen = model.gen_samples((x.shape[0], opt.n_c, opt.n_p, 3), 
                                        device, new_y_chain(device,y.shape[0],opt.num_classes), clip_denoised=False,
                                        cfg=opt.cfg_scale)
                if opt.eval_fps_points > 0:
                    gen = sample_farthest_points(gen.flatten(1,2).contiguous(), K=opt.eval_fps_points)[0]
                else:
                    gen = gen.flatten(1,2)
                    gen = gen[:, torch.randperm(gen.shape[1])[:opt.eval_npoints], :]
                gen = gen.detach().cpu()
                
                gen = gen * s + m
                x = x * s + m
                
                samples.append(gen.to(device).contiguous())
                reference.append(x.to(device).contiguous())
                
                if should_diag:
                    logger.info('Start saving: (%d/%d)' % (i, len(test_dataloader)))
                ep_save_dir = os.path.join(outf_syn, 'batch_%03d' % i)
                os.makedirs(ep_save_dir, exist_ok=True)
                for idx in range(gen.shape[0]):
                    pts = gen[idx].cpu().numpy()
                    np.savetxt(os.path.join(ep_save_dir, 'sample_%d.xyz' % idx), pts)
                    ref_pts = x[idx].cpu().numpy()
                    np.savetxt(os.path.join(ep_save_dir, 'reference_%d.xyz' % idx), ref_pts)
                    concat_pts = np.concatenate([x[idx].cpu().numpy(), pts], axis=0)
                    np.savetxt(os.path.join(ep_save_dir, 'concat_%d.xyz' % idx), concat_pts)
                
                if should_diag:
                    logger.info('Finish saving: (%d/%d)' % (i, len(test_dataloader)))
                
        samples = torch.cat(samples, dim=0)
        reference = torch.cat(reference, dim=0)
        
        if opt.distributed:
            samples_gather = concat_all_gather(samples)
            reference_gather = concat_all_gather(reference)
            samples = samples_gather
            reference = reference_gather
        torch.save(samples, opt.eval_path)
        
        logger.info('Computing metrics...')
        logger.info('Samples shape: %s' % str(samples.shape))
        logger.info('Reference shape: %s' % str(reference.shape))
        
        results = compute_all_metrics(samples, reference, opt.bs)
        results = {k: (v.cpu().detach().item()
                    if not isinstance(v, float) else v) for k, v in results.items()}
        
        jsd = JSD(samples.cpu().numpy(), reference.cpu().numpy())
        
        evaluator.update(results, jsd)
        stats = evaluator.finalize_stats()
        
        stats['model_name'] = opt.model_type
        stats['Epoch'] = opt.niter-1
        stats['n_params'] = sum(param.numel() for param in model.parameters())/1e6
                
        if should_diag:
            logger.info(stats)
        
        result_path = os.path.join(output_dir, 'results.csv')
        if os.path.exists(result_path):
            df = pd.read_csv(result_path)
            df_stats = pd.DataFrame(stats, index=[0], columns=df.keys())
            df = pd.concat([df, df_stats], ignore_index=True)
            df.to_csv(result_path, index=False)
        else:
            # model_name,n_params,n_layers,n_hidden,patch_size,n_heads,Epoch,1-NNA-CD,1-NNA-EMD,COV-CD,COV-EMD,MMD-CD,MMD-EMD,JSD
            df_stats = pd.DataFrame(stats, index=[0], columns=['model_name','n_params','Epoch','1-NNA-CD','1-NNA-EMD','COV-CD','COV-EMD','MMD-CD','MMD-EMD','JSD'])
            df_stats.to_csv(result_path, index=False)
    
def train(opt, output_dir, logger, tb_writer, should_diag, model, train_sampler, ema, device, train_dataloader):
    # logger.info('='*20)
    # logger.info('Start training')
    # logger.info('='*20)
    
    if should_diag:
        # outf_syn, = setup_output_subdirs(output_dir, 'syn')
        outf_syn = os.path.join(output_dir, 'syn')
        os.makedirs(outf_syn, exist_ok=True)
    
    # ================ general training prep ================
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0)
    
    if opt.model != '':
        checkpoint = torch.load(opt.model)
        model_weights = checkpoint['ema' if opt.use_ema else 'model_state']
        # checkpoint_dict = {k.replace('model.', 'module.'): model_weights[k] for k in model_weights if k.startswith('model.')}
        checkpoint_dict = {k.replace('module.', ''): model_weights[k] for k in model_weights} if not opt.distributed else model_weights
        # checkpoint_dict = {'module.' + k: model_weights[k] for k in model_weights} if opt.distributed else model_weights
        model.load_state_dict(checkpoint_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
        if opt.use_ema:
            update_ema(ema, model.model, decay=0)  # Ensure EMA is initialized with synced weights

    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)

    def new_y_chain(y, num_chain, num_classes):
        return torch.randint(low=0,high=num_classes,size=(num_chain,),device=y.device)

    # Prepare models for training:
    if opt.use_ema:
        model.train()  # important! This enables embedding dropout for classifier-free guidance
        ema.eval()  # EMA model should always be in eval mode
    # ================ general training prep ================

    # ================ start training ================
    for epoch in range(start_epoch, opt.niter):
        # ================ training ================
        if opt.distributed:
            train_sampler.set_epoch(epoch)
            
        model.train()
        for i, data in enumerate(train_dataloader):
            x = data['train_points'].transpose(1,2) # [B, D, N]
            y = data['cate_idx']
            # print(x.shape, noises_batch.shape, y.shape) if i == 0 and epoch == 0 else None
            
            x, y, noises_batch = x.to(device), y.to(device), noises_batch.to(device)
            
            x = model.fold_model(x)
            
            loss, t_avg = model.get_loss_iter(x, noises_batch, y)
            loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                
            optimizer.step()
            
            if opt.use_ema:
                update_ema(ema, model.model)
                
            if not opt.debug:
                global_step = i + len(train_dataloader) * epoch
                if opt.use_tb:
                    tb_writer.add_scalar('train_loss', loss.item(), global_step)
                    tb_writer.add_scalar('train_lr', optimizer.param_groups[0]['lr'], global_step)
                    
            if i % opt.print_freq == 0 and should_diag:
                logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},   t_avg: {:.2f}'
                             .format(
                        epoch, opt.niter, i, len(train_dataloader), loss.item(), t_avg.item()
                        ))
        # ================ training ================
        
        
        model.eval()        
        # ================ visualization ================
        if (epoch + 1) % opt.vizIter == 0 and should_diag:
            logger.info('Generation: eval')
            
            with torch.no_grad():
                x_gen_eval = model.gen_samples(new_x_chain(x, 25).shape, device, new_y_chain(y,25,opt.num_classes), clip_denoised=False).flatten(1, 2)
                x_gen_list = model.gen_sample_traj(new_x_chain(x, 1).shape, device, new_y_chain(y,1,opt.num_classes), freq=40, clip_denoised=False)
                x_gen_all = torch.cat(x_gen_list, dim=0).flatten(1, 2)
                N_S = x_gen_all.shape[0] // x_gen_eval.shape[0]
                gen_stats = [x_gen_eval.mean(), x_gen_eval.std()]
                gen_eval_range = [x_gen_eval.min().item(), x_gen_eval.max().item()]
                
                logger.info('      [{:>3d}/{:>3d}]  '
                                'eval_gen_range: [{:>10.4f}, {:>10.4f}]     '
                                'eval_gen_stats: [mean={:>10.4f}, std={:>10.4f}]      '
                    .format(
                    epoch, opt.niter,
                    *gen_eval_range, *gen_stats,
                ))
            ep_save_dir = os.path.join(outf_syn, 'epoch_%03d' % epoch)
            os.makedirs(ep_save_dir, exist_ok=True)
            
            visualize_pointcloud_batch('%s/epoch_%03d_samples_eval.png' % (outf_syn, epoch),
                                       x_gen_eval, None, None,
                                       None)
            
            visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_all.png' % (outf_syn, epoch),
                                        x_gen_all, None,
                                        None,
                                        None)
            
            visualize_pointcloud_batch('%s/epoch_%03d_x.png' % (outf_syn, epoch), x.transpose(1, 2), None,
                                        None,
                                        None)
            for idx in range(x_gen_eval.shape[0]):
                pts = x_gen_eval[idx].cpu().numpy()
                np.savetxt(os.path.join(ep_save_dir, 'sample_%d.xyz' % idx), pts)
                
            logger.info('Generation: train')
        # ================ visualization ================
        
        # ================ save model ================
        if (epoch + 1) % opt.saveIter == 0:
            if should_diag:
                logger.info('Saving model...')
                save_dict = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }
                
                if opt.use_ema:
                    save_dict.update({'ema': ema.state_dict()})
                
                torch.save(save_dict, '%s/checkpoint.pth' % (output_dir))
                
            if opt.distributed:
                dist.barrier()
                map_location = {'cuda:%d' % 0: 'cuda:%d' % device}
                checkpoint = torch.load('%s/checkpoint.pth' % (output_dir), map_location=map_location)
            else:
                checkpoint = torch.load('%s/checkpoint.pth' % (output_dir))
            model_weights = checkpoint['ema' if opt.use_ema else 'model_state']
            # checkpoint_dict = {k.replace('model.', 'module.' if opt.distributed else ''): model_weights[k] for k in model_weights if k.startswith('model.')}
            checkpoint_dict = {k.replace('module.', ''): model_weights[k] for k in model_weights} if not opt.distributed else model_weights
            # checkpoint_dict = {'module.' + k: model_weights[k] for k in model_weights} if opt.distributed else model_weights
            model.load_state_dict(checkpoint_dict)
        # ================ save model ================      
                    
    if opt.distributed:
        dist.barrier()
        
    # logger.info('='*20)
    # logger.info('End training')
    # logger.info('='*20)
    # ================ end training ================


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='moment_gpt', help='experiment name (used for checkpointing and logging)')

    # Data params
    parser.add_argument('--dataroot', default='/home/yz4450/data/ShapeNetCore.v2.PC15k')
    parser.add_argument('--category', default='chair')
    parser.add_argument('--num_classes', type=int, default=1)

    parser.add_argument('--bs', type=int, default=128, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--npoints', type=int, default=2048)
    parser.add_argument('--fps_points', type=int, default=-1)
    parser.add_argument('--eval_npoints', type=int, default=2048)
    parser.add_argument('--eval_fps_points', type=int, default=-1)
    
    '''model'''
    parser.add_argument("--model_type", type=str, choices=list(DiT_models.keys()), default="DiT-S/4")
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', type=int, default=1000)

    #params
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

    parser.add_argument('--model', default='', help="path to model (to continue training)")
    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate model')
    parser.add_argument('--cfg_scale', type=float, default=1.0, help='scale the model cfg')

    '''poly'''
    parser.add_argument('--poly_type', default='chebyshev', choices=['monomial', 'chebyshev'])
    parser.add_argument('--poly_degree', default=15, type=int, help='degree of the polynomial')

    '''distributed'''
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='Number of distributed nodes.')
    # parser.add_argument('--node', type=str, default='localhost')
    # parser.add_argument('--port', type=int, default=12345)
    # parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    # parser.add_argument('--dist_backend', default='nccl', type=str,
    #                     help='distributed backend')
    # parser.add_argument('--distribution_type', default=None, choices=['multi', 'single', None],
    #                     help='Use multi-processing distributed training to launch '
    #                          'N processes per node, which has N GPUs. This is the '
                             
    #                          'fastest way to use PyTorch for either single node or '
    #                          'multi node data parallel training')
    # parser.add_argument('--rank', default=0, type=int,
    #                     help='node rank for distributed training')
    # parser.add_argument('--gpu', default=None, type=int,
    #                     help='GPU id to use. None means using all available GPUs.')

    '''eval'''
    parser.add_argument('--saveIter', default=1000, type=int, help='unit: epoch')
    parser.add_argument('--diagIter', default=1000, type=int, help='unit: epoch')
    parser.add_argument('--vizIter', default=1000, type=int, help='unit: epoch')
    parser.add_argument('--print_freq', default=50, type=int, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')
    parser.add_argument('--nrepeats', default=3, type=int, help='number of repeats for evaluation')

    parser.add_argument('--debug', action='store_true', default=False, help = 'debug mode')
    parser.add_argument('--use_tb', action='store_true', default=False, help = 'use tensorboard')
    parser.add_argument('--use_pretrained', action='store_true', default=False, help = 'use pretrained 2d DiT weights')
    parser.add_argument('--use_ema', action='store_true', default=False, help = 'use ema')

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()
