import argparse
import os
import logging

import torch
import torch.distributed
import torch.nn as nn
from torch import optim
from source.training_utils import save_checkpoint, save_model, LinearWarmupScheduler, add_gradient_histograms
from source.data.datasets.objs.load_data import load_data

from source.utils import str2bool
from torch.utils.tensorboard import SummaryWriter

import accelerate
from accelerate import Accelerator

# Visualization
import torch.nn.functional as F
from tqdm import tqdm

# for distributed training
from torch.distributed.nn.functional import all_gather


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


def simclr(zs, temperature=1.0, normalize=True, loss_type="ip"):
    # zs: list of tensors. Each tensor has shape (n, d)
    if normalize:
        zs = [F.normalize(z, p=2, dim=-1) for z in zs]
    if zs[0].dim() == 3:
        zs = [z.flatten(1, 2) for z in zs]
    m = len(zs)
    n = zs[0].shape[0]
    device = zs[0].device
    mask = torch.eye(n * m, device=device)
    label0 = torch.fmod(n + torch.arange(0, m * n, device=device), n * m)
    z = torch.cat(zs, 0)
    if loss_type == "euclid":  # euclidean distance
        sim = -torch.cdist(z, z)
    elif loss_type == "sq":  # squared euclidean distance
        sim = -(torch.cdist(z, z) ** 2)
    elif loss_type == "ip":  # inner product
        sim = torch.matmul(z, z.transpose(0, 1))
    else:
        raise NotImplementedError
    logit_zz = sim / temperature
    logit_zz += mask * -1e8
    loss = nn.CrossEntropyLoss()(logit_zz, label0)
    return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Training options
    parser.add_argument("--exp_name", type=str, help="expname")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--beta", type=float, default=0.998, help="ema decay")
    parser.add_argument("--epochs", type=int, default=500, help="num of epochs")
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="save checkpoint every specified epochs",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="lr")
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument(
        "--finetune",
        type=str,
        default=None,
        help="path to the checkpoint. Training starts from that checkpoint",
    )

    # Data loading
    parser.add_argument("--limit_cores_used", type=str2bool, default=False)
    parser.add_argument("--cpu_core_start", type=int, default=0, help="start core")
    parser.add_argument("--cpu_core_end", type=int, default=32, help="end core")
    parser.add_argument("--data", type=str, default="clevrtex")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="optional. you can specify the dir path if the default path of each dataset is not appropritate one. Currently only applied to ImageNet",
    )
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--data_imsize",
        type=int,
        default=None,
        help="Image size. If None, use the default size of each dataset",
    )

    # Simclr options
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--temp", type=float, default=0.1, help="simclr temperature.")

    # General model options
    parser.add_argument("--model", type=str, default="knet", help="model")
    parser.add_argument("--L", type=int, default=2, help="num of layers")
    parser.add_argument("--ch", type=int, default=256, help="num of channels")
    parser.add_argument(
        "--model_imsize",
        type=int,
        default=None,
        help=
        """
        Model's imsize that was set when it was initialized. 
        This is used when you want finetune a pretrained model 
        that was trained on images with different resolution than the finetune image dataset.
        """
    )
    parser.add_argument("--autorescale", type=str2bool, default=False)
    parser.add_argument("--psize", type=int, default=8, help="patch size")
    parser.add_argument("--ksize", type=int, default=1, help="kernel size")
    parser.add_argument("--T", type=int, default=8, help="num of recurrence")
    parser.add_argument(
        "--maxpool", type=str2bool, default=True, help="max pooling or avg pooling"
    )
    parser.add_argument(
        "--heads", type=int, default=8, help="num of heads in self-attention"
    )
    parser.add_argument(
        "--gta",
        type=str2bool,
        default=True,
        help="""
        use Geometric Transform Attention (https://github.com/autonomousvision/gta) as positional encoding.
        If False, use standard absolute positional encoding
        """,
    )

    # AKOrN options
    parser.add_argument("--N", type=int, default=4, help="num of rotating dimensions")
    parser.add_argument("--gamma", type=float, default=1.0, help="step size")
    parser.add_argument("--J", type=str, default="conv", help="connectivity")
    parser.add_argument("--use_omega", type=str2bool, default=False)
    parser.add_argument("--global_omg", type=str2bool, default=False)
    parser.add_argument(
        "--c_norm",
        type=str,
        default="gn",
        help="normalization. gn, sandb(scale and bias), or none",
    )
    parser.add_argument(
        "--init_omg", type=float, default=0.01, help="initial omega length"
    )
    parser.add_argument("--learn_omg", type=str2bool, default=False)
    parser.add_argument(
        "--use_ro_x",
        type=str2bool,
        default=False,
        help="apply linear transform to oscillators between consecutive layers",
    )

    # ablation of some components in the AKOrN's block
    parser.add_argument(
        "--no_ro", type=str2bool, default=False, help="ablation: no use readout module"
    )
    parser.add_argument(
        "--project",
        type=str2bool,
        default=True,
        help="use projection or not in the Kuramoto layer",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
    )

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(enabled=True)
    # Setup accelerator
    accelerator = Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(args.seed + accelerator.process_index)

    # Create job directory and logger
    jobdir = f"runs/{args.exp_name}/"
    if accelerator.is_main_process:
        if not os.path.exists(jobdir):
            os.makedirs(jobdir)  # Make results folder (holds all experiment subfolders)
            logger = create_logger(jobdir)
            logger.info(f"Experiment directory created at {jobdir}")
        else:
            logger = create_logger(jobdir)

    if args.limit_cores_used:
        def worker_init_fn(worker_id):
            os.sched_setaffinity(0, range(args.cpu_core_start, args.cpu_core_end))

    else:
        worker_init_fn = None

    sstrainset, imsize, _ = load_data(args.data, args.data_root, args.data_imsize, False)

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(sstrainset):,} images")

    ssloader = torch.utils.data.DataLoader(
        sstrainset,
        batch_size=int(args.batchsize // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )

    if accelerator.is_main_process:
        writer = SummaryWriter(jobdir)

    def train(net, ema, opt, scheduler, loader, epoch):
        losses = []
        initial_params = {name: param.clone() for name, param in net.named_parameters()}
        running_loss = 0.0
        n = 0

        for i, data in tqdm(enumerate(loader, 0)):
            net.train()
            inputs = data.view(-1, 3, imsize, imsize).to(device)  # 2x batchsize

            # forward
            outputs = net(inputs)

            # gather outputs because simclr loss requires all outputs across all processes
            if accelerator.num_processes > 1:
                outputs = torch.cat(all_gather(outputs), 0)
            outputs = outputs.unflatten(0, (outputs.shape[0] // 2, 2))

            loss = simclr(
                [outputs[:, 0], outputs[:, 1]],
                temperature=args.temp,
                normalize=args.normalize,
                loss_type="ip",
            )

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            scheduler.step()

            running_loss += loss.item() * inputs.shape[0]
            n += inputs.shape[0]

            ema.update()

        if accelerator.is_main_process:
            add_gradient_histograms(writer, net, epoch)
            for name, param in net.named_parameters():
                diff = param - initial_params[name]
                writer.add_histogram(f"{name}_diff", diff, epoch)
        if accelerator.is_main_process:
            logger.info(
                f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss/n:.3f}"
            )

        total_loss = running_loss / n
        if accelerator.is_main_process:
            writer.add_scalar("training loss", total_loss, epoch)

        return total_loss

    if args.model == "akorn":
        from source.models.objs.knet import AKOrN

        net = AKOrN(
            args.N,
            ch=args.ch,
            L=args.L,
            T=args.T,
            gamma=args.gamma,
            J=args.J,  # "conv" or "attention",
            use_omega=args.use_omega,
            global_omg=args.global_omg,
            c_norm=args.c_norm,
            psize=args.psize,
            imsize=imsize if args.model_imsize is None else args.model_imsize,
            autorescale=args.autorescale,
            init_omg=args.init_omg,
            learn_omg=args.learn_omg,
            maxpool=args.maxpool,
            project=args.project,
            heads=args.heads,
            use_ro_x=args.use_ro_x,
            no_ro=args.no_ro,
            gta=args.gta,
        ).to("cuda")

    elif args.model == "vit":
        from source.models.objs.vit import ViT
        # ItrSA if T > 1. Otherwise, it becomesa a standard transformer
        net = ViT(
            psize=args.psize,
            imsize=imsize if args.model_imsize is None else args.model_imsize,
            autorescale=args.autorescale,
            ch=args.ch,
            blocks=args.L,
            heads=args.heads,
            mlp_dim=2 * args.ch,
            T=args.T,
            maxpool=args.maxpool,
            gta=args.gta,
        ).cuda()
    else:
        raise NotImplementedError

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total number of basemodel parameters: {total_params}")

    if args.finetune:
        if accelerator.is_main_process:
            logger.info("Loading checkpoint...")
        net.load_state_dict(torch.load(args.finetune)["model_state_dict"])

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)

    if args.finetune:
        if accelerator.is_main_process:
            logger.info("Loading optimizer state...")
        optimizer.load_state_dict(torch.load(args.finetune)["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr

    from ema_pytorch import EMA

    ema = EMA(net, beta=args.beta, update_every=10, update_after_step=200)

    if args.finetune:
        if accelerator.is_main_process:
            logger.info("Loading checkpoint...")
        dir_name, file_name = os.path.split(args.finetune)
        file_name = file_name.replace("checkpoint", "ema")
        ema_path = os.path.join(dir_name, file_name)
        ema.load_state_dict(torch.load(ema_path)["model_state_dict"])

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")

    net, optimizer, ssloader = accelerator.prepare(net, optimizer, ssloader)

    scheduler = LinearWarmupScheduler(optimizer, warmup_iters=args.warmup_iters)

    for epoch in range(0, args.epochs):
        total_loss = train(net, ema, optimizer, scheduler, ssloader, epoch)
        if (epoch + 1) % args.checkpoint_every == 0:
            if accelerator.is_main_process:
                save_checkpoint(
                    accelerator.unwrap_model(net),
                    optimizer,
                    epoch,
                    total_loss,
                    checkpoint_dir=jobdir,
                )
                save_model(ema, epoch, checkpoint_dir=jobdir, prefix="ema")
    if accelerator.is_main_process:
        torch.save(
            accelerator.unwrap_model(net).state_dict(),
            os.path.join(jobdir, f"model.pth"),
        )
        torch.save(ema.state_dict(), os.path.join(jobdir, f"ema_model.pth"))
