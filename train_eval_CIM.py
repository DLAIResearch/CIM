
import argparse
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_CUDA_ARCH_LIST"] = "sm_90"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import random
import shutil
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataset.imagefolder_CIM import ImageFolder
import models.resnet_multigpu_CIM as resnet
import torch.nn.functional as F
from  CLIP_Util import get_CLIP_text_feature

import logging
from sklearn.metrics import recall_score, f1_score,precision_score
def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r", encoding='utf-8') as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


model_names = ['resnet18' , 'resnet50']
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',default='dataset/cars',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained',default=True, action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--save_dir', default='checkpoint', type=str, metavar='SV_PATH',
                    help='path to save checkpoints (default: none)')
parser.add_argument('--log_dir', default='logs-cars-18_test', type=str, metavar='LG_PATH',
                    help='path to write logs (default: logs)')
parser.add_argument('--dataset', type=str, default='cars',
                            help='dataset to use: [imagenet, tiny_imagenet]')
parser.add_argument('-t', type=float, default=0.5)
#TO loss
parser.add_argument('--lambda', default=1, type=float,
                    metavar='LAM', help='lambda hyperparameter ', dest='lambda_val')
#CR-loss
parser.add_argument('--beta', default=1, type=float,
                    metavar='BET', help='lambda hyperparameter', dest='beta_val')

best_acc1 = 0


def main():
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(logpath=os.path.join(args.log_dir, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, logger)


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self , T=0.01):
        super(NCESoftmaxLoss, self).__init__()
        self.T = T
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        x = torch.div(x, self.T)
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss

class EIR_Loss(nn.Module):

    def __init__(self, normalize=False, temperature=0.5):
        super(EIR_Loss, self).__init__()
        self.normalize = normalize
        self.temperature = temperature

    def forward(self,zi, zo):

        bs = zi.shape[0]
        zi=torch.flatten(zi, start_dim=1)
        zo=torch.flatten(zo, start_dim=1)
        zi_norm = F.normalize(zi, p=2, dim=-1)
        zo_norm = F.normalize(zo, p=2, dim=-1)
        logits_io = torch.mm(zi_norm, zi_norm.t()) / self.temperature

        probs_io = F.softmax(logits_io, -1)

        logits_zo = torch.mm(zo_norm, zo_norm.t()) / self.temperature

        probs_zo = F.softmax(logits_zo, -1)

        Zo_zI_loss = F.kl_div(torch.log(probs_io+1e-8), probs_zo, log_target=False, reduction="batchmean")

        return Zo_zI_loss


class EIR_Loss2(nn.Module):

    def __init__(self, normalize=False, temperature=0.5):
        super(EIR_Loss2, self).__init__()
        self.normalize = normalize
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self,zi, zo, T):

        bs = zo.shape[0]
        # zi = zi / zi.norm(dim=1, keepdim=True)
        zo = zo / zo.norm(dim=1, keepdim=True)
        T = T / T.norm(dim=1, keepdim=True).float()
        logit_scale = self.logit_scale.exp()
        # logits_zi = logit_scale * zi @ T.t()
        logits_zo = logit_scale * zo @ T.t()
        logits_zi=zi.softmax(dim=-1)
        logits_zo = logits_zo.softmax(dim=-1)
        abs_diff = torch.abs(logits_zi - logits_zo)
        # 求平均
        loss = torch.sum(abs_diff)/bs
        return loss


class SingleLayerPerceptron(nn.Module):
    def __init__(self):
        super(SingleLayerPerceptron, self).__init__()
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = self.fc(x)
        return x
def main_worker(gpu, ngpus_per_node, args, logger):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    kwargs = {}
    num_classes = 1000
    val_dir_name = 'val'
    if args.dataset == 'imagenet':
        kwargs = {'num_classes': 100}
        num_classes = 100
    elif args.dataset == 'cub':
        kwargs = {'num_classes': 200}
        num_classes = 200
    elif args.dataset == 'cars':
        kwargs = {'num_classes': 196}
        num_classes = 196

    # create model
    if args.pretrained:
        if args.arch == 'resnet18':
            logger.info("=> using pre-trained model 'resnet18'")
            model = resnet.resnet18(pretrained=True)
        elif args.arch == 'resnet50':
            logger.info("=> using pre-trained model 'resnet50'")
            model = resnet.resnet50(pretrained=True)
        else:
            print('Arch not supported!!')
            exit()
    else:
        if args.arch == 'resnet18' :
            logger.info("=> creating model 'resnet18'")
            model = resnet.resnet18()
        elif args.arch == 'resnet50':
            logger.info("=> creating model 'resnet50'")
            model = resnet.resnet50()
        else:
            print('Arch not supported!!')
            exit()
    if args.arch == 'resnet18':
        model.fc = nn.Linear(512, num_classes)
    elif args.arch == 'resnet50':
        model.fc = nn.Linear(2048, num_classes)
    # model = torch.nn.DataParallel(model)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # if args.arch == 'resnet18':
    #     model.module.fc = nn.Linear(512, num_classes)
    # elif args.arch == 'resnet50':
    #     model.module.fc = nn.Linear(2048, num_classes)

    mlp_model = SingleLayerPerceptron().cuda()
    model = model.cuda()
    print(mlp_model)
    print(model)
    logger.info(model)

    # define loss function (criterion) and optimizer
    xent_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # contrastive_criterion = NCESoftmaxLoss(args.t).cuda(args.gpu)
    CR_Loss= EIR_Loss().cuda(args.gpu)
    TO_Loss = EIR_Loss2().cuda(args.gpu)
    cudnn.benchmark = True
    optimizer = torch.optim.SGD(list(model.parameters())+list(mlp_model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, val_dir_name)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolder(traindir)   # transforms are handled within the implementation

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_batch_size = args.batch_size
    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    if args.evaluate:
        validate(val_loader, model, EIR_Loss, xent_criterion, args, logger)
        return
    text_features=get_CLIP_text_feature()
    text_features = torch.stack(text_features).squeeze(1)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model,mlp_model, CR_Loss ,TO_Loss, xent_criterion, optimizer, epoch,text_features, args, logger)

        # evaluate on validation set
        acc1 = validate(val_loader, model, EIR_Loss, xent_criterion, args, logger)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                # 'mlp_model': mlp_model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best,args.save_dir)






def train(train_loader, model, mlp_model,CR_Loss ,TO_Loss,xent_criterion, optimizer, epoch,text_features, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    xe_losses = AverageMeter('xe Loss', ':.4e')
    TO_losses = AverageMeter('TO Loss', ':.4e')
    CR_losses= AverageMeter('CR Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.4f')
    top5 = AverageMeter('Acc@5', ':6.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, xe_losses, TO_losses,CR_losses, losses, top1, top5],
        logger,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    mlp_model.train()
    train_len = len(train_loader)
    train_iter = iter(train_loader)
    end = time.time()
    for i in range(train_len):
        optimizer.zero_grad()
        xe_images,images, aug_images,targets = train_iter.__next__()
        text_features1=mlp_model(text_features.float())
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.cuda(args.gpu, non_blocking=True)
        aug_images = aug_images.cuda(args.gpu, non_blocking=True)
        # text_features_batch=[text_cfeatures[target] for target in targets]
        targets = targets.cuda(args.gpu, non_blocking=True)
        xe_images= xe_images.cuda(args.gpu, non_blocking=True)
        aug_output, xe_loss, CR_loss, TO_loss= model(images, xe_images=xe_images, aug_images=aug_images, CR_Loss=CR_Loss,
                                                      TO_Loss=TO_Loss,  targets=targets, xent_criterion=xent_criterion,text_feature=text_features1,vanilla=False)
        xe_loss = xe_loss.mean()
        CR_loss = CR_loss.mean()
        TO_loss = TO_loss.mean()
        loss = xe_loss +args.beta_val * CR_loss + args.lambda_val * TO_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(aug_output, targets, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        xe_losses.update(xe_loss.item(), images.size(0))
        TO_losses.update(TO_loss.item(), images.size(0))
        CR_losses.update(CR_loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(val_loader, model, TO_Loss, criterion, args, logger):
    batch_time = AverageMeter('Time', ':6.4f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.4f')
    top5 = AverageMeter('Acc@5', ':6.4f')
    f1s = AverageMeter('f1', ':6.4f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5,f1s],
        logger,
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        all_targets = []
        all_preds = []
        for i, (images, targets) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

            output = model(images, vanilla=True)
            loss = criterion(output, targets)

            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            preds = torch.argmax(output, dim=1)

            # 收集标签和预测，放到cpu，追加保存
            all_targets.append(targets.detach().cpu())
            all_preds.append(preds.detach().cpu())

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # 循环结束，拼接全部
        all_targets = torch.cat(all_targets).numpy()
        all_preds = torch.cat(all_preds).numpy()
        f1_global = f1_score(all_targets, all_preds, average='weighted')

        logger.info(' * Acc@1 {top1.avg:.4f} Acc@5 {top5.avg:.4f} f1 {f1:.4f} '
                    .format(top1=top1, top5=top5, f1=f1_global))
    return top1.avg

def save_checkpoint(state, is_best, save_dir):
    epoch = state['epoch']
    filename = 'checkpoint_' + str(epoch).zfill(3) + '.pth.tar'
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)
    if is_best:
        best_filename = 'model_best.pth_imagenet-18-cars-test.tar'
        best_save_path = os.path.join(save_dir, best_filename)
        shutil.copyfile(save_path, best_save_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters,logger, prefix="" ):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]

        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch //30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

