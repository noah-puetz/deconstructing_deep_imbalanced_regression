import argparse
import os
import sys
import glob
import logging
import torch
import time
from model import Encoder, model_dict
from dataset import *
from utils import *
import datetime

import torch.nn as nn
from scipy.stats import gmean
from collections import defaultdict
import csv

print = logging.info


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=90, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')

    parser.add_argument('--data_folder', type=str, default='data', help='path to custom dataset')
    parser.add_argument('--split_csv', type=str, default='data/agedb.csv', help='path to split csv file')
    parser.add_argument('--dataset', type=str, default='AgeDB', choices=['AgeDB'], help='dataset')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--resume', type=str, default='', help='resume ckpt path')
    parser.add_argument('--aug', type=str, default='crop,flip,color,grayscale', help='augmentations')

    parser.add_argument('--ckpt', type=str, default='', help='path to the trained encoder')
    parser.add_argument('--output_csv', type=str, default='outputs.csv', help='path to the output csv file')
    parser.add_argument('--name', type=str, default='default', help='name of the run')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    opt = parser.parse_args()
     
    timestamp = str(datetime.datetime.now())
    timestamp = '-'.join(timestamp.split(' '))

    opt.model_name = 'Regressor_{}_ep_{}_lr_{}_d_{}_wd_{}_mmt_{}_bsz_{}_trial_{}_seed_{}_{}'. \
        format(opt.dataset, opt.epochs, opt.learning_rate, opt.lr_decay_rate,
               opt.weight_decay, opt.momentum, opt.batch_size, opt.trial, opt.seed,
               timestamp)
    
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-1][:-len('_last.pth')]
    
    # opt.save_folder = '/'.join(opt.ckpt.split('/')[:-1])
    opt.save_folder = './checkpoint_rnc/'

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, f'{opt.model_name}.log')),
            logging.StreamHandler()
        ]
    )

    print(f"Model name: {opt.model_name}")
    print(f"Save folder: {opt.save_folder}")
    print(f"Options: {opt}")
    
    if opt.device is None:
        opt.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {opt.device}")
    
    if opt.ckpt == 'latest':
        print("Loading latest checkpoint...")
        # get parent folder of all subdirs
        
        dir_key = './checkpoint_rnc/'
        all_subdirs = [d for d in os.listdir(dir_key) if os.path.isdir(os.path.join(dir_key, d))]        
        print(f"All subdirs: {all_subdirs}")
        all_subdirs = [f'{dir_key}{d}' for d in all_subdirs]
        latest_subdir = max(all_subdirs, key=os.path.getmtime)
        opt.ckpt = glob.glob(f'{latest_subdir}/last.pth')[0]
        print(f"Latest checkpoint: {opt.ckpt}")
        

    return opt


def set_loader(opt, g):
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    train_transform = get_transforms(split='train', aug=opt.aug)
    val_transform = get_transforms(split='val', aug=opt.aug)
    print(f"Train Transforms: {train_transform}")
    print(f"Val Transforms: {val_transform}")

    train_dataset = globals()[opt.dataset](data_folder=opt.data_folder, split_csv=opt.split_csv, transform=train_transform, split='train')
    val_dataset = globals()[opt.dataset](data_folder=opt.data_folder, split_csv=opt.split_csv, transform=val_transform, split='val')
    test_dataset = globals()[opt.dataset](data_folder=opt.data_folder, split_csv=opt.split_csv, transform=val_transform, split='test')

    print(f'Train set size: {train_dataset.__len__()}\t'
          f'Val set size: {val_dataset.__len__()}\t'
          f'Test set size: {test_dataset.__len__()}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )

    return train_loader, val_loader, test_loader


def set_model(opt):
    model = Encoder(name=opt.model)
    criterion = torch.nn.L1Loss()

    dim_in = model_dict[opt.model][1]
    dim_out = get_label_dim(opt.dataset)
    regressor = torch.nn.Linear(dim_in, dim_out)
    ckpt = torch.load(opt.ckpt, map_location='cpu', weights_only=False)
    state_dict = ckpt['model']

    if torch.cuda.device_count() > 1:
        model.encoder = torch.nn.DataParallel(model.encoder)
        torch.backends.cudnn.benchmark = False # for reproducibility
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict

    model = model.to(opt.device)
    regressor = regressor.to(opt.device)
    criterion = criterion.to(opt.device)

    model.load_state_dict(state_dict)
    print(f"<=== Epoch [{ckpt['epoch']}] checkpoint Loaded from {opt.ckpt}!")

    return model, regressor, criterion


def train(train_loader, model, regressor, criterion, optimizer, epoch, opt):
    model.eval()
    regressor.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.shape[0]

        with torch.no_grad():
            features = model(images)

        output = regressor(features.detach())
        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()


def validate(val_loader, model, regressor, opt):
    model.eval()
    regressor.eval()

    losses = AverageMeter()
    criterion_l1 = torch.nn.L1Loss()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.to(opt.device)
            labels = labels.to(opt.device)
            bsz = labels.shape[0]

            features = model(images)
            output = regressor(features)

            loss_l1 = criterion_l1(output, labels)
            losses.update(loss_l1.item(), bsz)

    return losses.avg


def final_validation(val_loader, model, regressor, opt, train_labels=None):
    def shot_metrics(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
        train_labels = np.array(train_labels).astype(int)

        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        elif isinstance(preds, np.ndarray):
            pass
        else:
            raise TypeError(f'Type ({type(preds)}) of predictions not supported')

        train_class_count, test_class_count = [], []
        mse_per_class, l1_per_class, l1_all_per_class = [], [], []
        l_unique = np.unique(labels)
        for l in np.unique(labels):
            train_class_count.append(len(train_labels[train_labels == l]))
            test_class_count.append(len(labels[labels == l]))
            mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
            l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
            l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

        many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
        many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
        many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
        many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []
        many_l, median_l, low_l = [], [], []

        for i in range(len(train_class_count)):
            if train_class_count[i] > many_shot_thr:
                many_shot_mse.append(mse_per_class[i])
                many_shot_l1.append(l1_per_class[i])
                many_shot_gmean += list(l1_all_per_class[i])
                many_shot_cnt.append(test_class_count[i])
                many_l.append(l_unique[i])
            elif train_class_count[i] < low_shot_thr:
                low_shot_mse.append(mse_per_class[i])
                low_shot_l1.append(l1_per_class[i])
                low_shot_gmean += list(l1_all_per_class[i])
                low_shot_cnt.append(test_class_count[i])
                low_l.append(l_unique[i])
            else:
                median_shot_mse.append(mse_per_class[i])
                median_shot_l1.append(l1_per_class[i])
                median_shot_gmean += list(l1_all_per_class[i])
                median_shot_cnt.append(test_class_count[i])
                median_l.append(l_unique[i])

        shot_dict = defaultdict(dict)
        if sum(many_shot_cnt) > 0:
            shot_dict['many']['mse'] = np.sum(many_shot_mse) / np.sum(many_shot_cnt)
            shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
            shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
        else:
            shot_dict['many']['mse'] = shot_dict['many']['l1'] = shot_dict['many']['gmean'] = 0

        if sum(median_shot_cnt) > 0:
            shot_dict['median']['mse'] = np.sum(median_shot_mse) / np.sum(median_shot_cnt)
            shot_dict['median']['l1'] = np.sum(median_shot_l1) / np.sum(median_shot_cnt)
            shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
        else:
            shot_dict['median']['mse'] = shot_dict['median']['l1'] = shot_dict['median']['gmean'] = 0

        if sum(low_shot_cnt) > 0:
            shot_dict['low']['mse'] = np.sum(low_shot_mse) / np.sum(low_shot_cnt)
            shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
            shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)
        else:
            shot_dict['low']['mse'] = shot_dict['low']['l1'] = shot_dict['low']['gmean'] = 0
    
        
        shot_dict['many']['label'] = many_l
        shot_dict['median']['label'] = median_l
        shot_dict['low']['label'] = low_l

        return shot_dict

    class AverageMeter(object):
        """Computes and stores the average and current value"""

        def __init__(self, name, fmt=":f"):
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
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
            return fmtstr.format(**self.__dict__)
    
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')

    model.eval()
    regressor.eval()
    losses_all = []
    preds, labels = [], []
    f=[]

    with torch.no_grad():
        pred_dict ={}
        for idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(opt.device, non_blocking=True), targets.to(opt.device, non_blocking=True)
            outputs = regressor(model(inputs))
        
            preds.extend(outputs.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())

            loss_mse = criterion_mse(outputs, targets)
            loss_l1 = criterion_l1(outputs, targets)
            loss_all = criterion_gmean(outputs, targets)
            losses_all.extend(loss_all.cpu().numpy())

            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))
    
        shot_dict = shot_metrics(np.hstack(preds), np.hstack(labels), train_labels)
        loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
        print(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")
        print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\t"
              f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}")
        print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\t"
              f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}")
        print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\t"
              f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}")
        
    return losses_mse.avg, losses_l1.avg, loss_gmean, shot_dict

def main():
    opt = parse_option()

    seed_everything(opt.seed)

    g = torch.Generator()
    g.manual_seed(opt.seed)

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt, g)

    # build model and criterion
    model, regressor, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, regressor)

    save_file_best = os.path.join(opt.save_folder, f"{opt.model_name}_best.pth")
    save_file_last = os.path.join(opt.save_folder, f"{opt.model_name}_last.pth")
    best_error = 1e5

    start_epoch = 1
    if len(opt.resume):
        ckpt_state = torch.load(opt.resume)
        regressor.load_state_dict(ckpt_state['state_dict'])
        start_epoch = ckpt_state['epoch'] + 1
        best_error = ckpt_state['best_error']
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {opt.resume}!")


    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, regressor, criterion, optimizer, epoch, opt)

        valid_error = validate(val_loader, model, regressor, opt)
        print('Val L1 error: {:.3f}'.format(valid_error))

        is_best = valid_error < best_error
        best_error = min(valid_error, best_error)
        print(f"Best Error: {best_error:.3f}")

        if is_best:
            torch.save({
                'epoch': epoch,
                'state_dict': regressor.state_dict(),
                'best_error': best_error
            }, save_file_best)

        torch.save({
            'epoch': epoch,
            'state_dict': regressor.state_dict(),
            'last_error': valid_error
        }, save_file_last)


    print("=" * 120)
    print("Test best model on test set...")
    checkpoint = torch.load(save_file_best)
    regressor.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded best model, epoch {checkpoint['epoch']}, best val error {checkpoint['best_error']:.3f}")
    test_loss = validate(test_loader, model, regressor, opt)
    to_print = 'Test L1 error: {:.3f}'.format(test_loss)
    print(to_print)
    
    df = pd.read_csv(opt.split_csv)
    df_train = df[df['split'] == 'train']
    
    test_loss_mse, test_loss_l1, test_loss_gmean,shot_dict = final_validation(test_loader, model, regressor, opt=opt, train_labels=df_train['age'])

    file_exists = os.path.isfile(opt.output_csv)
    with open(opt.output_csv, mode='a') as outputs_file:
        outputs = csv.writer(outputs_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        # Only write header if file doesn't exist
        if not file_exists:
            outputs.writerow(['Run','MAE All', 'MAE Man', 'MAE Med','MAE Few','GMean All', 'GMean Man', 'GMean Med','GMean Few'])
            
        to_write = [opt.name,test_loss_l1,shot_dict['many']['l1'],shot_dict['median']['l1'],shot_dict['low']['l1'],test_loss_gmean,shot_dict['many']['gmean'],shot_dict['median']['gmean'],shot_dict['low']['gmean']]
        outputs.writerow(to_write)


if __name__ == '__main__':
    main()
