import time
import torch
from data import data_preparation
from models.vtr import VTR, TiT
from train import test, train
from utils.earlystop import EarlyStopping
from utils.checkpointsaver import CheckpointSaver, resume_checkpoint
import warnings
import argparse
import csv
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(0)

# Run arguments
parser = argparse.ArgumentParser(description='visual transformer with resnet backbone')
parser.add_argument('--model_type', help='which model to use, currently VTR_Resnet and TNT are supported',
                    default='vtr_resnet')
parser.add_argument('--patch_dim', help='Patch embedding dimensions for TNT',
                    default=384)
parser.add_argument('--pixel_dim', help='Pixel embedding dimensions for TNT',
                    default=32)
parser.add_argument('--patch_size', help='Patch size for TNT',
                    default=16)
parser.add_argument('--pixel_size', help='Pixel size for TNT',
                    default=8)
parser.add_argument('--data_dir', help='path to dataset', default='D:/datasets/imagenet')
parser.add_argument('--img_mean', type=float, nargs='+', default=[0.485, 0.456, 0.406],
                    help='dataset mean pixel values, default [0.485, 0.456, 0.406] for ImageNet')
parser.add_argument('--img_std', type=float, nargs='+', default=[0.229, 0.224, 0.225],
                    help='dataset std pixel values, default [0.229, 0.224, 0.225] for ImageNet')
parser.add_argument('--ds_split_ratio', type=int, nargs='+', default=[1000000, 281167],
                    help='Dataset train-test split ratio, must sum to len(dataset)')
parser.add_argument('--num_classes', type=int, default=1000,
                    help='Number of classes in the dataset, default 1000 for Imagenet')
parser.add_argument('--image_size', type=int, default=224, help='size of the input images(x,x square)')
parser.add_argument('--image_channels', type=int, default=3,
                    help='Number of channels in the image, default 3: RGB')
parser.add_argument('--batch_size_train', type=int, default=64, help='Batch size for the training data')
parser.add_argument('--batch_size_test', type=int, default=64, help='Batch size for the test data')
parser.add_argument('--n_epochs', type=int, default=50, help='Number of training Epochs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for the datagen, default 2')
parser.add_argument('--prefetch_factor', type=int, default=2, help='Data Prefetch factor')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate, default 1e-4')
parser.add_argument('--lr_decay_gamma', type=float, default=0.7, help='Gamma for learning rate decay')
parser.add_argument('--lr_eps', type=int, default=2e-7, help='Epsilon for AdamW')
parser.add_argument('--lr_weight_decay', type=int, default=1e-5, help='LR decay for AdamW')
parser.add_argument('--lr_decay_type', type=str, default='step',
                    help='Learning rate decay type, currently "step" and "cosine" are supported')
parser.add_argument('--step_decay_per', type=int, default=2, help='Step learning decay frequency')
parser.add_argument('--lr_cosine_T0', type=int, default=6, help='T0 for cosine decay')
parser.add_argument('--lr_cosine_TMULT', type=int, default=3, help='Multiplier for cosine LR decay')
parser.add_argument('--early_stop_patience', type=int, default=3, help='Early stopping patience')
parser.add_argument('--early_stop_min_delta', type=float, default=0,
                    help='Early stopping minimum delta to be counted as improvement')
parser.add_argument('--chkpoint_path', type=str, default='checkpoints', help='checkpoint path')
parser.add_argument('--chkpoint_maxhist', type=int, default=5,
                    help='checkpoint max history to keep before cleanup')
parser.add_argument('--transformer_depth', type=int, default=6, help='Depth of the transformer')
parser.add_argument('--transformer_heads', type=int, default=8, help='Heads per transformer')
parser.add_argument('--mlp_hidden', type=int, default=1536, help='MLP hidden dimension')
parser.add_argument('--dim', type=int, default=768,
                    help='Dimensions for the visual transformer and for the embedding')
parser.add_argument('--vtr_dropout', type=float, default=0.15, help='Dropout rate for the visual transformer')
parser.add_argument('--attn_dropout', type=float, default=0.1, help='Attention dropout rate')
parser.add_argument('--emb_dropout', type=float, default=0.1, help='Embedding dropout rate')
parser.add_argument('--disable_amp', action='store_false', default=False, help='Disable Automatic mixed precision')
parser.add_argument('--opt_level', type=str, default='O1', help='AMP opt level, default O1, O2 might cause NaN loss')
parser.add_argument('--backbone', type=str, default='resnest200',
                    help='CNN backbone to be used for feature extraction, resnet 50-101-152,'
                         'resnext 50-101, resnest 50-101-200-269 are supported, resnest200 is default')
parser.add_argument('--backbone_repo', type=str, default='zhanghang1989/ResNeSt',
                    help='repository for the backbone, see https://pytorch.org/docs/stable/hub.html')
parser.add_argument('--resume_checkpoint', action='store_true', help='resume from last checkpoint')
parser.add_argument('--last_checkpoint_path', help='path to last checkpoint')
args = parser.parse_args()

# CUDA/AMP
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = 'cpu'
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        use_fp16 = not args.disable_amp
        amp_scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
except AttributeError:
    pass

# Model selection
if args.model_type == 'vtr_resnet':
    model = VTR(img_size=args.image_size, in_chans=args.image_channels, num_classes=args.num_classes, dim=args.dim,
                depth=args.transformer_depth, num_heads=args.transformer_heads, mlp_hidden_dim=args.mlp_hidden,
                dropout=args.vtr_dropout, attn_dropout=args.attn_dropout, emb_dropout=args.emb_dropout,
                backbone=args.backbone,
                backbone_repo=args.backbone_repo).to(DEVICE)
elif args.model_type == 'TNT':
    model = TiT(image_size=args.image_size, patch_dim=args.patch_dim, pixel_dim=args.pixel_dim, patch_size=args.patch_size, pixel_size=args.pixel_size,
                depth=args.transformer_depth, num_classes=args.num_classes, attn_dropout=args.attn_dropout,
                dropout=args.vtr_dropout).to(DEVICE)

# Optimizer / decay
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, eps=args.lr_eps, weight_decay=args.lr_weight_decay)
if args.lr_decay_type == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_decay_per, gamma=args.lr_decay_gamma)
elif args.lr_decay_type == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                     T_0=args.lr_cosine_T0, T_mult=args.lr_cosine_TMULT)
else:
    print('Currently only step and cosine LR decay are supported')

# Utils
early_stopping = EarlyStopping(patience=args.early_stop_patience, delta=args.early_stop_min_delta)
checkpointer = CheckpointSaver(model=model, optimizer=optimizer, checkpoint_dir=args.chkpoint_path,
                               decreasing=True, max_history=args.chkpoint_maxhist, amp_scaler=amp_scaler)

# Data
train_loader, test_loader = data_preparation(
    data_dir=args.data_dir, batch_size_train=args.batch_size_train, batch_size_test=args.batch_size_test,
    image_size=args.image_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
    img_mean=args.img_mean, img_std=args.img_std, ds_split=args.ds_split_ratio)

if args.resume_checkpoint:
    resume_checkpoint(model, checkpoint_path=args.last_checkpoint_path, optimizer=optimizer, loss_scaler=amp_scaler)


# Training loop
def main():
    train_loss_history, test_loss_history, time_per_epoch = [], [], []
    for epoch in range(1, args.n_epochs + 1):
        total_time_left = round(sum(time_per_epoch) / epoch * (args.n_epochs - epoch) / 60)
        current_learning_rate = optimizer.param_groups[0]['lr']
        print(f'EPOCH: [{epoch}/{args.n_epochs}], Learning Rate: [{current_learning_rate}],'
              f'Estimated time left for {args.n_epochs} Epochs: {total_time_left} m')
        start_time = time.time()
        train(model, optimizer, train_loader, train_loss_history, use_fp16=use_fp16,
              amp_scaler=amp_scaler, device=DEVICE)
        with open('./trainloss.txt', mode='w') as trainloss:
            trainwriter = csv.writer(trainloss)
            trainwriter.writerow(train_loss_history)
        test(model=model, data_loader=test_loader, loss_history=test_loss_history, device=DEVICE)
        with open('./testloss.txt', mode='w') as testloss:
            testwriter = csv.writer(testloss)
            testwriter.writerow(test_loss_history)
        checkpointer.save_checkpoint(epoch, test_loss_history[-1])
        early_stopping(val_loss=test_loss_history[-1])
        epoch_time = time.time() - start_time
        time_per_epoch.append(epoch_time)
        if early_stopping.early_stop:
            print('Early stopping')
            break
        scheduler.step()
        print(f'\n{(time.time() - start_time):5.2f} second(s) / EPOCH')


if __name__ == '__main__':
    main()
