import time
from project.data import data_preparation
from project.hyperparameters import *
from project.models.vtr import VTR
from project.train import test, train
from project.utils.earlystop import EarlyStopping
from project.utils.checkpointsaver import CheckpointSaver
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(0)

# CUDA/AMP
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = 'cpu'
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        use_fp16 = True
        amp_scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
except AttributeError:
    pass

model = VTR().to(DEVICE)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LEARNING_RATE, eps=EPSILON, weight_decay=WEIGHT_DECAY)
if DECAY_TYPE == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_PER, gamma=GAMMA)
elif DECAY_TYPE == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                     T_0=COS_T0, T_mult=COS_T_MULT)

early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, delta=EARLY_STOP_DELTA)
checkpointer = CheckpointSaver(model=model, optimizer=optimizer, checkpoint_dir=CHECKPOINT_PATH,
                               decreasing=True, max_history=CHECKPOINT_MAX_HIST, amp_scaler=None)

train_loader, test_loader = data_preparation(DATA_DIR, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)


def main():
    train_loss_history, test_loss_history, time_per_epoch = [], [], []

    for epoch in range(1, N_EPOCHS + 1):
        total_time_left = round(sum(time_per_epoch)/epoch * (N_EPOCHS - epoch) / 60)
        current_learning_rate = optimizer.param_groups[0]['lr']
        print(f'EPOCH: [{epoch}/{N_EPOCHS}], Learning Rate: [{current_learning_rate}],'
              f'Estimated time left for {N_EPOCHS} Epochs: {total_time_left} m')
        start_time = time.time()
        train(model, optimizer, train_loader, train_loss_history, use_fp16=use_fp16,
              amp_scaler=amp_scaler, device=DEVICE)
        test(model, test_loader, test_loss_history, device=DEVICE)
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
