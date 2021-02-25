import torch
import torch.nn.functional as F
import time


def train(model, optimizer, data_loader, loss_history, amp_scaler, device, use_fp16=True):
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    model.train()
    total_step_time = []
    for i, (data, target) in enumerate(data_loader, start=1):
        step_start = time.time()
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_fp16):
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target)
        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()
        optimizer.zero_grad()
        _, pred = torch.max(output, dim=1)
        correct_samples += pred.eq(target).sum()
        loss_history.append(loss.item())
        print(f'\r[{i*len(data)}/{total_samples}({(100*i/len(data_loader)):3.0f}%)], '
              f'Avg loss:[{(sum(loss_history[-i:])/i):2.4f}], Training Accuracy:'
              f'[{(100.0 * correct_samples / total_samples):4.2f}%], Elapsed Time: {round(sum(total_step_time)):.0f}s '
              f'Time left: {sum(total_step_time)/i*(len(data_loader) - i):.0f}s', end='')
        total_step_time.append(time.time() - step_start)


def test(model, data_loader, loss_history, device):
    model.eval()
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader, start=1):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()
            if i == 1:
                print('\n')
            print(f'\rTesting [{((i/len(data_loader))*100):3.2f}%] Test loss: '
                  f'[{(total_loss/total_samples):6.4f}], Test accuracy: {correct_samples:5}/{total_samples:5}'
                  f' ({(100*correct_samples/total_samples):4.2f}%)', end='')
    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print(f'\rAverage test loss:[{avg_loss:.4f}], Test Accuracy:[{correct_samples}/{total_samples}'
          f'({(100.0 * correct_samples / total_samples):4.2f}%)]')
