import torch
from torch.autograd import Variable
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    epoch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    accuracies_img = AverageMeter()

    epoch_end_time = time.time()
    end_time = time.time()
    for i, (inputs, targets, target_imgs) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets_vid = targets.cuda(non_blocking=True)
            #targets_img = target_imgs.view(-1).cuda(non_blocking=False)
            
        inputs = Variable(inputs)
        outputs = model(inputs)

        targets_vid = Variable(targets_vid)
        # targets_img = Variable(targets_img)
        # img_pred = torch.reshape(outputs[0], (-1, 400))
        loss_vid = criterion(outputs[0], targets_vid.squeeze())
        # loss_img = criterion(img_pred, targets_img)
        loss = loss_vid #+ loss_img
        acc_vid = calculate_accuracy(outputs[1], targets_vid)
        accuracies.update(acc_vid, inputs.size(0))
        acc_img = 0.0  # calculate_accuracy(img_pred, targets_img)
        accuracies_img.update(acc_img, inputs.size(0))
        losses.update(loss.data.cpu(), inputs.size(0))
        """
        targets_vid = Variable(targets_vid)
        targets_img = Variable(targets_img)
        loss_img = criterion(outputs, targets_vid)
        loss = loss_img
        acc_vid = 0
        accuracies.update(acc_vid, inputs.size(0))
        acc_img = calculate_accuracy(outputs, targets_vid)
        accuracies_img.update(acc_img, inputs.size(0))
        losses.update(loss.data.cpu(), inputs.size(0))
        """

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'acc_img': accuracies_img.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc_vid {acc_vid.val:.4f} ({acc_vid.avg:.4f})\t'
              'Acc_img {acc_img.val:.3f} ({acc_img.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc_vid=accuracies,
                  acc_img=accuracies_img))

    epoch_time.update(time.time() - epoch_end_time)

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'acc_img': accuracies_img.avg,
        'lr': optimizer.param_groups[-1]['lr'],
        'epoch_time': epoch_time.val
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

    return accuracies.avg
