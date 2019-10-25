from train import parse_args, PRED_THRESHOLD
from dist_helpers import *
from logger import setup_logger, json_dump
from affordancedata import ToTensorBCHW, GrayAffordancesDataset
from models import InitialConvModel
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
import torch.nn as nn
import torch
import os
import sys
import cv2
from PIL import Image
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)


affords = ["solid", "movable", "destroyable",
           "dangerous", "gettable", "portal", "usable", "changeable", "ui"]


def run_eval(args):
    #logger = args.logger
    #logger.info("Eval Started for {}_{}".format(args.model, args.trial_id))
    logger = args.logger
    writer = SummaryWriter(log_dir=os.path.join(
        'runs', 'conv_' + args.trial_id))
    test_dataset = GrayAffordancesDataset(
        image_dir="data/sm3/img/", affordances_dir="data/sm3/label/", transform=ToTensorBCHW())
    test_dataset = Subset(test_dataset, range(8))
    # train_dataset, test_dataset = random_split(
    #     test_dataset, [len(test_dataset)-100, 100])

    device = args.device
    test_loader = DataLoader(dataset=test_dataset,
                             shuffle=False,
                             batch_size=1,
                             num_workers=args.workers,
                             pin_memory=True)
    if args.model == 'conv':
        model = InitialConvModel().to(device)
    if args.not_best:
        saved_model_path = os.path.join(
            args.save_dir, "{}_{}.pth".format(args.model, args.trial_id))
    else:
        saved_model_path = os.path.join(
            args.save_dir, "{}_{}_best_model.pth".format(args.model, args.trial_id))
    model.load_state_dict(torch.load(saved_model_path))
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').to(device)
    best_pred = 1.0

    model.eval()
    hamming_dists = torch.zeros(len(test_loader), device=device)
    hamming_diffs = torch.zeros((len(test_loader), 9, 224, 256), device=device)
    predictions = torch.zeros((len(test_loader), 9, 224, 256), device=device)
    images = torch.empty(len(test_loader), 1, 224, 256)
    print(hamming_dists.size(), hamming_diffs.size())
    for i, data in enumerate(test_loader):
        image = data['image'].to(device)
        target = data['affordances'].to(device)

        with torch.no_grad():
            output = model(image)
        #metric.update(output[0], target)
        output = torch.sigmoid(output)
        hamming_dists[i] = hamming_dist_norm(output, target, device)
        hamming_diffs[i] = hamming_diff(
            output.squeeze(0), target.squeeze(0), device)
        bi_pred = torch.where(output.squeeze(0) > PRED_THRESHOLD, torch.ones(
            [1], device=device, dtype=torch.int), torch.zeros([1], device=device, dtype=torch.int))

        predictions[i] = bi_pred
        images[i] = image.squeeze(0)
        # pixAcc, mIoU = metric.get()
        # writer.add_image('prediction_'+str(i)+'/11',
        #          image.squeeze(0), 0, dataformats='CHW')
        # for x in range(9):
        #     channel = output[0,x,:,:]
        #     #print(channel.size())
        #     #print('chan', affords[x], 'max' ,channel.max(), 'min', channel.min())
        #     writer.add_image('prediction_'+str(i)+'/'+affords[x],
        #                  channel, 0, dataformats='HW')

        #logger.info("Sample: {:d}, Validation ham_loss: {:.4f}".format(i + 1, hamming_dists[i]))
    ham_loss = torch.mean(hamming_dists)
    ham_max = torch.max(hamming_dists)
    ham_min = torch.min(hamming_dists)

    values, indices = torch.topk(hamming_dists, 5)
    logger.info('hamm: {}'.format(hamming_dists.data))
    logger.info('worst hamming dists: {},  indices: {}'.format(
        values.data, indices.data))
    # for x in indices:
    #     diffs = hamming_diffs[x]
    #     output = predictions[x]
    #     writer.add_image('worst_prediction_'+str(x.item())+'/11',
    #                      images[x], 0, dataformats='CHW')
    #     for y in range(9):
    #         channel = diffs[y, :, :]
    #         channel_pred = output[y, :, :]
    #         #logger.info(channel.size())
    #         #logger.info('chan', affords[x], 'max' ,channel.max(), 'min', channel.min())
    #         # writer.add_image('worst_hamming_'+str(x.item())+'/'+affords[y],
    #         #                  channel, 0, dataformats='HW')
    #         writer.add_image('worst_prediction_'+str(x.item())+'/'+affords[y],
    #                          channel_pred, 0, dataformats='HW')

    logger.info('Validation hamming_avg: {:.4f}, ham_max: {:.4f}, ham_min: {:.4f}'.format(
        ham_loss, ham_max, ham_min))

    values, indices = torch.topk(hamming_dists, 5, largest=False)
    logger.info('best hamming dists: {},  indices: {}'.format(
        values.data, indices.data))
    for x in range(8):
        diffs = hamming_diffs[x]
        output = predictions[x]
        #print(images[x].size(), type(images[x]), np.array(images[x]).shape)
        # out = transforms.ToPILImage()(images[x])
        # out.save('paper_images/'+'hamming_'+str(x.item())+'_11.png')
        writer.add_image('prediction_'+str(x)+'/11',
                         images[x], 0, dataformats='CHW')

        for y in range(9):
            channel = diffs[y, :, :]
            channel_pred = output[y, :, :]
            #print(channel.size())
            #print('chan', affords[x], 'max' ,channel.max(), 'min', channel.min())
            # writer.add_image('hamming_'+str(x.item())+'/'+affords[y],
            #                  channel, 0, dataformats='HW')
            # transforms.ToPILImage()(channel.cpu()).save('paper_images/'+'hamming_'
            #                                             + str(x.item())+'_'+affords[y]+'.png')

            writer.add_image('prediction_'+str(x)+'/'+affords[y],
                             channel_pred, 0, dataformats='HW')

            # transforms.ToPILImage()(channel_pred.cpu()).save('paper_images/'+'best_pred_'
            #                                                  + str(x.item())+'_'+affords[y]+'.png')

    #logger.info('Validation hamming_avg: {:.4f}, ham_max: {:.4f}, ham_min: {:.4f}'.format(ham_loss, ham_max, ham_min))
    writer.close()


def hamming_diff(prediction, target, device):
    #diffs = torch.zeros_like(prediction)
    pre = torch.histc(prediction, bins=40, min=0, max=1)
    #print('pre', pre.data[36:])
    bi_pred = torch.where(prediction > PRED_THRESHOLD, torch.ones(
        [1], device=device, dtype=torch.int), torch.zeros([1], device=device, dtype=torch.int))
    post = torch.histc(bi_pred, bins=2, min=0, max=1)
    #print('post', post.data)
    #print(', min, max predicition', torch.min(prediction).item(), torch.max(prediction).item())
    targ = torch.histc(target, bins=2, min=0, max=1)
    #print('targ', targ.data)
    bi_targ = torch.where(target > 0.5, torch.ones(
        [1], device=device, dtype=torch.int), torch.zeros([1], device=device, dtype=torch.int))

    #size = diffs.size()

    # bi_pred = bi_pred.view(-1)
    # bi_targ = bi_targ.view(-1)

    diffs = torch.where(bi_pred != bi_targ, torch.ones(
        [1], device=device, dtype=torch.int), torch.zeros([1], device=device, dtype=torch.int))
    #print(diffs.size())
    #binsum = torch.bincount(diffs)
    #print('ERROR', torch.histc(diffs, bins=2, min=0, max=1).data)
    #print('bin[1] total error', binsum[1].item(), '/',diffs.size()[0])

    return diffs


def hamming_dist_norm(prediction, target, device='cpu'):
    bi_pred = torch.where(prediction > PRED_THRESHOLD, torch.ones(
        [1], device=device, dtype=torch.int), torch.zeros([1], device=device, dtype=torch.int))
    bi_targ = torch.where(target > 0.5, torch.ones(
        [1], device=device, dtype=torch.int), torch.zeros([1], device=device, dtype=torch.int))

    bi_pred = bi_pred.view(-1)
    bi_targ = bi_targ.view(-1)

    diffs = torch.where(bi_pred != bi_targ, torch.ones(
        [1], device=device, dtype=torch.int), torch.zeros([1], device=device, dtype=torch.int))
    binsum = torch.bincount(diffs)
    #print('bin[1] total error', binsum[1].item(), '/',diffs.size()[0])

    return binsum[1].item() / diffs.size()[0]


def main():
    args = parse_args()
    if "trial_id" not in args:
        print('Need Model Trial ID')
        exit()
    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"

    args.save_pred = True
    if args.save_pred:
        outdir = 'runs/{}_{}'.format(args.model, args.trial_id)
        args.out_dir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    logger = setup_logger("affordance_prediction", args.log_dir, get_rank(),
                          filename='{}_{}_eval_log.txt'.format(args.model, args.trial_id))
    args.logger = logger
    run_eval(args)


if __name__ == '__main__':
    main()
