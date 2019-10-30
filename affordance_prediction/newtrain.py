from dist_helpers import *
from logger import setup_logger, json_dump
from affordancedata import ToTensorBCHW, GameAffordancesDataset
from models import InitialConvModel
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
import torch.nn as nn
import torch
import argparse
import time
import datetime
import os
import shutil
import sys
import uuid


cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)


SEED = 4747
PRED_THRESHOLD = 0.5
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Affordance Assignment Training')
    # model and dataset
    parser.add_argument('--model', type=str, default='conv',
                        choices=['conv'],
                        help='model name (default: conv)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'SGD'],
                        help='optimizer (default: Adam)')
    parser.add_argument('--weight', type=str, default='None',
                        choices=['None', 'Frequency'],
                        help='class weights (default: None)')

    parser.add_argument('--crop-size', type=int, default=180,
                        help='crop image size')
    parser.add_argument('--num-images', type=int, default=2400,
                        help='images in dataset')
    parser.add_argument('--workers', '-j', type=int, default=16,
                        help='dataloader threads')
    parser.add_argument('--k-folds', '-k', type=int, default=0,
                        help='k for k-fold cross-validation')

    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--kernel-size', type=int, default=7, metavar='N',
                        help='input batch size for training (default: 7)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')

    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='trained_models/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='trained_models/logs/',
                        help='Directory for saving logs')
    parser.add_argument('--log-iter', type=int, default=200,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--trial-id', type=str, default=None,
                        help='path to trained model')
    parser.add_argument('--not-best', action='store_true', default=False,
                        help='path doesnt have best')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        #     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        input_transform = transforms.Compose([ToTensorBCHW()])

        aug_dataset = GameAffordancesDataset(
            game='loz', data_dir='app/games', transform=input_transform)
        test_len = int(0.2 * len(aug_dataset))
        args.logger.info(
            f'Dataset loaded containing {len(aug_dataset)} entries. test length: {test_len}')
        train_dataset, test_dataset = random_split(
            aug_dataset, [len(aug_dataset)-test_len, test_len])

        args.iters_per_epoch = len(
            train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        self.k_folds = args.k_folds

        self.train_loader = DataLoader(dataset=train_dataset,
                                       shuffle=True,
                                       batch_size=args.batch_size,
                                       num_workers=args.workers,
                                       pin_memory=True)
        self.test_loader = DataLoader(dataset=test_dataset,
                                      shuffle=False,
                                      batch_size=1,
                                      num_workers=args.workers,
                                      pin_memory=True)

        # create network
        # BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        if args.model == 'conv':
            self.model = InitialConvModel().to(self.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(
                    args.resume, map_location=lambda storage, loc: storage))

        #pos_examples, neg_examples = calculate_weights(aug_dataset)
        if args.weight == 'None':
            pos_weight = torch.ones(9, 224, 256)
        else:
            pos_weight = calculate_weights(aug_dataset)
            #pos_weight = median_freq_weights(aug_dataset)

        self.criterion = torch.nn.BCEWithLogitsLoss(
            reduction='mean', pos_weight=pos_weight).to(self.device)

        if args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=args.lr)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=args.lr)

        # lr scheduling
        # self.lr_scheduler = WarmupPolyLR(self.optimizer,
        #                                  max_iters=args.max_iters,
        #                                  power=0.9,
        #                                  warmup_factor=args.warmup_factor,
        #                                  warmup_iters=args.warmup_iters,
        #                                  warmup_method=args.warmup_method)
        #
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank)

        self.best_pred = 1.0
        self.best_bce = 1.0

    def k_fold(self):
        logger = self.args.logger
        start_time = time.time()
        # model = NeuralNetClassifier(
        #         module=InitialConvModel,
        #         criterion=nn.functional.binary_cross_entropy_with_logits(torch.randn(224, 256, 9), torch.randn(224, 256, 9), reduction='mean'),
        #         optimizer=torch.optim.Adam(self.model.parameters()),
        #         lr=self.args.lr,
        #         max_epochs=self.args.epochs,
        #         device=self.args.device,
        #         batch_size=self.args.batch_size,
        #         train_split=None
        # )

        init_file = save_initial(self.model, self.args)
        logger.info('saved initial weights to {}'.format(init_file))
        dataset = GrayAffordancesDataset(
            image_dir="data/loz/img/", affordances_dir="data/loz/label/", transform=ToTensorBCHW())
        #dataset = Subset(dataset, range(self.args.num_images))
        # input = SliceDataset(dataset, idx=0)
        # target = SliceDataset(dataset, idx=1)
        k_folds = KFold(self.k_folds, shuffle=True, random_state=SEED)
        ham_losses = torch.ones(self.k_folds, dtype=torch.float)
        ham_worsts = torch.ones(self.k_folds, dtype=torch.float)
        for i, (train, test) in enumerate(k_folds.split(dataset)):
            self.model.load_state_dict(torch.load(init_file))
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.args.lr)
            #print(i, len(train), len(test), type(train), type(test), type(train[0]))
            data_train = Subset(dataset, train)
            data_test = Subset(dataset, test)
            self.train_loader = DataLoader(dataset=data_train,
                                           shuffle=True,
                                           batch_size=self.args.batch_size,
                                           num_workers=self.args.workers,
                                           pin_memory=True)
            self.test_loader = DataLoader(dataset=data_test,
                                          shuffle=False,
                                          batch_size=1,
                                          num_workers=self.args.workers,
                                          pin_memory=True)
            ham_loss, ham_max = self.k_fold_train(i+1)
            ham_losses[i] = ham_loss
            ham_worsts[i] = ham_max
            if torch.argmin(ham_losses) == i:
                save_checkpoint(self.model, self.args, is_best=True)
        total_training_time = time.time() - start_time
        total_training_str = str(
            datetime.timedelta(seconds=total_training_time))
        logger.info('End k-fold training {}, max avg ham loss: {:.4f} (fold: {}) || avg ham loss: {:.4f} || Best BCE avg: {:.4f} || total time {}'.format(
            self.args.trial_id, torch.max(ham_losses), torch.argmax(ham_losses), torch.mean(ham_losses), self.best_bce, total_training_str))
        return {'trial_id': self.args.trial_id,
                'ham_avg': torch.mean(ham_losses).item(),
                'loss': torch.max(ham_losses).item(),
                'ham_best': self.best_pred,
                'bce_best': self.best_bce,
                'args': self.args}
        #scores = cross_val_score(model, X=input, y=target, cv=5, scoring=metrics.make_scorer(sk_hamming_loss, greater_is_better=False))
        #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def k_fold_train(self, k):
        logger = self.args.logger
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * \
            self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        # logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))
        logger.info('Start training fold {:d}, Total Epochs: {:d} = Total Iterations {:d}'.format(
            k, epochs, max_iters))

        self.model.train()
        iteration = 0
        for e in range(epochs):
            for i, data in enumerate(self.train_loader):
                iteration = iteration + 1
                #self.lr_scheduler.step()
                images = data['image'].to(self.device, non_blocking=True)
                targets = data['affordances'].to(
                    self.device, non_blocking=True)
                outputs = self.model(images)
                # loss_dict = self.criterion(outputs, targets)
                #
                # losses = sum(loss for loss in loss_dict.values())
                #
                # # reduce losses over all GPUs for logging purposes
                # loss_dict_reduced = reduce_loss_dict(loss_dict)
                # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                eta_seconds = ((time.time() - start_time)
                               / iteration) * (max_iters - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if iteration % log_per_iters == 0 and save_to_disk:
                    logger.info(
                        "k: {:d} || i: {:d}/{:d} || E: {:d}/{:d} || Loss: {:.5f} || Iter Time: {} || Est Time: {}".format(
                            k, iteration, max_iters, e+1, epochs, loss.item(),
                            str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

                if iteration % save_per_iters == 0 and save_to_disk:
                    save_checkpoint(self.model, self.args, is_best=False)

        ham_loss, ham_max = self.validation()
        self.model.train()
        save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(
            datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Trial {}_{} k-fold {:d} Done. Total training time: {} ({:.4f}s / iter). Avg hamming loss: {:.4f}".format(
                self.args.model, self.args.trial_id, k, total_training_str, total_training_time / max_iters, ham_loss))
        return ham_loss, ham_max

    def train(self):
        logger = self.args.logger
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * \
            self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()

        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(
            epochs, max_iters))

        self.model.train()
        iteration = 0
        first_run = True
        for e in range(epochs):
            for i, data in enumerate(self.train_loader):
                iteration = iteration + 1
                #self.lr_scheduler.step()

                images = data['image'].to(self.device, non_blocking=True)
                targets = data['affordances'].to(
                    self.device, non_blocking=True)

                if first_run:
                    logger.info(
                        f'images: {type(images)}, {images.shape}, {type(images[0])}, range: [ {torch.min(images)}, {torch.max(images)}]')
                    logger.info(
                        f'targets: {type(targets)}, {targets.shape}, {type(targets[0])}, range: [ {torch.min(targets)}, {torch.max(targets)}]')

                outputs = self.model(images)

                if first_run:
                    logger.info(
                        f'outputs: {type(outputs)}, {outputs.shape}, {type(outputs[0])}, range: [ {torch.min(outputs)}, {torch.max(outputs)}]')
                # loss_dict = self.criterion(outputs, targets)
                #
                # losses = sum(loss for loss in loss_dict.values())
                #
                # # reduce losses over all GPUs for logging purposes
                # loss_dict_reduced = reduce_loss_dict(loss_dict)
                # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                eta_seconds = ((time.time() - start_time)
                               / iteration) * (max_iters - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                first_run = False
                if iteration % log_per_iters == 0 and save_to_disk:
                    logger.info(
                        "i: {:d}/{:d} || E: {:d}/{:d} || Loss: {:.5f} || Iter Time: {} || Est Time: {}".format(
                            iteration, max_iters, e+1, epochs, loss.item(),
                            str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

                if iteration % save_per_iters == 0 and save_to_disk:
                    save_checkpoint(self.model, self.args, is_best=False)

                if not self.args.skip_val and iteration % val_per_iters == 0:
                    self.validation()
                    self.model.train()

        save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(
            datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Trial {}_{} Done. Total training time: {} ({:.4f}s / iter). Best normalized hamming loss: {:.4f}".format(
                self.args.model, self.args.trial_id, total_training_str, total_training_time / max_iters, self.best_pred))

    def validation(self):
        logger = self.args.logger
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        #self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        #torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        hamming_dists = torch.empty(len(self.test_loader), device=self.device)
        bce_losses = torch.empty(len(self.test_loader), device=self.device)
        for i, data in enumerate(self.test_loader):
            image = data['image'].to(self.device)
            target = data['affordances'].to(self.device)

            with torch.no_grad():
                output = model(image)
            #self.metric.update(output[0], target)
            bce_losses[i] = self.criterion(output, target)

            output = torch.sigmoid(output)
            hamming_dists[i] = hamming_dist_norm(output, target, self.device)
            # pixAcc, mIoU = self.metric.get()
            # logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))
        ham_loss = torch.mean(hamming_dists)
        ham_max = torch.max(hamming_dists)
        ham_min = torch.min(hamming_dists)
        bce_avg = torch.mean(bce_losses)
        if bce_avg < self.best_bce:
            self.best_bce = bce_avg
        logger.info('Validation hamming_avg: {:.4f}, ham_max: {:.4f}, ham_min: {:.4f}, BCE_Avg: {:.4f}'.format(
            ham_loss, ham_max, ham_min, bce_avg))

        new_pred = ham_loss
        if new_pred < self.best_pred:
            print('new best', new_pred)
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.args, is_best)
        synchronize()
        return ham_loss, ham_max


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


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}.pth'.format(args.model, args.trial_id)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_best_model.pth'.format(
            args.model, args.trial_id)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


def save_initial(model, args):
    """Save Initial Weights for k-fold resets"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_init.pth'.format(args.model, args.trial_id)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    return filename


def calculate_weights(dataset):
    pos_examples = torch.zeros(9, 224, 256, dtype=torch.float)
    neg_examples = torch.zeros(9, 224, 256, dtype=torch.float)
    for i in range(len(dataset)):
        image = dataset[i]['image']
        target = dataset[i]['affordances']
        for x in range(9):
            targ_channel = target[x, :, :]
            bi_targ = torch.where(targ_channel > 0.5, torch.ones(
                [1], dtype=torch.int), torch.zeros([1], dtype=torch.int))
            bi_targ = bi_targ.view(-1)
            binsum = torch.bincount(bi_targ)
            if len(binsum) == 2:
                pos_examples[x, :, :] += binsum[1].item()
            neg_examples[x, :, :] += binsum[0].item()
    pos_examples += 0.00001
    ratios = torch.div(neg_examples, pos_examples)
    normalized = ratios / ratios.sum(0).expand_as(ratios)
    c = x / x.sum(0).expand_as(x)
    print('pos_examples: {},\nneg_examples: {},\nNormWeighted Ratios: {}'.format(
        pos_examples[:, 0, 0].data, neg_examples[:, 0, 0].data, ratios[:, 0, 0].data))
    return ratios


def median_freq_weights(dataset):
    frequencies = []
    for i in range(9):
        frequencies.append(torch.empty(0))
    pos_examples = torch.zeros(9, dtype=torch.float)
    total_examples = torch.zeros(9, dtype=torch.float)
    for i in range(len(dataset)):
        image = dataset[i]['image']
        target = dataset[i]['affordances']
        for x in range(9):
            targ_channel = target[x, :, :]
            bi_targ = torch.where(targ_channel > 0.5, torch.ones(
                [1], dtype=torch.int), torch.zeros([1], dtype=torch.int))
            bi_targ = bi_targ.view(-1)
            binsum = torch.bincount(bi_targ)
            if len(binsum) == 2:
                pos_examples[x] += binsum[1].item()
                total_examples[x] += binsum[0].item() + binsum[1].item()
                frequencies[x] = torch.cat([frequencies[x], torch.tensor(
                    [(binsum[1].item()) / (binsum[0].item() + binsum[1].item())])])
    pos_examples += 0.00001

    freq_c = torch.div(pos_examples, total_examples)

    med_frequencies = torch.zeros(9, dtype=torch.float)
    for i in range(9):
        med_frequencies[i] = torch.median(
            frequencies[i], dim=0).values

    pos_weight = torch.div(med_frequencies, freq_c)
    print('MED FREQ WEIGHTS: {}'.format(pos_weight.data))
    output = torch.empty(9, 224, 256, dtype=torch.float)
    for i in range(224):
        for j in range(256):
            output[:, i, j] = pos_weight
    return output


def main():
    args = parse_args()
    args.trial_id = str(uuid.uuid4())[:4]
    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus

    if not args.no_cuda and torch.cuda.is_available():
        #should speed things up when input sizes aren't changing
        cudnn.benchmark = True
        args.device = "cuda"
        args.distributed = num_gpus > 1
    else:
        args.distributed = False
        args.device = "cpu"

    if args.distributed:
        print('init distributed')
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://")
        synchronize()

    #args.lr = args.lr * num_gpus
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger("affordance_prediction", args.log_dir, get_rank(), filename='{}_{}_log.txt'.format(
        args.model, args.trial_id))
    logger.info("Using {} GPUs. Is Distributed? {}".format(
        num_gpus, args.distributed))
    logger.info(f'Using Device: {args.device}')
    logger.info(args)
    args.logger = logger
    trainer = Trainer(args)
    if args.k_folds > 1:
        print('k_fold validation')
        trainer.k_fold()
        print('done k_fold, exiting')
    else:
        trainer.train()
    torch.cuda.empty_cache()


def grid_search():
    args = parse_args()

    trials = []
    while True:
        args.trial_id = str(uuid.uuid4())[:4]
        num_gpus = int(os.environ["WORLD_SIZE"]
                       ) if "WORLD_SIZE" in os.environ else 1
        args.num_gpus = num_gpus

        if not args.no_cuda and torch.cuda.is_available():
            #should speed things up when input sizes aren't changing
            cudnn.benchmark = True
            args.device = "cuda"
            args.distributed = num_gpus > 1
        else:
            args.distributed = False
            args.device = "cpu"
        if args.distributed:
            print('init distributed')
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://")
            synchronize()

        #args.lr = args.lr * num_gpus

        logger = setup_logger("affordance_prediction", args.log_dir, get_rank(), filename='{}_{}_log.txt'.format(
            args.model, args.trial_id))
        logger.info("Using {} GPUs. Is Distributed? {}".format(
            num_gpus, args.distributed))
        logger.info(args)
        args.logger = logger
        trainer = Trainer(args)
        if args.k_folds > 1:
            print('k_fold validation')
            x = trainer.k_fold()
            print(x['loss'], x['bce_best'], x['trial_id'])
            trials.append(x)
            print('done k_fold, exiting')
        else:
            trainer.train()
        args.batch_size = args.batch_size * 2 if args.batch_size * 2 < 100 else 4
        args.lr = args.lr * \
            0.1 if args.lr * 0.1 > 0.00001 else 0.1
        args.optimizer = 'Adam' if args.optimizer == 'SGD' else 'SGD'
        args.weight = 'None' if args.weight == 'Frequency' else 'Frequency'
        #args.epochs = args.epochs + 3 if args.epochs + 3 < 21 else 5
        torch.cuda.empty_cache()


if __name__ == '__main__':
    print(f'Hello docker training world {len(sys.argv)} args')
    main()
