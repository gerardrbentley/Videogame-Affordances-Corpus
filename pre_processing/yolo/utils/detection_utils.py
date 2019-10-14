
from collections import defaultdict, deque
import datetime
import pickle
import time

import torch
import torch.distributed as dist

import errno
import os
import random
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torch.utils.data import Dataset

'''
Orig: https://github.com/pytorch/vision/tree/master/references/detection
'''


class RandomSpritesDataset(Dataset):
    def __init__(self, backgrounds, sprites, sprite_masks, num_images, sprites_per_image=2, transform=None):
        self.background_images = backgrounds
        self.sprite_images = sprites
        self.sprite_masks = sprite_masks
        self.length = num_images
        self.transform = transform
        self.sprites_per_image = sprites_per_image
        self.prev_used = {}
        self.flag = True

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx in self.prev_used.keys():
            if self.transform:
                img_mod, target = self.transform(
                    self.prev_used[idx][0], self.prev_used[idx][1])

            return img_mod, target

        bg_idx = torch.randint(
            low=0, high=len(self.background_images), size=(1,))
        img = self.background_images[bg_idx]
        img_w, img_h = img.size
        img_mod = img.copy()

        boxes = []
        #array of np arrays
        masks = []
        xywh = []
        num_sprites = torch.randint(
            low=3, high=self.sprites_per_image, size=(1,))
        for i in range(num_sprites):

            sprite_idx = torch.randint(
                low=0, high=len(self.sprite_images), size=(1,))
            sprite = self.sprite_images[sprite_idx]
            mask_sprite = self.sprite_masks[sprite_idx]
            mask = np.zeros((img_h, img_w), dtype=np.float32)

            w, h = sprite.size
            new_x = torch.randint(0, img_w-w, (1,))
            new_y = torch.randint(0, img_h-h, (1,))
            paste_box = (new_x, new_y, new_x + w, new_y+h)
            img_mod.paste(sprite, paste_box, mask=mask_sprite)
            mask[new_y:new_y+h, new_x:new_x+w] = 1
            boxes.append([new_x, new_y, new_x + w, new_y + h])
            masks.append(mask)
            yolo_x = (new_x.item() + w/2.0)
            yolo_x *= (1.0/img_w)
            yolo_y = (new_y.item() + h/2.0)
            yolo_y *= (1.0/img_h)
            yolo_w = w * (1.0/img_w)
            yolo_h = h * (1.0/img_h)
            xywh.append([yolo_x, yolo_y, yolo_w, yolo_h])

        n = len(boxes)
        #torch tensor of boxes
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        xywh = torch.as_tensor(xywh, dtype=torch.float32)

        #ones bc only detect sprite class
        labels = torch.ones((num_sprites,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        #from docs
        image_id = torch.tensor([idx], dtype=torch.bool)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_sprites,), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["yolo_xywh"] = xywh

        self.prev_used[idx] = (img_mod, target)

        if self.transform:
            img_mod, target = self.transform(img_mod, target)

        return img_mod, target


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        # print('ToTensor Called, pre: {}, extrema: {}'.format(
        #     type(image), image.getextrema()))
        image = F.to_tensor(image)
        # print('ToTensor Called, post: {}, extrema: {}, {}'.format(
        #     type(image), torch.min(image), torch.max(image)))
        return image, target


class Normalize(object):
    def __call__(self, image, target):
        # print('Normalize Called, pre: {}, extrema: {}, {}'.format(
        #     type(image), torch.min(image), torch.max(image)))
        image = F.normalize(image, [0.485, 0.456, 0.406], [
                            0.229, 0.224, 0.225])
        # print('Normalize Called, post: {}, extrema: {}, {}'.format(
        #     type(image), torch.min(image), torch.max(image)))
        return image, target


def inv_normalize(img): return F.normalize(img,
                                           mean=[-0.485/0.229, -0.456
                                                 / 0.224, -0.406/0.225],
                                           std=[1/0.229, 1/0.224, 1/0.225]
                                           )


def npbbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return img[ymin:ymax+1, xmin:xmax+1]


def multiplot(imgs):
    # settings
    h, w = 224, 224        # for raster image
    nrows, ncols = 1, 2  # array of sub-plots
    figsize = [6, 8]     # figure size, inches

    # prep (x,y) for extra plotting on selected sub-plots
    xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
    ys = np.abs(np.sin(xs))           # absolute of sine

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        img = imgs[i]
        axi.imshow(img)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))

    # one can access the axes by ax[row_id][col_id]
    # do additional plotting on ax[row_id][col_id] of your choice
    # ax[0][2].plot(xs, 3*ys, color='red', linewidth=3)
    # ax[4][3].plot(ys**2, xs, color='green', linewidth=3)

    plt.tight_layout(True)
    plt.show()


class ToTensorBCHW(object):
    """Convert image and affordance to Tensors for input to model
    in the form Batch x Channels x Height x Width """

    def __call__(self, sample):
        image, affordances = sample['image'], sample['affordances']
        fn = F.to_tensor()
        affordances = list(map(fn, affordances))
        stack = torch.stack(affordances, dim=1)
        stack.requires_grad_(False)
        stack = stack.squeeze(0)

        image = fn(image)
        image.requires_grad_(False)
        return {'image': image, 'affordances': stack}


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty(
            (max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,),
                              dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
