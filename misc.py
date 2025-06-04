from PIL import Image
import os
import shutil
import pickle as pkl
import time
from datetime import datetime
import logging # <-- 确保这里导入了logging
import sys # <-- 导入sys

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Logger(object):
    def __init__(self):
        self._logger = None
        self._initialized = False # <-- 新增标志位

    def init(self, logdir, name='log'):
        if not self._initialized: # <-- 只有在未初始化时才执行初始化逻辑
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file) # 清空旧日志文件
            
            # 使用logging.getLogger(__name__)或指定一个名字，避免获取root logger可能导致的冲突
            # 使用 logdir 命名 logger，确保在不同 logdir 时是不同的 logger 实例
            self._logger = logging.getLogger(f"my_logger_{logdir.replace('/', '_')}")
            self._logger.setLevel('INFO')
            self._logger.propagate = False # 避免日志信息传递给 root logger，导致重复输出

            # 避免重复添加 handler
            if not self._logger.handlers: # 只有在没有 handler 时才添加
                fh = logging.FileHandler(log_file)
                # 将 StreamHandler 的输出目标设置为 sys.stdout，避免默认的红色 stderr
                ch = logging.StreamHandler(sys.stdout) 
                formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
                fh.setFormatter(formatter)
                ch.setFormatter(formatter)
                self._logger.addHandler(fh)
                self._logger.addHandler(ch)
            
            self._initialized = True # 标记为已初始化

    def info(self, str_info):
        # 假设 prepare_logging 已经调用了 init，这里不再无条件调用 init
        # 如果在 prepare_logging 之前调用了 info，可以临时初始化到一个默认位置
        if self._logger is None: 
            self.init(os.path.expanduser('~/tmp_log'), 'tmp.log')
        
        self._logger.info(str_info)


logger = Logger()


def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)


def load_pickle(path, verbose=True):
    begin_st = time.time()
    with open(path, 'rb') as f:
        if verbose:
            print("Loading pickle object from {}".format(path))
        v = pkl.load(f)
    if verbose:
        print("=> Done ({:.4f} s)".format(time.time() - begin_st))
    return v


def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        print("Dumping pickle object to {}".format(path))
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


def prepare_logging(args):
    args.logdir = os.path.join('./logs', args.logdir)
    ensure_dir(args.logdir) # 确保目录先创建好
    logger.init(args.logdir, 'log') # 在这里只调用一次初始化
    
    logger.info("=================FLAGS==================")
    for k, v in args.__dict__.items():
        logger.info('{}: {}'.format(k, v))
    logger.info("========================================")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
