from torchvision import transforms, datasets
import torch.nn.functional as F
import numpy as np
import torch
import argparse
import os

from decision import default_graph, apply_func, replace_func, \
    collect_params, set_deterministic_value, normalize_head_weights
import models
import misc

np.set_printoptions(precision=2, linewidth=160)
print = misc.logger.info

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--arch', '-a', default='resnet56', type=str)
parser.add_argument('--action_num', default=40, type=int)
parser.add_argument('--sparsity_level', default=0.7, type=float)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--lambd', default=0.5, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--train_batch_size', default=128, type=int)

args = parser.parse_args()

args.num_classes = 10 if args.dataset == 'cifar10' else 100

args.device = 'cuda'
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.logdir = 'decision-%d/%s-%s/sparsity-%.2f' % (
    args.action_num, args.dataset, args.arch, args.sparsity_level
)
misc.prepare_logging(args)

print('==> Preparing data..')

if args.dataset == 'cifar10':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

elif args.dataset == 'cifar100':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

print('==> Initializing model...')
model = models.__dict__['cifar_' + args.arch](args.num_classes)
model_params = []
for p in model.parameters():
    model_params.append(p)

print('==> Loading pretrained model...')
model.load_state_dict(torch.load('logs/pretrained/%s/%s/checkpoint.pth' % (args.dataset, args.arch)))

if args.arch in ['resnet20', 'resnet56']:
    from decision import init_decision_basicblock, decision_basicblock_forward
    init_func = init_decision_basicblock
    new_forward = decision_basicblock_forward
    module_type = 'BasicBlock'

else:
    from decision import init_decision_convbn, decision_convbn_forward
    init_func = init_decision_convbn
    new_forward = decision_convbn_forward
    module_type = 'ConvBNReLU'

print('==> Transforming model...')

apply_func(model, module_type, init_func, action_num=args.action_num)
apply_func(model, 'DecisionHead', collect_params)
replace_func(model, module_type, new_forward)
apply_func(model, 'DecisionHead', normalize_head_weights)
model = model.to(args.device)

head_params = default_graph.get_tensor_list('head_params')
gate_params = default_graph.get_tensor_list('gate_params')

optimizer_gate = torch.optim.Adam(head_params + gate_params, lr=args.lr)
optimizer_model = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=1e-9)

def train(epoch):
    model.train()
    apply_func(model, 'DecisionHead', set_deterministic_value, deterministic=False)
    for i, (data, target) in enumerate(trainloader):
        default_graph.clear_all_tensors()

        data = data.to(args.device)
        target = target.to(args.device)

        optimizer_gate.zero_grad()
        output = model(data)
        loss_ce = F.cross_entropy(output, target)
        selected_channels = default_graph.get_tensor_list('selected_channels')
        loss_reg = args.lambd * (torch.cat(selected_channels, dim=1).abs().mean() - args.sparsity_level) ** 2
        loss = loss_ce + loss_reg

        loss.backward()
        optimizer_gate.step()

        for p in gate_params:
            p.data.clamp_(0, 5)

        apply_func(model, 'DecisionHead', normalize_head_weights)

        optimizer_model.zero_grad()
        output = model(data)
        loss_model = F.cross_entropy(output, target)
        loss_model.backward()
        optimizer_model.step()

        if i % args.log_interval == 0:
            concat_channels = torch.cat(selected_channels, dim=1)
            sparsity = (concat_channels != 0).float().mean()
            mean_gate = concat_channels.mean()
            acc = (output.max(1)[1] == target).float().mean()

            print('Train Epoch: %d [%d/%d]\tLoss: %.4f, Loss_CE: %.4f, Loss_REG: %.4f, '
                  'Sparsity: %.4f, Mean gate: %.4f, Accuracy: %.4f' % (
                epoch, i, len(trainloader), loss.item(), loss_ce.item(), loss_reg.item(),
                sparsity.item(), mean_gate.item(), acc.item()
            ))

def test():
    model.eval()
    apply_func(model, 'DecisionHead', set_deterministic_value, deterministic=True)
    test_loss_ce = []
    test_loss_reg = []
    test_sparsity = []
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            default_graph.clear_all_tensors()

            data, target = data.to(args.device), target.to(args.device)
            output = model(data)

            selected_channels = default_graph.get_tensor_list('selected_channels')
            concat_channels = torch.cat(selected_channels, dim=1)

            test_loss_ce.append(F.cross_entropy(output, target).item())
            test_loss_reg.append(args.lambd * concat_channels.abs().sum().item())
            test_sparsity.append((concat_channels != 0).float().mean().item())

            pred = output.max(1)[1]
            correct += (pred == target).float().sum().item()

    actions = torch.stack(default_graph.get_tensor_list('sampled_actions')).permute(1, 0)
    acc = correct / len(testloader.dataset)
    print('Test set: Loss: %.4f, Loss_CE: %.4f, Loss_REG: %.4f, '
          'Sparsity: %.4f, Accuracy: %.4f' % (
        np.mean(test_loss_ce) + np.mean(test_loss_reg), np.mean(test_loss_ce), np.mean(test_loss_reg),
        np.mean(test_sparsity), acc
    ))
    print('   First 10 sampled actions: \n' + str(actions[:10].cpu().numpy()))
    print('   First 10 targets: ' + str(target[:10].cpu().numpy()) + '\n')
    return acc, np.mean(test_sparsity)


def save_checkpoint(state, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))


for epoch in range(args.epochs):
    train(epoch)
    acc, sparsity = test()

    if sparsity <= args.sparsity_level:
        print('Save best checkpoint @ Epoch %d, Accuracy = %.4f, Sparsity = %.4f\n' % (
            epoch, acc, sparsity
        ))

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, filepath=args.logdir)



from torchvision import transforms, datasets
import torch.nn.functional as F
import numpy as np
import torch
import argparse
import os

from decision import default_graph, apply_func, replace_func, \
    collect_params, set_deterministic_value, normalize_head_weights, set_temperature_value # 导入新的 set_temperature_value
import models
import misc

np.set_printoptions(precision=2, linewidth=160)
print = misc.logger.info

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str, help='GPU ID to use (e.g., 0, 1)')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use: cifar10 or cifar100')
parser.add_argument('--arch', '-a', default='resnet56', type=str, help='Model architecture: resnet20, resnet56, vgg16, vgg19')
parser.add_argument('--action_num', default=40, type=int, help='Number of actions for decision head')
parser.add_argument('--sparsity_level', default=0.7, type=float, help='Target sparsity level (e.g., 0.7 for 70% pruned)')
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate for optimization')
parser.add_argument('--lambd', default=0.5, type=float, help='Weight for sparsity regularization loss')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs for training') # 周期改为10
parser.add_argument('--log_interval', default=100, type=int, help='Logging interval (batches)')
parser.add_argument('--train_batch_size', default=128, type=int, help='Batch size for training')
# Gumbel-softmax 温度退火参数
parser.add_argument('--temp_start', default=5.0, type=float, help='Starting temperature for Gumbel-softmax annealing')
parser.add_argument('--temp_end', default=0.5, type=float, help='Ending temperature for Gumbel-softmax annealing')

args = parser.parse_args()

args.num_classes = 10 if args.dataset == 'cifar10' else 100

args.device = 'cuda'
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.logdir = 'decision-%d/%s-%s/sparsity-%.2f' % (
    args.action_num, args.dataset, args.arch, args.sparsity_level
)
misc.prepare_logging(args)

print('==> Preparing data..')

if args.dataset == 'cifar10':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

elif args.dataset == 'cifar100':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

print('==> Initializing model...')
model = models.__dict__['cifar_' + args.arch](args.num_classes)
model_params = []
for p in model.parameters():
    model_params.append(p)

print('==> Loading pretrained model...')
pretrained_model_path = 'logs/pretrained/%s/%s/checkpoint.pth' % (args.dataset, args.arch)
if not os.path.exists(pretrained_model_path):
    # 如果.pth文件不存在，尝试加载.pth.tar文件
    pretrained_model_path = 'logs/pretrained/%s/%s/checkpoint.pth.tar' % (args.dataset, args.arch)
    if os.path.exists(pretrained_model_path):
        ckpt_data = torch.load(pretrained_model_path)
        model.load_state_dict(ckpt_data['state_dict']) # 如果保存的是字典，需要取state_dict
    else:
        raise FileNotFoundError(f"Pretrained model not found at {pretrained_model_path} or its .tar version.")
else:
    model.load_state_dict(torch.load(pretrained_model_path))

if args.arch in ['resnet20', 'resnet56']:
    from decision import init_decision_basicblock, decision_basicblock_forward
    init_func = init_decision_basicblock
    new_forward = decision_basicblock_forward
    module_type = 'BasicBlock'

else:
    from decision import init_decision_convbn, decision_convbn_forward
    init_func = init_decision_convbn
    new_forward = decision_convbn_forward
    module_type = 'ConvBNReLU'

print('==> Transforming model...')

# 传递起始温度给 init_func，作为 DecisionHead 的初始温度
apply_func(model, module_type, init_func, action_num=args.action_num, temperature=args.temp_start)
apply_func(model, 'DecisionHead', collect_params)
replace_func(model, module_type, new_forward)
apply_func(model, 'DecisionHead', normalize_head_weights)
model = model.to(args.device)


head_params = default_graph.get_tensor_list('head_params')
gate_params = default_graph.get_tensor_list('gate_params')

optimizer_gate = torch.optim.Adam(head_params + gate_params, lr=args.lr)
optimizer_model = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=1e-9)


def train(epoch):
    model.train()
    apply_func(model, 'DecisionHead', set_deterministic_value, deterministic=False)
    for i, (data, target) in enumerate(trainloader):
        default_graph.clear_all_tensors()

        data = data.to(args.device)
        target = target.to(args.device)

        optimizer_gate.zero_grad()
        output = model(data)
        loss_ce = F.cross_entropy(output, target)
        selected_channels = default_graph.get_tensor_list('selected_channels')
        loss_reg = args.lambd * (torch.cat(selected_channels, dim=1).abs().mean() - args.sparsity_level) ** 2
        loss = loss_ce + loss_reg

        loss.backward()
        optimizer_gate.step()

        for p in gate_params:
            p.data.clamp_(0, 5)

        apply_func(model, 'DecisionHead', normalize_head_weights)

        optimizer_model.zero_grad()
        output = model(data)
        loss_model = F.cross_entropy(output, target)
        loss_model.backward()
        optimizer_model.step()

        if i % args.log_interval == 0:
            concat_channels = torch.cat(selected_channels, dim=1)
            sparsity = (concat_channels != 0).float().mean()
            mean_gate = concat_channels.mean()
            acc = (output.max(1)[1] == target).float().mean()

            print('Train Epoch: %d [%d/%d]\tLoss: %.4f, Loss_CE: %.4f, Loss_REG: %.4f, '
                  'Sparsity: %.4f, Mean gate: %.4f, Accuracy: %.4f' % (
                      epoch, i, len(trainloader), loss.item(), loss_ce.item(), loss_reg.item(),
                      sparsity.item(), mean_gate.item(), acc.item()
                  ))

def test():
    model.eval()
    apply_func(model, 'DecisionHead', set_deterministic_value, deterministic=True)
    test_loss_ce = []
    test_loss_reg = []
    test_sparsity = []
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            default_graph.clear_all_tensors()

            data, target = data.to(args.device), target.to(args.device)
            output = model(data)

            selected_channels = default_graph.get_tensor_list('selected_channels')
            concat_channels = torch.cat(selected_channels, dim=1)

            test_loss_ce.append(F.cross_entropy(output, target).item())
            test_loss_reg.append(args.lambd * concat_channels.abs().sum().item())
            test_sparsity.append((concat_channels != 0).float().mean().item())

            pred = output.max(1)[1]
            correct += (pred == target).float().sum().item()

    all_sampled_actions = default_graph.get_tensor_list('sampled_actions')
    actions = torch.stack(all_sampled_actions).permute(1, 0) if all_sampled_actions else torch.tensor([])

    acc = correct / len(testloader.dataset)
    print('Test set: Loss: %.4f, Loss_CE: %.4f, Loss_REG: %.4f, '
          'Sparsity: %.4f, Accuracy: %.4f' % (
              np.mean(test_loss_ce) + np.mean(test_loss_reg), np.mean(test_loss_ce), np.mean(test_loss_reg),
              np.mean(test_sparsity), acc
          ))
    if actions.numel() > 0:
        print('    First 10 sampled actions: \n' + str(actions[:min(10, actions.shape[0])].cpu().numpy()))
        print('    First 10 targets: ' + str(target[:min(10, target.shape[0])].cpu().numpy()) + '\n')
    else:
        print('    No sampled actions to display (graph might be cleared or decision head not activated).\n')
    return acc, np.mean(test_sparsity)


def save_checkpoint(state, filepath):
    torch.save(state, os.path.join(filepath, 'model.pth.tar'))


for epoch in range(args.epochs):
    # === 温度退火逻辑 ===
    # 计算当前 epoch 的温度
    if args.epochs > 1: # 避免除以零
        current_temperature = args.temp_start - (args.temp_start - args.temp_end) * (epoch / (args.epochs - 1))
    else: # 如果只有一个epoch，温度保持起始值
        current_temperature = args.temp_start
    
    # 将当前温度设置到模型中所有 DecisionHead 实例的 temperature 属性
    apply_func(model, 'DecisionHead', set_temperature_value, temperature=current_temperature)
    print(f'Temperature for epoch {epoch}: {current_temperature:.4f}') # 打印当前温度

    train(epoch) # 训练一个 epoch
    acc, sparsity = test() # 测试模型性能

    # main.py 的保存条件：仅当稀疏度达标时保存
    if sparsity <= args.sparsity_level:
        print('Save best checkpoint @ Epoch %d, Accuracy = %.4f, Sparsity = %.4f\n' % (
            epoch, acc, sparsity
        ))
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, filepath=args.logdir)

print('Training complete.')
