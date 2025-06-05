# analysis.py
import torch
import argparse
import os
import numpy as np
import pickle as pkl # 用于加载保存的数据
from collections import defaultdict # 用于更方便地组织数据

# 导入你的模块
import models
import misc
from decision import default_graph, apply_func, set_deterministic_value, DecisionHead, replace_func # 确保导入DecisionHead和replace_func
# 导入模型转换时用到的init_func和new_forward
from decision import init_decision_basicblock, decision_basicblock_forward # 假设是resnet
from decision import init_decision_convbn, decision_convbn_forward # 假设是vgg

# 配置日志
print = misc.logger.info
np.set_printoptions(precision=4, linewidth=160)

# --- 1. 参数解析 ---
parser = argparse.ArgumentParser(description="Analyze DecisionHead behaviors after finetuning.")
parser.add_argument('--gpu', default='0', type=str, help='GPU ID to use (e.g., 0, 1)')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use: cifar10 or cifar100')
parser.add_argument('--arch', '-a', default='resnet56', type=str, help='Model architecture: resnet20, resnet56, vgg16, vgg19')
parser.add_argument('--action_num', default=40, type=int, help='Number of actions for decision head')
parser.add_argument('--sparsity_level', default=0.7, type=float, help='Target sparsity level used during training')
parser.add_argument('--batch_size', default=100, type=int, help='Batch size for analysis (testloader batch size)')

args = parser.parse_args()

args.num_classes = 10 if args.dataset == 'cifar10' else 100
args.device = 'cuda'
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# 确保日志路径与finetune阶段一致，以便加载模型和保存分析结果
args.logdir = 'finetune-decision-%d/%s-%s/sparsity-%.2f' % (
    args.action_num, args.dataset, args.arch, args.sparsity_level
)
misc.prepare_logging(args) # 使用misc的日志功能

# --- 2. 数据加载 (仅用于获取测试集和标签) ---
print('==> Preparing data for analysis..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'cifar10':
    from torchvision import datasets, transforms
    testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    from torchvision import datasets, transforms
    testset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
else:
    raise ValueError(f"Dataset {args.dataset} not supported.")

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# --- 3. 模型初始化与加载 ---
print('==> Initializing model for analysis...')
model = models.__dict__['cifar_' + args.arch](args.num_classes)

# 改造模型，插入DecisionHead并替换forward
if args.arch in ['resnet20', 'resnet56']:
    init_func = init_decision_basicblock
    new_forward = decision_basicblock_forward
    module_type = 'BasicBlock'
else:
    init_func = init_decision_convbn
    new_forward = decision_convbn_forward
    module_type = 'ConvBNReLU'

# 注意：这里传递的temperature值不重要，因为分析阶段是deterministic=True
apply_func(model, module_type, init_func, action_num=args.action_num, temperature=1.0)
replace_func(model, module_type, new_forward)

print('==> Loading finetuned model...')
model_path = os.path.join(args.logdir, 'checkpoint.pth') # 加载finetune阶段保存的模型
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Finetuned model not found at {model_path}. Please run finetune.py first.")

ckpt_state_dict = torch.load(model_path, map_location=args.device)
model.load_state_dict(ckpt_state_dict)
model = model.to(args.device)
model.eval() # 设置为评估模式
apply_func(model, 'DecisionHead', set_deterministic_value, deterministic=True) # 确保决策头是确定性模式

# --- 4. 收集数据 ---
print('==> Collecting decision data...')

# 建立DecisionHead实例到其名称和索引的映射
decision_head_map = {} # {id(dh_instance): 'Layer_X_ModuleName'}
layer_idx = 0
for name, module in model.named_modules():
    if hasattr(module, 'decision_head') and isinstance(module.decision_head, DecisionHead):
        layer_idx += 1
        layer_name = f"Layer_{layer_idx}_{name}" # 示例名称：Layer_1_model.layer1.0.decision_head
        decision_head_map[id(module.decision_head)] = layer_name
        
# 存储收集到的数据: {layer_name: {'sampled_actions': [], 'action_probs': []}}
collected_decision_data = defaultdict(lambda: defaultdict(list))
all_labels = []

with torch.no_grad():
    for i, (data, target) in enumerate(testloader):
        default_graph.clear_all_tensors() # 清空default_graph以记录当前batch

        data = data.to(args.device)
        target = target.to(args.device)
        _ = model(data) # 运行前向传播，DecisionHead会将数据存储到default_graph

        # 从default_graph中提取并存储
        for dh_id, dh_name in decision_head_map.items():
            sampled_actions_list = default_graph.get_tensor_list(f'sampled_actions_{dh_id}')
            action_probs_list = default_graph.get_tensor_list(f'action_probs_{dh_id}')

            if sampled_actions_list:
                # sampled_actions_list 和 action_probs_list 应该只包含一个元素（当前batch的数据）
                collected_decision_data[dh_name]['sampled_actions'].append(sampled_actions_list[0].cpu())
                collected_decision_data[dh_name]['action_probs'].append(action_probs_list[0].cpu())
        
        all_labels.append(target.cpu())

all_labels = torch.cat(all_labels, dim=0).numpy()


# --- 5. 最终处理和保存 ---
final_analysis_results = {}

# a. 提取并保存 channel_gates
print('==> Extracting final channel_gates...')
final_channel_gates_by_layer = {}
for dh_id, dh_name in decision_head_map.items():
    # 再次遍历模型，获取 DecisionHead 实例的引用
    # 这里通过 model.named_modules() 更稳定地获取引用
    for name, module in model.named_modules():
        if id(module) == dh_id: # 找到对应的DecisionHead实例
            if hasattr(module, 'channel_gates'): # 确保是DecisionHead本身
                channel_gates_tensor = module.channel_gates.data.cpu().numpy()
                final_channel_gates_by_layer[dh_name] = channel_gates_tensor
                print(f"  {dh_name} Channel Gates (Shape: {channel_gates_tensor.shape}):\n{channel_gates_tensor[:5]}") # 打印前5行
                break # 找到就跳出内层循环

misc.dump_pickle(final_channel_gates_by_layer, os.path.join(args.logdir, 'final_channel_gates.pkl'))
print(f'Final channel gates saved to {os.path.join(args.logdir, "final_channel_gates.pkl")}')


# b. 处理并分析不同类别样本的决策头选择概率权重
print('==> Analyzing action probabilities by class...')
action_probs_by_layer_and_class = {}

for dh_name, stats in collected_decision_data.items():
    if not stats['sampled_actions']: # 如果没有数据，跳过
        continue

    # 合并所有batch的数据
    all_sampled_actions = torch.cat(stats['sampled_actions'], dim=0).numpy()
    all_action_probs = torch.cat(stats['action_probs'], dim=0).numpy()

    action_probs_by_layer_and_class[dh_name] = {}
    
    print(f"\n--- Analysis for {dh_name} ---")
    
    # 总体动作频率
    total_action_counts = np.bincount(all_sampled_actions, minlength=args.action_num)
    print(f"  Overall Action Frequencies: {total_action_counts / all_sampled_actions.shape[0]}")

    # 按类别分析
    for class_id in range(args.num_classes):
        class_indices = np.where(all_labels == class_id)[0]
        if len(class_indices) == 0:
            continue

        class_sampled_actions = all_sampled_actions[class_indices]
        class_action_probs = all_action_probs[class_indices]

        # 统计每个类别选择每个动作的频率
        class_action_counts = np.bincount(class_sampled_actions, minlength=args.action_num)
        class_action_frequencies = class_action_counts / len(class_indices)
        
        # 计算每个类别在每个动作上的平均概率（更软性的指标）
        mean_action_probs = class_action_probs.mean(axis=0)

        action_probs_by_layer_and_class[dh_name][class_id] = {
            'action_frequencies': class_action_frequencies,
            'mean_action_probs': mean_action_probs
        }
        
        print(f"  Class {class_id} (N={len(class_indices)}):")
        print(f"    Action Frequencies: {class_action_frequencies}")
        print(f"    Mean Action Probs:  {mean_action_probs}")

misc.dump_pickle(action_probs_by_layer_and_class, os.path.join(args.logdir, 'action_probs_by_class.pkl'))
print(f'Action probabilities by class saved to {os.path.join(args.logdir, "action_probs_by_class.pkl")}')

print('\nAnalysis complete.')
