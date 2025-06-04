

def train(epoch):
    model.train()
    apply_func(model, 'DecisionHead', set_deterministic_value, deterministic=True)
    for i, (data, target) in enumerate(trainloader):
        default_graph.clear_all_tensors()

        data = data.to(args.device)
        target = target.to(args.device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            selected_channels = default_graph.get_tensor_list('selected_channels')
            concat_channels = torch.cat(selected_channels, dim=1)
            sparsity = (concat_channels != 0).float().mean()
            acc = (output.max(1)[1] == target).float().mean()

            print('Train Epoch: %d [%d/%d]\tLoss: %.4f, '
                  'Sparsity: %.4f, Accuracy: %.4f' % (
                epoch, i, len(trainloader), loss.item(),
                sparsity.item(), acc.item()
            ))

def test():
    model.eval()
    apply_func(model, 'DecisionHead', set_deterministic_value, deterministic=True)
    test_loss = []
    test_sparsity = []
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            default_graph.clear_all_tensors()

            data, target = data.to(args.device), target.to(args.device)
            output = model(data)

            selected_channels = default_graph.get_tensor_list('selected_channels')
            concat_channels = torch.cat(selected_channels, dim=1)

            test_loss.append(F.cross_entropy(output, target).item())
            test_sparsity.append((concat_channels != 0).float().mean().item())

            pred = output.max(1)[1]
            correct += (pred == target).float().sum().item()

    acc = correct / len(testloader.dataset)
    print('Test set: Loss: %.4f, '
          'Sparsity: %.4f, Accuracy: %.4f\n' % (
        np.mean(test_loss), np.mean(test_sparsity), acc
    ))
    return acc, np.mean(test_sparsity)


for epoch in range(args.epochs):
    train(epoch)
    acc, sparsity = test()
    torch.save(model.state_dict(), os.path.join(args.logdir, 'checkpoint.pth'))
    print('Save checkpoint @ Epoch %d, Accuracy = %.4f, Sparsity = %.4f\n' % (
        epoch, acc, sparsity
    ))



from torchvision import transforms, datasets
import torch.nn.functional as F
import numpy as np
import torch
import argparse
import os
import pickle as pkl # 用于保存桑基图数据

from decision import default_graph, apply_func, replace_func, set_deterministic_value, DecisionHead # 导入 DecisionHead
import models
import misc

print = misc.logger.info

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--arch', '-a', default='resnet56', type=str)
parser.add_argument('--action_num', default=40, type=int)
parser.add_argument('--sparsity_level', default=0.7, type=float)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--epochs', default=16, type=int) # 周期改为16
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--train_batch_size', default=128, type=int)

args = parser.parse_args()

args.num_classes = 10 if args.dataset == 'cifar10' else 100

args.device = 'cuda'
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.logdir = 'finetune-decision-%d/%s-%s/sparsity-%.2f' % (
    args.action_num, args.dataset, args.arch, args.sparsity_level
)
misc.prepare_logging(args)

print('==> Preparing data..')

if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

elif args.dataset == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

print('==> Initializing model...')
model = models.__dict__['cifar_' + args.arch](args.num_classes)
model_params = []
for p in model.parameters():
    model_params.append(p)

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

# 这里传递的 temperature 值不重要，因为 finetune 阶段是 deterministic=True
apply_func(model, module_type, init_func, action_num=args.action_num, temperature=args.action_num) # 这里的 temperature 可以是任意值，如 1.0
replace_func(model, module_type, new_forward)

print('==> Loading pretrained decision model...')
# finetune.py 从 main.py 保存的模型中加载权重
ckpt = torch.load(
    'logs/decision-%d/%s-%s/sparsity-%.2f/model.pth.tar' % (
        args.action_num, args.dataset, args.arch, args.sparsity_level
))
model.load_state_dict(ckpt['state_dict'])

model = model.to(args.device)

optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

def train(epoch):
    model.train()
    apply_func(model, 'DecisionHead', set_deterministic_value, deterministic=True) # 微调阶段，决策头是确定性的
    for i, (data, target) in enumerate(trainloader):
        default_graph.clear_all_tensors()

        data = data.to(args.device)
        target = target.to(args.device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            selected_channels = default_graph.get_tensor_list('selected_channels')
            concat_channels = torch.cat(selected_channels, dim=1)
            sparsity = (concat_channels != 0).float().mean()
            acc = (output.max(1)[1] == target).float().mean()

            print('Train Epoch: %d [%d/%d]\tLoss: %.4f, '
                  'Sparsity: %.4f, Accuracy: %.4f' % (
                      epoch, i, len(trainloader), loss.item(),
                      sparsity.item(), acc.item()
                  ))

# 新增：收集桑基图数据的函数
def collect_sankey_data(model, module_type, input_channels_initial):
    """
    遍历模型，收集每个 DecisionHead 模块的剪枝信息。
    返回一个列表，包含每层剪枝后的通道数量和原始通道数量。
    """
    sankey_data = []
    
    # 桑基图的第一个节点是输入层
    sankey_data.append({'layer_name': 'Input', 'active_channels': input_channels_initial, 'total_channels': input_channels_initial})

    # 递归遍历模型模块
    layer_idx = 0
    for name, module in model.named_modules():
        # 针对 BasicBlock 或 ConvBNReLU 模块进行处理
        if module.__class__.__name__ == module_type:
            layer_idx += 1
            layer_name = f'L{layer_idx}' # 简单的层命名方式
            
            # 确保模块有 decision_head 属性
            if hasattr(module, 'decision_head') and isinstance(module.decision_head, DecisionHead):
                # 在确定性模式下，decision_head.channel_gates 包含了固定剪枝决策
                # sampled_actions 是一个 batch 的结果，我们希望得到一个总体的决策
                # 最简单的方法是检查 channel_gates 中哪些行的均值或最大值很高
                # 或者直接取 decision_head.channel_gates 中，对应每个 action 概率最高的行的值。
                # 考虑到 finetune 阶段是 deterministic，我们可以直接取 DecisionHead.channel_gates
                
                # 获取决策头的权重，它决定了选择哪个通道组合
                # 需要知道哪个action被最常选择，或者直接使用最接近1的channel_gates
                # 对于桑基图，我们只需要最终的稀疏度
                
                # 最直接的方法是运行一个 batch 的数据，然后捕获 selected_channels
                # 或者，如果决策头是确定的，并且 channel_gates 已经学好了，
                # 我们可以推断出剪枝后的通道数
                
                # 由于 DecisionHead.forward 会返回 selected_channels，
                # 我们在 test 函数中运行一次前向传播后，可以收集到这个信息。
                
                # 在这里，我们假设 selected_channels 已经从 default_graph 中收集到了
                # 并且它们是 batch_size x out_channels 的张量
                # 为了简化，我们取它们的平均值，并设定一个阈值来判断通道是否被保留
                
                # 这是一个示意性的收集方法，具体取决于你的 DecisionHead 如何输出最终通道权重
                # 最准确的方式是使用 test() 函数中的 selected_channels
                # 由于 test() 函数已经清除了 default_graph，
                # 所以我们不能直接在这里获取，需要测试函数运行一次后返回这些信息。
                
                # 桑基图需要知道每一层（或每一块）的输入通道数和输出通道数。
                # BasicBlock 的输入通道是 conv1.in_channels, 输出是 conv1.out_channels
                # ConvBNReLU 的输入通道是 conv.in_channels, 输出是 conv.out_channels
                
                # 原始总通道数
                total_out_channels = module.decision_head.out_channels
                
                # ！！！注意：这里无法直接获取当前层的 active_channels，
                # 因为它需要实际运行 forward 并查看 selected_channels。
                # 我们需要修改 test() 函数，让它返回每一层剪枝后的通道数。
                # 或者，我们可以重新跑一个测试循环，专门用来收集这个数据。
                
                # 暂时返回占位符，待 test 函数补充
                sankey_data.append({
                    'layer_name': layer_name,
                    'input_channels': module.decision_head.in_channels, # 输入通道数
                    'output_channels': total_out_channels, # 原始输出通道数
                    'pruned_channels_info': None # 待填充，例如 [batch_size, out_channels]
                })
    return sankey_data


def test():
    model.eval()
    apply_func(model, 'DecisionHead', set_deterministic_value, deterministic=True)
    test_loss = []
    test_sparsity = []
    correct = 0
    
    # 用于收集每层剪枝后通道信息的列表
    # 每个元素将是一个 (layer_name, selected_channels_tensor)
    per_layer_selected_channels = {} 
    
    with torch.no_grad():
        for i, (data, target) in enumerate(testloader): # 遍历一个 batch 来收集数据
            default_graph.clear_all_tensors() # 清空非持久化的张量

            data, target = data.to(args.device), target.to(args.device)
            output = model(data)

            # 收集所有 DecisionHead 模块的 selected_channels
            # 注意：default_graph.get_tensor_list('selected_channels') 
            # 是所有 DecisionHead 模块在一个batch中产生的 selected_channels 的 concat 列表
            # 我们需要区分是哪个 DecisionHead 产生的
            
            # 为了准确获取每层的剪枝结果，我们需要遍历模型的模块
            layer_idx = 0
            for name, module in model.named_modules():
                if hasattr(module, 'decision_head') and isinstance(module.decision_head, DecisionHead):
                    layer_idx += 1
                    # 在确定性模式下，DecisionHead(x) 会返回 sampled_actions, selected_channels
                    # 我们可以通过再次运行它来获取，或者依赖 forward hook
                    # 但由于 DecisionHead.forward 会把 selected_channels 放到 default_graph
                    # 并且 default_graph 是在每个 batch 清空的，
                    # 我们需要确保这里获取的是当前 batch 的完整信息。
                    
                    # 更好的方法是在 DecisionHead.forward 中也把 layer_name 或 id 存入 default_graph
                    # 或者，我们可以直接通过模块实例获取，但需要知道如何从 DecisionHead 内部获取其最终决策
                    
                    # 最简单的方法是在 test() 函数中只运行一个 batch
                    # 并从 default_graph.get_tensor_list('selected_channels') 中
                    # 按照 DecisionHead 出现的顺序进行分割和映射
                    
                    # 假设 `selected_channels` 列表中元素的顺序与模块遍历的顺序一致
                    # 并且每个元素对应一个 DecisionHead 的输出
                    # TODO: 需要验证 default_graph.get_tensor_list('selected_channels') 的顺序是否可靠
                    # 如果不可靠，需要修改 decision.py 让它存入带 ID 的信息
                    
                    # 假设 selected_channels 列表的第 k 个元素对应第 k 个 DecisionHead
                    # 并且这个元素是 batch_size x out_channels
                    
                    # 为了桑基图，我们通常只需要一个代表性的结果，例如第一个 batch 的平均值
                    # 或者，对 testloader 中的所有 batch 运行，然后对 `selected_channels` 求平均
                    # 为了简化，我们只取第一个 batch 的结果作为示例
                    if i == 0: # 只处理第一个 batch 的数据
                        current_selected_channels_for_layer = None # 占位符
                        # 假设 default_graph.get_tensor_list('selected_channels') 是按顺序填充的
                        # 那么可以通过索引获取
                        # 这是一个复杂的地方，因为 default_graph 返回的是一个扁平列表
                        # 确切的实现需要对 decision.py 和 model 的结构有非常清楚的了解
                        
                        # 简单地，我们可以遍历模型，并获取每个 DecisionHead 的 channel_gates
                        # 因为在确定性模式下，它们会根据 channel_gates 决定
                        
                        # 假设我们只取 DecisionHead 的 out_channels 来代表层的宽度
                        # 并且通过计算 gate_params 中非零/有效通道的数量来获取剪枝后的数量
                        
                        # 获取 DecisionHead.channel_gates
                        channel_gates_tensor = module.decision_head.channel_gates # (action_num, out_channels)
                        
                        # 在确定性模式下，决策头会选择一个 action
                        # 找到最常选择的 action
                        # (这里需要知道 DecisionHead 在前向传播时选择了哪个 action)
                        # 最简单的是假设某个 action 是主要的，或者根据 mean_gate 来判断
                        
                        # 为了桑基图，我们直接使用每个 DecisionHead 的输出通道数作为宽度，
                        # 以及剪枝后激活的通道数。
                        
                        # 假设 test() 函数在运行后，default_graph.get_tensor_list('selected_channels')
                        # 已经包含了所有 DecisionHead 的最终输出（按顺序排列）
                        
                        # 实际的 selected_channels 是一个列表，每个元素是一个 batch 的输出
                        # 我们需要将其聚合
                        # 我们可以直接从 test() 运行后 `default_graph` 中提取
                        pass # 实际的提取在 test() 循环后进行
            
            # 基础的测试指标计算
            test_loss.append(F.cross_entropy(output, target).item())
            
            current_batch_selected_channels = default_graph.get_tensor_list('selected_channels')
            if current_batch_selected_channels: # 确保非空
                concat_channels = torch.cat(current_batch_selected_channels, dim=1)
                test_sparsity.append((concat_channels != 0).float().mean().item())
            else:
                test_sparsity.append(0.0) # 如果没有剪枝层，则稀疏度为0

            pred = output.max(1)[1]
            correct += (pred == target).float().sum().item()
            
            # 在这里收集桑基图所需的数据
            # 这是一个关键步骤：我们需要确保在每个batch中，从每一层DecisionHead获取的selected_channels
            # 能够被准确地识别并累积。
            # 为了简化，我们只在测试循环结束后，重新运行一次模型的前向传播
            # 或者从已经保存的 `default_graph.get_tensor_list('selected_channels')` 中获取信息

    acc = correct / len(testloader.dataset)
    print('Test set: Loss: %.4f, '
          'Sparsity: %.4f, Accuracy: %.4f\n' % (
              np.mean(test_loss), np.mean(test_sparsity), acc
          ))
    
    # === 桑基图数据收集 ===
    # 桑基图需要从模型中提取层名、输入通道数、剪枝后的输出通道数。
    # 这里我们只运行一个小的 dummy_input 来触发 forward 并收集 selected_channels
    
    # 确保模型在 eval 模式且确定性
    model.eval()
    apply_func(model, 'DecisionHead', set_deterministic_value, deterministic=True)
    
    dummy_input = torch.randn(1, 3, 32, 32).to(args.device) # 假定 CIFAR 输入是 3x32x32
    default_graph.clear_all_tensors() # 再次清除，确保只收集一个 dummy_input 的信息
    _ = model(dummy_input) # 运行一次前向传播以填充 default_graph
    
    all_selected_channels_tensors = default_graph.get_tensor_list('selected_channels')
    
    # 获取模型中所有 DecisionHead 所在的模块名称，以及其原始输入输出通道
    sankey_nodes_info = []
    sankey_links_data = [] # [[source_node_idx, target_node_idx, value]]
    
    # 映射模型模块名到桑基图节点名和索引
    node_labels = ['Input']
    node_indices = {'Input': 0}
    current_node_idx = 0
    
    current_out_channels = 3 # 初始输入通道数
    
    # 桑基图需要了解每层（或每个 BasicBlock/ConvBNReLU）的输入输出通道数
    # 并计算实际激活的通道数
    
    # 遍历模型并收集信息
    layer_iter = 0
    decision_head_idx = 0 # 用于匹配 all_selected_channels_tensors 的索引
    for name, module in model.named_modules():
        if hasattr(module, 'decision_head') and isinstance(module.decision_head, DecisionHead):
            layer_iter += 1
            node_name = f'Layer{layer_iter}' # 简化的层名
            node_labels.append(node_name)
            current_node_idx += 1
            node_indices[node_name] = current_node_idx
            
            # 获取当前 DecisionHead 的输入通道数（它的前一层输出）
            input_channels_this_layer = module.decision_head.in_channels
            
            # 获取当前 DecisionHead 的输出通道数（剪枝前的总通道数）
            original_output_channels_this_layer = module.decision_head.out_channels
            
            # 从 default_graph 中获取该 DecisionHead 对应的 selected_channels
            # 假设 all_selected_channels_tensors 中的顺序与 module.named_modules() 中的顺序一致
            # 这是关键假设，如果不对，需要更复杂的逻辑来匹配
            if decision_head_idx < len(all_selected_channels_tensors):
                # selected_channels 是 (batch_size, out_channels)
                layer_selected_channels = all_selected_channels_tensors[decision_head_idx]
                
                # 确定剪枝后的实际通道数（例如，非零通道的数量）
                # 这里我们假设 selected_channels 是0或1的近似值，或者直接用其均值
                # 为了绘制桑基图，我们通常需要一个整数通道数
                # 我们可以用一个阈值来确定哪些通道被认为是“激活的”
                threshold = 0.5 # 假设激活通道的权重大于0.5
                pruned_active_channels_count = (layer_selected_channels[0,:] > threshold).float().sum().item() # 取第一个样本的决策
                
                # 桑基图链接：从上一层输出到当前层剪枝后的输入（即当前层有效输出）
                sankey_links_data.append({
                    'source': node_indices[node_labels[current_node_idx-1]], # 上一个节点的索引
                    'target': node_indices[node_name],
                    'value': pruned_active_channels_count, # 剪枝后的通道数
                    'label': f'Active: {int(pruned_active_channels_count)}'
                })
                
                current_out_channels = pruned_active_channels_count # 更新当前输出通道数，作为下一层的输入
            
            decision_head_idx += 1
    
    # 如果要添加 Output 层
    node_labels.append('Output')
    current_node_idx += 1
    node_indices['Output'] = current_node_idx
    
    # 最后一个链接：从最后一层到输出层
    if len(node_labels) > 2: # 确保有中间层
        sankey_links_data.append({
            'source': node_indices[node_labels[-2]],
            'target': node_indices['Output'],
            'value': current_out_channels, # 最后一层剪枝后的输出通道数
            'label': f'Final: {int(current_out_channels)}'
        })

    # 保存桑基图数据
    sankey_output_data = {
        'node_labels': node_labels,
        'links': sankey_links_data
    }
    misc.dump_pickle(sankey_output_data, os.path.join(args.logdir, 'sankey_data.pkl'))
    
    # 打印一些桑基图收集到的信息，用于调试
    print("\n--- Sankey Diagram Data Collected ---")
    print(f"Nodes: {node_labels}")
    for link in sankey_links_data:
        print(f"Link: {node_labels[link['source']]} -> {node_labels[link['target']]}, Value: {link['value']}")
    print("-------------------------------------\n")


    return acc, np.mean(test_sparsity)


def save_checkpoint(state, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth')) # finetune 保存 checkpoint.pth


for epoch in range(args.epochs):
    train(epoch)
    acc, sparsity = test()
    torch.save(model.state_dict(), os.path.join(args.logdir, 'checkpoint.pth'))
    print('Save checkpoint @ Epoch %d, Accuracy = %.4f, Sparsity = %.4f\n' % (
        epoch, acc, sparsity
    ))

print('Training complete.')
