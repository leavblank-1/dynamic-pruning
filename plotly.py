import plotly.graph_objects as go
import pickle as pkl
import os
import numpy as np

# --- Configuration ---
# 替换为你的实际日志目录，用于加载分析结果
analysis_logdir = 'finetune-decision-40/cifar10-resnet56/sparsity-0.7' 
num_classes = 10 # 你的数据集类别数
input_flow_value = 1000 # 输入层的总流量，可以是你希望的任何基础值，例如总样本数或1000

# --- Load Data ---
# 加载你分析阶段保存的每个层、每个类别对应的动作概率数据
action_probs_by_layer_and_class = pkl.load(open(os.path.join(analysis_logdir, 'action_probs_by_layer_and_class.pkl'), 'rb'))

# --- Prepare Sankey Data Structures ---
node_labels = [] # 存储桑基图所有节点的标签
node_indices = {} # 存储节点标签到其索引的映射
links = [] # 存储桑基图所有链接的信息

current_node_idx = 0

# --- Add Input Node ---
# 添加图的起始节点
node_labels.append('Input')
node_indices['Input'] = current_node_idx
current_node_idx += 1

# --- Process Layers and Actions ---
# 对层名进行排序，确保桑基图的层级顺序正确
sorted_layer_names = sorted(action_probs_by_layer_and_class.keys(), key=lambda x: int(x.split('_')[1]))

# 这个字典用于存储层名和动作索引到其桑基图节点索引的映射
layer_action_node_map = {} 

# 遍历每个决策头层
for layer_name in sorted_layer_names:
    layer_data = action_probs_by_layer_and_class[layer_name]

    # 计算该层所有类别样本的平均动作频率
    all_class_frequencies = []
    for class_id, class_data in layer_data.items():
        all_class_frequencies.append(class_data['action_frequencies'])
    
    if not all_class_frequencies:
        continue # 如果该层没有数据，则跳过

    avg_action_frequencies = np.mean(all_class_frequencies, axis=0) # 对所有类别的频率取平均

    # 为当前层的每个动作添加节点
    for action_idx in range(len(avg_action_frequencies)):
        action_node_label = f'{layer_name}_Action{action_idx}' # 节点标签示例：Layer_1_Action0
        node_labels.append(action_node_label)
        node_indices[action_node_label] = current_node_idx
        layer_action_node_map[(layer_name, action_idx)] = current_node_idx # 存储映射
        current_node_idx += 1

    # 创建从上一层或“Input”节点到当前层动作节点的链接
    if layer_name == sorted_layer_names[0]: # 如果是第一个决策头层
        # 从“Input”节点连接到当前层的每个动作节点
        for action_idx, freq in enumerate(avg_action_frequencies):
            links.append(dict(
                source=node_indices['Input'], # 源节点是“Input”
                target=layer_action_node_map[(layer_name, action_idx)], # 目标节点是当前层的某个动作节点
                value=input_flow_value * freq, # 流量值 = 总输入流量 * 该动作的频率
                label=f'Action {action_idx}: {freq:.2f}' # 链接标签显示动作和频率
            ))
    else: # 如果是后续的决策头层
        prev_layer_name = sorted_layer_names[sorted_layer_names.index(layer_name) - 1]
        
        # 复杂性说明：
        # 这里为了简化，我们没有精确地连接上一层的每个动作到当前层的每个动作（这样会非常复杂）。
        # 而是假定从上一层所有动作的总流量中，按当前层动作的频率进行分配。
        # 最简单的方式是创建一个代表上一层“整体输出”的节点，或者直接从上一层的各个动作节点拉出总流量。
        
        # 为了可视化决策流，我们创建了中间的“输出”节点来汇聚流量
        # 获取上一层总体的平均动作频率
        prev_layer_avg_freq = np.mean([d['action_frequencies'] for d in action_probs_by_layer_and_class[prev_layer_name].values()], axis=0)
        
        # 计算进入当前层动作的总流量 (来自上一层所有动作的总和)
        # 简单地假设前一层的总流量等于其动作频率之和（理论上是1）乘以初始流量
        # 或者，如果之前已经汇聚到 'prev_layer_name_Output' 节点，则从那里获取
        total_flow_into_current_layer_actions = input_flow_value * np.sum(prev_layer_avg_freq) # 假设总和接近1

        for action_idx, freq in enumerate(avg_action_frequencies):
            # 将前一层的总流量按当前层动作的频率分配
            links.append(dict(
                source=node_indices[f'{prev_layer_name}_Output'], # 源节点是上一层的聚合输出节点
                target=layer_action_node_map[(layer_name, action_idx)],
                value=total_flow_into_current_layer_actions * freq, # 流量值
                label=f'Action {action_idx}: {freq:.2f}'
            ))
    
    # 为当前层添加一个“输出”节点，用于汇聚从该层所有动作流出的流量
    # 这样可以清晰地显示从一层流向下一层的总流量
    if layer_name != sorted_layer_names[-1]: # 最后一层之后没有再连接到其他决策层
        layer_output_node_label = f'{layer_name}_Output'
        node_labels.append(layer_output_node_label)
        node_indices[layer_output_node_label] = current_node_idx
        current_node_idx += 1
        
        # 从当前层的每个动作节点连接到当前层的聚合输出节点
        for action_idx, freq in enumerate(avg_action_frequencies):
            links.append(dict(
                source=layer_action_node_map[(layer_name, action_idx)], # 源节点是当前层的某个动作节点
                target=node_indices[layer_output_node_label], # 目标节点是当前层的聚合输出节点
                value=total_flow_into_current_layer_actions * freq, # 流量值
                label=f'To next: {freq:.2f}'
            ))

# --- Add Final Output Node ---
# 添加图的最终输出节点
node_labels.append('Final Output')
node_indices['Final Output'] = current_node_idx

# 获取最后一层动作的平均频率
last_layer_name = sorted_layer_names[-1]
last_layer_avg_freq = np.mean([d['action_frequencies'] for d in action_probs_by_layer_and_class[last_layer_name].values()], axis=0)

# 计算进入最后一层动作的总流量（来自前一层的总流量）
total_flow_into_last_layer_actions = input_flow_value * np.prod([np.sum(np.mean([d['action_frequencies'] for d in action_probs_by_layer_and_class[ln].values()], axis=0)) for ln in sorted_layer_names[:-1]]) if len(sorted_layer_names) > 1 else input_flow_value

# 将最后一层所有动作的流量连接到“Final Output”节点
for action_idx, freq in enumerate(last_layer_avg_freq):
    links.append(dict(
        source=layer_action_node_map[(last_layer_name, action_idx)], # 源节点是最后一层的某个动作节点
        target=node_indices['Final Output'], # 目标节点是“Final Output”
        value=total_flow_into_last_layer_actions * freq,
        label=f'To Final: {freq:.2f}'
    ))

# --- Create and Display Sankey Diagram ---
# 使用 Plotly 创建桑基图
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15, # 节点之间的间距
        thickness=20, # 节点的厚度
        line=dict(color="black", width=0.5), # 节点边框
        label=node_labels, # 节点标签
    ),
    link=dict(
        source=[link['source'] for link in links], # 链接的源节点索引
        target=[link['target'] for link in links], # 链接的目标节点索引
        value=[link['value'] for link in links], # 链接的流量值
        label=[link['label'] for link in links], # 链接的标签
    ))])

fig.update_layout(title_text="Decision Flow through Dynamic Pruning Network", font_size=10) # 设置图表标题和字体
fig.show() # 显示图表

# fig.write_html(os.path.join(analysis_logdir, 'decision_sankey_diagram.html')) # 可以保存为HTML文件
