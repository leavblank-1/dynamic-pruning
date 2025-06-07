import plotly.graph_objects as go
import pickle as pkl
import os
import numpy as np
import re # 导入正则表达式模块

def generate_sankey_from_pkl(pkl_file_path):
    """
    从 .pkl 文件加载数据，并生成 CIFAR-10 类别在各层决策头选择的桑基图。

    Args:
        pkl_file_path (str): .pkl 文件的路径。
    """

    # --- 1. 读取 PKL 文件 ---
    print(f"正在尝试从 '{pkl_file_path}' 读取数据...")
    if not os.path.exists(pkl_file_path):
        print(f"错误：文件 '{pkl_file_path}' 不存在。请检查文件路径或确保文件已放置在正确位置。")
        return

    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        print("数据读取成功。")
    except Exception as e:
        print(f"读取 .pkl 文件时发生错误：{e}")
        print("请确保 .pkl 文件是由 Python 的 pickle 模块正确序列化的，且与当前 Python 版本兼容。")
        return

    # --- 2. 准备桑基图所需的数据 ---

    # 定义 CIFAR-10 类别名称
    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    num_classes = len(cifar10_classes)

    # 定义颜色，每个类别使用一种颜色
    class_colors = [
        'rgba(31, 119, 180, 0.8)',  # blue
        'rgba(255, 127, 14, 0.8)',  # orange
        'rgba(44, 160, 44, 0.8)',   # green
        'rgba(214, 39, 40, 0.8)',   # red
        'rgba(148, 103, 189, 0.8)', # purple
        'rgba(140, 86, 75, 0.8)',   # brown
        'rgba(227, 119, 194, 0.8)', # pink
        'rgba(127, 127, 127, 0.8)', # gray
        'rgba(188, 189, 34, 0.8)',  # olive
        'rgba(23, 190, 207, 0.8)'   # cyan
    ]

    # 初始化用于存储桑基图节点和链接的列表
    sankey_nodes_labels = [] 
    sankey_links_source = [] 
    sankey_links_target = [] 
    sankey_links_value = []  
    sankey_links_label = []  
    sankey_links_color = []  # 存储链接的颜色

    # 用于跟踪唯一节点并为其分配 Plotly 所需的索引
    node_label_to_index = {}
    current_node_index = 0

    def add_node_if_new(label):
        nonlocal current_node_index
        if label not in node_label_to_index:
            node_label_to_index[label] = current_node_index
            sankey_nodes_labels.append(label)
            current_node_index += 1
        return node_label_to_index[label]

    # --- 使用自然排序获取并排序所有层级的名称 ---
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]

    layer_names_raw = sorted(data.keys(), key=natural_sort_key)
    if not layer_names_raw:
        print("数据中没有找到任何层级信息，无法生成桑基图。")
        return
    
    # 简化层名映射：Layer_X_original_path -> Layer X
    layer_name_map = {}
    for raw_name in layer_names_raw:
        match = re.search(r'Layer_(\d+)', raw_name)
        if match:
            simplified_name = f"Layer {match.group(1)}"
        else:
            simplified_name = raw_name 
        layer_name_map[raw_name] = simplified_name


    # Store the total flow for each class as it passes through layers
    current_class_flow = {} 

    # --- 添加初始类别节点 (表示输入) ---
    input_flow_per_class = 1000 / num_classes # 每个类别的初始流量
    
    for class_id in range(num_classes):
        # *** 修改点 1: 初始节点标签直接使用类别名称 ***
        node_label = f'{cifar10_classes[class_id]}' # 示例：'airplane'
        add_node_if_new(node_label)
        current_class_flow[('Input', class_id)] = input_flow_per_class


    # --- 处理每一层数据，构建桑基图的链接 ---
    print("正在处理数据以构建桑基图链接...")
    for i, raw_layer_name in enumerate(layer_names_raw): # 遍历原始的层名
        layer_name = layer_name_map[raw_layer_name] # 使用简化的层名作为节点标签
        current_layer_data = data[raw_layer_name] # 但数据仍然从原始层名中获取

        # --- 为当前层添加动作节点 (使用简化标签) ---
        first_class_id_with_data = next(iter(current_layer_data.keys()), None)
        if first_class_id_with_data is None:
            continue
        num_actions_this_layer = len(current_layer_data[first_class_id_with_data]['mean_action_probs']) 

        for action_idx in range(num_actions_this_layer):
            # 节点标签只显示 Layer X_Action Y (简化后的层名)
            action_node_label = f'{layer_name}_Action{action_idx}' # 示例：Layer 1_Action0
            add_node_if_new(action_node_label)
        
        # --- 为当前层添加聚合输出节点 (使用简化标签) ---
        layer_output_node_label = f'{layer_name}_Output' # 示例：Layer 1_Output
        add_node_if_new(layer_output_node_label)

        # --- 遍历当前层中的每一个 CIFAR-10 类别，创建链接 ---
        for class_id in range(num_classes):
            class_data = current_layer_data.get(class_id)
            if class_data is None or len(class_data['mean_action_probs']) == 0:
                continue

            class_mean_action_probs = class_data['mean_action_probs'] # <-- 使用 mean_action_probs
            
            # --- 确定源节点和当前类别的流入流量 ---
            source_node_label_for_class = ''
            flow_into_class_this_layer = 0.0

            if i == 0: # 如果是第一层 DecisionHead
                source_node_label_for_class = f'{cifar10_classes[class_id]}' # 源节点是类别名称 (与输入节点标签一致)
                flow_into_class_this_layer = current_class_flow[('Input', class_id)]
            else: # 如果是后续层
                prev_raw_layer_name = layer_names_raw[i - 1] # 获取前一层的原始名称
                # 源节点是上一层的聚合输出节点 (使用简化后的前一层输出节点名)
                source_node_label_for_class = f'{layer_name_map[prev_raw_layer_name]}_Output' 
                flow_into_class_this_layer = current_class_flow[(prev_raw_layer_name, class_id)] # 获取该类别从上一层流出的总流量

            # --- 从源节点链接到当前层的各个动作节点 ---
            # 使用 mean_action_probs 来分配流量
            sum_probs = np.sum(class_mean_action_probs)
            normalized_mean_probs = class_mean_action_probs / sum_probs if sum_probs > 1e-9 else np.zeros_like(class_mean_action_probs)
            
            for action_idx, prob_val in enumerate(normalized_mean_probs): # <-- 遍历 normalized_mean_probs
                if prob_val * flow_into_class_this_layer > 1e-6: 
                    sankey_links_source.append(node_label_to_index[source_node_label_for_class])
                    sankey_links_target.append(node_label_to_index[f'{layer_name}_Action{action_idx}'])
                    sankey_links_value.append(flow_into_class_this_layer * prob_val) 
                    sankey_links_label.append(
                        f'{cifar10_classes[class_id]} 经 {layer_name}_Action{action_idx} (概率: {prob_val:.4f})' # 标签也使用简化层名
                    )
                    sankey_links_color.append(class_colors[class_id % len(class_colors)])

            # --- 更新当前类别在这一层的流出总量 ---
            total_flow_out_of_this_layer_for_this_class = flow_into_class_this_layer * np.sum(class_mean_action_probs) 
            current_class_flow[(raw_layer_name, class_id)] = total_flow_out_of_this_layer_for_this_class 


            # --- 从当前层动作节点链接到当前层的聚合输出节点 ---
            for action_idx, prob_val in enumerate(normalized_mean_probs):
                 if prob_val * total_flow_out_of_this_layer_for_this_class > 1e-6:
                    sankey_links_source.append(node_label_to_index[f'{layer_name}_Action{action_idx}']) 
                    sankey_links_target.append(node_label_to_index[layer_output_node_label]) 
                    sankey_links_value.append(total_flow_out_of_this_layer_for_this_class * prob_val) 
                    sankey_links_label.append(
                        f'{cifar10_classes[class_id]} 流向 {layer_name}_Output (概率: {prob_val:.4f})' # 标签也使用简化层名
                    )
                    sankey_links_color.append(class_colors[class_id % len(class_colors)])

    print("桑基图链接构建完成。")

    # --- 3. 添加最终输出节点 ---
    final_output_node_label = 'Final Output'
    add_node_if_new(final_output_node_label)

    # --- 从最后一层聚合输出节点链接到最终输出节点 ---
    last_raw_layer_name = layer_names_raw[-1]
    
    for class_id in range(num_classes):
        flow_from_last_layer_for_class = current_class_flow.get((last_raw_layer_name, class_id), 0.0)
        if flow_from_last_layer_for_class > 1e-6:
            # 源节点是最后一层的聚合输出节点 (使用简化后的最后一层输出节点名)
            sankey_links_source.append(node_label_to_index[f'{layer_name_map[last_raw_layer_name]}_Output']) 
            sankey_links_target.append(node_label_to_index[final_output_node_label])
            sankey_links_value.append(flow_from_last_layer_for_class)
            sankey_links_label.append(f'{cifar10_classes[class_id]} 最终流量: {flow_from_last_layer_for_class:.2f}')
            sankey_links_color.append(class_colors[class_id % len(class_colors)])


    # --- 4. 生成桑基图 ---
    print("正在生成桑基图...")
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_nodes_labels,
            # 你可以根据节点类型（类别或决策头）自定义颜色
            # 示例：
            # color=[node_colors[label.split(' ')[0]] if 'Action' in label else (class_colors[cifar10_classes.index(label.split(' ')[0])] if 'Input' not in label else 'gray') for label in sankey_nodes_labels]
        ),
        link=dict(
            source=sankey_links_source,
            target=sankey_links_target,
            value=sankey_links_value,
            label=sankey_links_label, 
            color=sankey_links_color 
        ))])

    fig.update_layout(
        title_text="<b>CIFAR-10 类别特定决策流 (软概率) 桑基图</b>",
        font_size=10
    )

    fig.show()
    print("桑基图已生成并显示。")

    # --- 5. 添加保存选项 ---
    save_path_dir = os.path.dirname(pkl_file_path)
    save_file_name = "class_soft_decision_sankey_diagram.html"
    save_full_path = os.path.join(save_path_dir, save_file_name)

    print(f"正在保存桑基图到: {save_full_path}")
    try:
        fig.write_html(save_full_path)
        print("桑基图保存成功。")
    except Exception as e:
        print(f"保存桑基图时发生错误: {e}")


# --- 使用方法 ---
if __name__ == "__main__":
    pkl_file = '/root/my_pruning/logs/finetune-decision-5/cifar10-resnet56/sparsity-0.40/action_probs_by_class.pkl'
    generate_sankey_from_pkl(pkl_file)
