import pickle
import numpy as np
import plotly.graph_objects as go
import os
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

    # 定义 CIFAR-10 类别名称，以便图例更具可读性
    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    # 将决策头的索引映射到更具描述性的名称
    # 假设有5个决策头，如果你的数据中有更多或更少，请调整范围
    action_names = {i: f'Decision Head {i}' for i in range(5)}

    # 初始化用于存储桑基图节点和链接的列表
    sankey_nodes = []
    sankey_links_source = []
    sankey_links_target = []
    sankey_links_value = []
    sankey_links_label = [] # 用于鼠标悬停时的提示文本

    # 用于跟踪唯一节点并为其分配 Plotly 所需的索引
    node_label_to_index = {}
    current_node_index = 0

    def add_node_if_new(label):
        nonlocal current_node_index
        if label not in node_label_to_index:
            node_label_to_index[label] = current_node_index
            sankey_nodes.append(label)
            current_node_index += 1
        return node_label_to_index[label]

    # --- 使用自然排序获取并排序所有层级的名称 ---
    # 这将确保 'Layer_10' 在 'Layer_2' 之后
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]

    layer_names = sorted(data.keys(), key=natural_sort_key)
    if not layer_names:
        print("数据中没有找到任何层级信息，无法生成桑基图。")
        return

    # 处理每一层数据，构建桑基图的链接
    print("正在处理数据以构建桑基图链接...")
    for i, layer_name in enumerate(layer_names):
        current_layer_data = data[layer_name]

        # 遍历当前层中的每一个 CIFAR-10 类别
        for class_id in range(len(cifar10_classes)):
            class_data = current_layer_data.get(class_id)
            if class_data is None:
                continue

            # 创建源节点：表示某个类别在某个层。
            # 基础标签不包含层名，用于在不同层之间表示相同的类别流。
            source_node_label_base = f'{cifar10_classes[class_id]} (Class {class_id})'
            source_node_label_full = f'{source_node_label_base} - {layer_name}'
            source_node_idx = add_node_if_new(source_node_label_full)

            # 遍历所有决策头，为每个决策头创建链接
            for action_index in range(len(class_data['action_frequencies'])):
                action_frequency = class_data['action_frequencies'][action_index]
                mean_action_prob = class_data['mean_action_probs'][action_index]
                action_name = action_names[action_index]

                # 只有当该决策头有非零的流量时才创建链接
                if action_frequency > 0:
                    # 创建中间节点：表示当前层选择的决策头
                    action_node_label = f'{action_name} - {layer_name}'
                    action_node_idx = add_node_if_new(action_node_label)

                    # 添加从“类别节点”到“决策头节点”的链接
                    sankey_links_source.append(source_node_idx)
                    sankey_links_target.append(action_node_idx)
                    sankey_links_value.append(action_frequency) # **使用 action_frequency 作为流量值**
                    sankey_links_label.append(
                        f'{cifar10_classes[class_id]} 在 {layer_name} '
                        f'通过 {action_name} (频率: {action_frequency:.2f}, '
                        f'平均概率: {mean_action_prob:.3f})'
                    )

                    # 如果不是最后一层，则从当前层的决策头链接到下一个层的相同类别
                    if i < len(layer_names) - 1:
                        next_layer_name = layer_names[i+1]
                        # 下一层的类别节点也使用相同的 base label
                        target_node_label_full = f'{source_node_label_base} - {next_layer_name}'
                        target_node_idx = add_node_if_new(target_node_label_full)

                        # 添加从“决策头节点”到“下一层类别节点”的链接
                        sankey_links_source.append(action_node_idx)
                        sankey_links_target.append(target_node_idx)
                        sankey_links_value.append(action_frequency) # **流量值继续使用 action_frequency**
                        sankey_links_label.append(
                            f'{action_name} 引导 {cifar10_classes[class_id]} 进入 {next_layer_name} '
                            f'(流量: {action_frequency:.2f})'
                        )
    print("桑基图链接构建完成。")

    # --- 3. 生成桑基图 ---
    print("正在生成桑基图...")
    fig = go.Figure(data=[go.Sankey(
        node=dict(
          pad=15,
          thickness=20,
          line=dict(color="black", width=0.5),
          label=sankey_nodes,
          # 你可以根据节点类型（类别或决策头）自定义颜色
          # 示例：
          # color=[
          #     "blue" if "Class" in label else "red" for label in sankey_nodes
          # ]
        ),
        link=dict(
          source=sankey_links_source,
          target=sankey_links_target,
          value=sankey_links_value,
          label=sankey_links_label, # 鼠标悬停时显示此文本
          # 你也可以根据源/目标节点或值自定义链接颜色
          # color="rgba(0,0,255,0.2)"
      ))])

    fig.update_layout(
        title_text="<b>CIFAR-10 类别在各层决策头选择的桑基图</b><br>"
                   "（图例：节点名称即是图例，流量宽度表示决策频率）", # 更新图例说明
        font_size=10
    )

    fig.show()
    print("桑基图已生成并显示。")

# --- 使用方法 ---
if __name__ == "__main__":
    pkl_file = '/root/my_pruning/logs/finetune-decision-5/cifar10-resnet56/sparsity-0.40/action_probs_by_class.pkl'
    generate_sankey_from_pkl(pkl_file)
