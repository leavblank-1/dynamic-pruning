import torch.nn.functional as F
import torch.nn as nn
import torch
import types
from torch.distributions import RelaxedOneHotCategorical


class TorchGraph(object):
    def __init__(self):
        self._graph = {}
        self.persistence = {}

    def add_tensor_list(self, name, persist=False):
        self._graph[name] = []
        self.persistence[name] = persist

    def append_tensor(self, name, val):
        self._graph[name].append(val)

    def clear_tensor_list(self, name):
        self._graph[name].clear()

    def get_tensor_list(self, name):
        return self._graph[name]

    def clear_all_tensors(self):
        for k in self._graph.keys():
            if not self.persistence[k]:
                self.clear_tensor_list(k)


default_graph = TorchGraph()
default_graph.add_tensor_list('head_params', True)
default_graph.add_tensor_list('gate_params', True)
default_graph.add_tensor_list('sampled_actions')
default_graph.add_tensor_list('selected_channels')
# 移除了 default_graph.add_tensor_list('temperature', True)


class DecisionHead(nn.Module):
    # 增加 temperature 参数
    def __init__(self, in_channels, out_channels, action_num, deterministic=False, temperature=1.0):
        super(DecisionHead, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.action_num = action_num
        self.deterministic = deterministic
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, action_num, bias=False)
        self.relu = nn.ReLU()
        self.channel_gates = nn.Parameter(torch.ones(action_num, out_channels))
        self.temperature = temperature # 将 temperature 存储为实例属性

    def head_params(self):
        return [self.fc1.weight]

    def gate_params(self):
        return [self.channel_gates]

    def normalize_weights(self):
        self.fc1.weight.data = F.normalize(self.fc1.weight.data, dim=1)

    def forward(self, x):
        out = self.avgpool(self.relu(x))
        out = out.view(x.shape[0], x.shape[1])
        out = self.fc1(out)
        action_probs = F.softmax(out, dim=1) # 得到概率分布

        # === 核心修改：在任何模式下都记录 action_probs ===
        # 使用模块实例的ID作为键，确保唯一性，方便分析时按层级提取
        default_graph.append_tensor(f'action_probs_{id(self)}', action_probs)

        if self.deterministic or not self.training:
            sampled_actions = action_probs.max(1)[1] # 确定性选择

            # === 核心修改：在确定性模式下也记录 sampled_actions ===
            default_graph.append_tensor(f'sampled_actions_{id(self)}', sampled_actions)

            selected_channels = self.channel_gates[sampled_actions]
        else:
            # 直接使用 self.temperature
            m = RelaxedOneHotCategorical(self.temperature, action_probs)
            actions = m.rsample()
            onehot_actions = torch.zeros(actions.size()).to(x.device)
            
            sampled_actions = actions.max(1)[1] # Gumbel-softmax 采样后的离散动作

            # === 核心修改：在Gumbel-Softmax模式下也记录 sampled_actions ===
            default_graph.append_tensor(f'sampled_actions_{id(self)}', sampled_actions)

            onehot_actions.scatter_(1, sampled_actions.unsqueeze(1), 1)
            substitute_actions = (onehot_actions - actions).detach() + actions
            selected_channels = torch.mm(substitute_actions, self.channel_gates)

        # === 核心修改：确保 selected_channels 总是被记录（这是桑基图和稀疏度分析的基础）===
        # 注意：这里是将所有DecisionHead的selected_channels放到一个公共的列表中
        default_graph.append_tensor('selected_channels', selected_channels) 

        return sampled_actions, selected_channels




def apply_func(model, module_type, func, **kwargs):
    for m in model.modules():
        if m.__class__.__name__ == module_type:
            func(m, **kwargs)


def replace_func(model, module_type, func):
    for m in model.modules():
        if m.__class__.__name__ == module_type:
            m.forward = types.MethodType(func, m)


def collect_params(m):
    for p in m.head_params():
        default_graph.append_tensor('head_params', p)

    for p in m.gate_params():
        default_graph.append_tensor('gate_params', p)


def set_deterministic_value(m, deterministic):
    m.deterministic = deterministic

# 新增：设置 DecisionHead 温度的函数
def set_temperature_value(m, temperature):
    if isinstance(m, DecisionHead): # 确保是 DecisionHead 实例
        m.temperature = temperature
    # else: 可以在这里添加警告或错误处理，如果尝试设置非 DecisionHead 模块的温度

def normalize_head_weights(m):
    m.normalize_weights()


# 增加 temperature 参数，并传递给 DecisionHead
def init_decision_convbn(m, action_num, temperature=1.0):
    m.decision_head = DecisionHead(
        m.conv.in_channels, m.conv.out_channels, action_num, temperature=temperature
    )


def decision_convbn_forward(self, x):
    out = self.conv(x)
    out = self.bn(out)
    if self.conv.in_channels > 3: # 假设第一层不剪枝
        sampled_actions, selected_channels = self.decision_head(x)

        default_graph.append_tensor('sampled_actions', sampled_actions)
        default_graph.append_tensor('selected_channels', selected_channels)

        out = selected_channels.unsqueeze(2).unsqueeze(3) * out
    out = self.relu(out)
    return out


# 增加 temperature 参数，并传递给 DecisionHead
def init_decision_basicblock(m, action_num, temperature=1.0):
    m.decision_head = DecisionHead(
        m.conv1.in_channels, m.conv1.out_channels, action_num, temperature=temperature
    )


def decision_basicblock_forward(self, x):
    sampled_actions, selected_channels = self.decision_head(x)

    default_graph.append_tensor('sampled_actions', sampled_actions)
    default_graph.append_tensor('selected_channels', selected_channels)

    out = self.conv1(x)
    out = self.bn1(out)
    out = selected_channels.unsqueeze(2).unsqueeze(3) * out # 剪枝应用在conv1输出之后
    out = F.relu(out)

    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


# 增加 temperature 参数，并传递给 DecisionHead
def init_decision_bottleneck(m, action_num, temperature=1.0):
    m.decision_head = DecisionHead(
        m.conv1.in_channels, m.conv1.out_channels, action_num, temperature=temperature
    )


def decision_bottleneck_forward(self, x):
    residual = x

    out = self.bn1(x)
    out = self.relu(out)

    sampled_actions, selected_channels = self.decision_head(out)

    default_graph.append_tensor('sampled_actions', sampled_actions)
    default_graph.append_tensor('selected_channels', selected_channels)

    out = self.conv1(out)

    out = self.bn2(out)
    out = selected_channels.unsqueeze(2).unsqueeze(3) * out # 剪枝应用在这里
    out = self.relu(out)
    out = self.conv2(out)

    out = self.bn3(out)
    out = self.relu(out)
    out = self.conv3(out)

    if self.downsample is not None:
        residual = self.downsample(x)

    out += residual

    return out
