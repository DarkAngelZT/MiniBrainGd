# MiniBrainGd

一个为 Godot 引擎设计的 C++ 神经网络扩展库，提供强化学习 (PPO) 代理功能。

## 项目简介

MiniBrainGd 是基于 MiniBrain 神经网络库的 Godot GDExtension，为游戏开发者提供集成的 AI 代理能力。支持**推理模式**（实时决策）和**训练模式**（策略学习），使用 PPO (Proximal Policy Optimization) 算法进行训练。

## 功能特性

- 🎮 **Godot 集成**：作为原生 GDExtension 无缝集成到 Godot 项目
- 🤖 **神经网络架构**：支持嵌入层、注意力机制、GRU、全连接层等
- 🎯 **双模式支持**：推理模式快速决策 + 训练模式在线学习
- 📊 **PPO 算法**：优化的策略梯度算法，适合游戏 AI 训练
- ⚡ **自动微分**：使用 autodiff 库进行高效的反向传播

## 编译与安装

### 前置要求

- Godot 4.5+
- Scons
- C++17 编译器(推荐 clang++)

### 编译

```bash
# 构建 godot-cpp 绑定
./build_godot_cpp.bat [编译器路径] [target]

# 完整构建（包含所有目标）
./build_all.bat [编译器路径] [target]
```

编译产物将输出到 `bin/` 目录。

## 使用指南

### 快速开始

```gdscript
# 创建 AI 代理（推理模式）
var agent = AIAgent.new()
agent.set_mode(AIAgent.INFERENCE)

# 初始化网络
agent.Init(
    input_dim=64,           # 输入维度
    move_dim=6,             # 移动动作维度
    shoot_dim=3,            # 射击动作维度
    entity_feature_dim=4    # 实体特征维度
)

# 推理单条输入
var actions = agent.ProcessSensorData(input_vector)
# 返回值：[horizon, vertical, shoot_angle, shoot_action]
```

### 推理模式 (INFERENCE)

用于实时决策，每次输入一条状态，输出对应的动作。

#### 主要函数

**`ProcessSensorData(data: PackedFloat32Array, isGameEnd: bool) -> PackedFloat32Array`**

对单条输入进行推理。

- **参数**：
  - `data`: 长度为 `input_dim` 的浮点数组，包含场景状态（如多个实体的位置、速度等特征）
  - `isGameEnd`: 标志位，标识当前游戏是否结束，结束会自动清空GRU记忆，防止后续数据产生记忆跳变

- **返回**：
  - 长度为 4 的动作数组：
    - `[0]`: 水平移动方向 (0/1/2)
    - `[1]`: 竖直移动方向 (0/1/2)
    - `[2]`: 射击角度 (-180 ~ 180)
    - `[3]`: 射击动作 (0.0/1.0)

#### 使用示例

```gdscript
# 创建推理代理
var agent = AIAgent.new()
agent.set_mode(AIAgent.INFERENCE)
agent.Init(input_dim=64, move_dim=6, shoot_dim=3)

# 每帧推理
func _process(delta):
    var sensor_data = collect_sensor_data()  # 收集环境状态
    var actions = agent.ProcessSensorData(sensor_data)
    
    apply_actions(actions[0], actions[1], actions[2], actions[3])
```

### 训练模式 (TRAINING)

支持多 Agent 并行采样和批量训练。

#### 工作流程

```
BatchProcessSensorData → [执行一帧环境交互] → PushTrainingData → [重复] → Train
```

#### 主要函数

**`SetBatchInfo(batch_size: int, action_dim: int, num_frames: int = 1)`**

初始化训练数据缓存。必须在训练前调用。

- **参数**：
  - `batch_size`: 单批次 Agent 数量
  - `action_dim`: 动作维度（通常为移动维度 + 射击维度）
  - `num_frames`: 每个 Agent 的历史帧数

**`BatchProcessSensorData(batch_data: Array, agent_ids: PackedInt32Array) -> Array`**

批量推理多个 Agent 的状态，并执行探索性采样。

- **参数**：
  - `batch_data`: Agent 数组，每个元素为 `PackedFloat32Array` 状态向量
  - `agent_ids`: 对应的 Agent ID 数组

- **返回**：
  - 动作数组，与 `batch_data` 同长度

- **副作用**：
  - 内部计算并存储：log_probs、critic values、动作
  - 准备数据用于后续的 `PushTrainingData`

**`PushTrainingData(batch_rewards: PackedFloat32Array, agent_ids: PackedInt32Array, batch_dones: PackedFloat32Array)`**

将环境反馈（奖励、终止信号）与采样数据关联。

- **参数**：
  - `batch_rewards`: 奖励值，长度为 `batch_size`
  - `agent_ids`: Agent ID，长度为 `batch_size`
  - `batch_dones`: 当前游戏是否结束标记，长度为 `batch_size` (0.0/1.0)

- **说明**：
  - 每帧调用一次，自动更新上一帧的 next state Q-value
  - 采样数据会按照 Agent ID 存储在缓冲区中

**`Train(step: int)`**

执行 PPO 训练算法。

- **参数**：
  - `step`: 训练轮数，建议10左右

- **说明**：
  - 计算优势函数和 TD 目标
  - 使用 PPO 损失函数进行策略和价值网络更新

#### 使用示例

```gdscript
# 创建训练代理
var agent = AIAgent.new()
agent.set_mode(AIAgent.TRAINING)
agent.Init(input_dim=6, move_dim=6, shoot_dim=3, entity_feature_dim=4, 
           embedding_dim=32, attention_key_dim=16, gru_hidden_dim=64, out_hidden_dim=128)

# 设置训练参数
agent.SetBatchInfo(batch_size=8, action_dim=9, num_frames=32)
agent.SetLearningParameters(gamma=0.93, lambda=0.9, clip_epsilon=0.2)

# 训练循环
var frame_count = 0
var total_frames = 32 * 8  # num_frames * batch_size

func _process(delta):
    # 1. 采集当前状态
    var batch_states = []
    var agent_ids = []
    for i in range(8):
        batch_states.append(agents[i].get_observation())
        agent_ids.append(i)
    
    # 2. 模型输出动作
    var actions = agent.BatchProcessSensorData(batch_states, PackedInt32Array(agent_ids))
    
    # 3. 执行动作，收集反馈
    var rewards = []
    var dones = []
    for i in range(8):
        var r = execute_action(i, actions[i])
        var d = check_done(i)
        rewards.append(r)
        dones.append(1.0 if d else 0.0)
    
    # 4. 推送训练数据
    agent.PushTrainingData(
        PackedFloat32Array(rewards),
        PackedInt32Array(agent_ids),
        PackedFloat32Array(dones)
    )
    
    frame_count += 8
    
    # 5. 收集够足够数据后训练
    if frame_count >= total_frames:
        agent.Train(step=4)  # 每批4轮更新
        frame_count = 0
```

### 保存与加载

`AIAgent` 支持将网络参数保存到磁盘，也可以从磁盘重新加载。保存与加载会根据当前 `mode` 使用不同的网络对象：

- 推理模式 `INFERENCE`：保存/加载 `m_preprocessNet`, `m_moveNet`, `m_shootNet`
- 训练模式 `TRAINING`：保存/加载 `m_actor_preprocessNet`, `m_actor_moveNet`, `m_actor_shootNet`

默认参数：
- `parent_folder = "ai"`
- `file_name = "checkpoint"`

保存与加载时会自动补全 `.param` 后缀，例如：
- `ai/checkpoint.param`

#### 代码示例

```gdscript
# 使用默认文件名保存网络参数
agent.Save()

# 指定保存路径与基础文件名
agent.Save("C:/path/to/your/project/ai", "checkpoint")

# 从指定路径加载网络参数
agent.Load("C:/path/to/your/project/ai", "checkpoint")
```

### 网络初始化参数

**`Init(input_dim, move_dim, shoot_dim, entity_feature_dim=16, embedding_dim=16, attention_key_dim=128, gru_hidden_dim=128, out_hidden_dim=128)`**

初始化神经网络架构。

- **参数**：
  - `input_dim`: 输入维度（所有实体特征的总维度）
  - `move_dim`: 移动动作维度（输出）
  - `shoot_dim`: 射击动作维度（输出）
  - `entity_feature_dim`: 单个实体的特征维度
  - `embedding_dim`: 单个实体特征升维后的输出维度
  - `attention_key_dim`: 注意力机制的键维度
  - `gru_hidden_dim`: GRU 隐层维度
  - `out_hidden_dim`: 输出网络隐层维度

### 训练参数

**`SetLearningParameters(gamma=0.93, lambda=0.9, clip_epsilon=0.2, continuous_gamma=0.9)`**

设置 PPO 训练超参数。

- **参数**：
  - `gamma`: 奖励衰减因子 (0 ~ 1)
  - `lambda`: GAE λ 参数，用于优势函数计算
  - `clip_epsilon`: PPO 裁剪参数
  - `continuous_gamma`: 连续动作衰减因子

## 网络架构

### 推理/Actor 网络

```
Input → Embedding → Attention → StatePooling → GRU → [Actor Head]
                                                      ├─ Move Net → Softmax
                                                      └─ Shoot Net → Gaussian
```

### Critic 网络

```
GRU Output → Dense(3×embedding_dim) → ReLU → Dense(1) → State Value
```

## 数据结构

### TrainingData

内部训练数据缓冲结构：

```
state           [state_dim × (batch_size × num_frames)]  # 存储的状态
actions         [action_dim × (batch_size × num_frames)] # 存储的动作
rewards         [1 × (batch_size × num_frames)]          # 奖励
done            [1 × (batch_size × num_frames)]          # 完成标记
old_log_probs   [1 × (batch_size × num_frames)]          # 采样动作的对数概率
old_critic_values [1 × (batch_size × num_frames)]        # 状态值估计
old_q_values    [1 × (batch_size × num_frames)]          # 当前状态Q值
```

## 高级用法

### 模式切换

```gdscript
# 获取当前模式
var current_mode = agent.get_mode()

# 切换模式（通常在初始化时指定，不建议运行时切换）
agent.set_mode(AIAgent.TRAINING)
```

### 多 Agent 训练

支持在同一代理对象上并行训练多个独立的 Agent：

```gdscript
var agents_data = []
for agent_id in range(num_agents):
    agents_data.append(get_agent_observation(agent_id))

var actions = agent.BatchProcessSensorData(agents_data, PackedInt32Array(range(num_agents)))
```

Agent 通过 `agent_id` 自动维护各自的 GRU 状态和轨迹缓冲。

## 常见问题

### Q: 推理和训练模式有什么区别？

**A**: 
- **推理模式**：使用 `float32` 计算，快速、轻量级，不支持训练，用于游戏运行时决策
- **训练模式**：使用自动微分变量 `AutoDiffVar`，支持反向传播，用于离线或在线学习

### Q: 如何处理变长序列？

**A**: 所有输入必须固定维度。处理变长场景建议：
1. 将所有实体填充到固定数量
2. 使用掩码机制（在 Attention 层中实现）
3. 使用池化操作处理变长特征

### Q: 如何保存和加载训练好的模型？

**A**: `AIAgent` 已支持内置保存与加载接口：
- 使用 `agent.Save()` 将当前网络参数写入磁盘；
- 使用 `agent.Load()` 从磁盘加载保存的参数。

默认文件路径与名称为：
- `parent_folder = "ai"`
- `file_name = "checkpoint"`

保存/加载时会自动补齐 `.param` 后缀，例如 `ai/checkpoint.param`。

注意：
- `INFERENCE` 模式保存/加载 `m_preprocessNet`, `m_moveNet`, `m_shootNet`
- `TRAINING` 模式保存/加载 `m_actor_preprocessNet`, `m_actor_moveNet`, `m_actor_shootNet`

示例：
```gdscript
agent.Save()
agent.Load()
agent.Save("C:/path/to/your/project/ai", "checkpoint")
agent.Load("C:/path/to/your/project/ai", "checkpoint")
```

## 项目结构

```
MiniBrainGd/
├── src/
│   ├── AIAgent.h/cpp         # AI 代理主类
│   ├── EmbeddingLayer.h/cpp  # 嵌入层实现
│   ├── StatePooling.h/cpp    # 状态池化层
│   └── register_types.h/cpp  # Godot 注册
├── MiniBrain/                # 神经网络核心库
│   ├── Source/
│   │   ├── Activations/      # 激活函数
│   │   ├── Layers/           # 网络层
│   │   ├── LossFunc/         # 损失函数
│   │   └── ...
│   └── Eigen/                # 第三方库
├── godot-cpp/                # Godot C++ 绑定
├── MiniBrainGd.gdextension  # GDExtension 清单
└── SConstruct                # 构建脚本
```

## 技术栈

- **Godot**: 4.5 GDExtension API
- **C++**: 17 标准
- **线性代数**: Eigen 3.4.1
- **自动微分**: autodiff 库
- **算法**: PPO, Adam 优化器

## 许可证

详见 [LICENSE](LICENSE) 文件。

## 贡献指南

欢迎提交 Issue 和 Pull Request！

---

**最后更新**: 2026
