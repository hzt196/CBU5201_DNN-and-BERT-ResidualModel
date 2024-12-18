# CBU5201_miniproject_hzt

DNN Method:

Architecture:
The DNN consists of a multi-layer structure made up of multiple fully connected neural units. The architecture includes hidden layers using activation functions such as ReLU (Rectified Linear Unit) to introduce non-linearity, enabling the network to learn complex patterns from input data.
Loss Function: For classification tasks, a cross-entropy loss function is used to ensure the model can effectively distinguish between different categories.
Training Process:
The DNN is trained using backpropagation and mini-batch stochastic gradient descent. The optimizer used is Adam, which is suitable for deep network training as it can dynamically adjust the learning rate.

BERT-ResidualModel Method:

BERT Integration:

BERT-ResidualModel combines the contextual embeddings of BERT with the benefits of a residual learning framework. In this experiment, speech is converted into text for subsequent text analysis.
BERT: BERT is pre-trained on large corpora to understand the context of words, capturing complex semantic relationships. This pre-trained model provides a powerful feature extractor for natural language understanding tasks.
Residual Connections: These connections are introduced into the BERT architecture to stabilize training and alleviate the problem of gradient vanishing, enabling the model to train deeper without losing information.
Model Design:

The model uses the transformer layers from multi-lingual BERT as a base and adds six layers of residual connections. This deep structure is crucial for reasoning and contextual understanding tasks.
Additional task-specific output layers, such as classification heads for sentiment analysis or intent classification, are integrated on top of BERT.
Training Process:

The BERT-ResidualModel is fine-tuned on task-specific corpora, with both BERT’s weights and the residual connections being updated to enhance task performance.
Techniques like learning rate warm-up and linear learning rate decay are used to optimize the training process.

DNN 方法
1. 架构：
  - 深度神经网络（DNN）在本实验中由多个全连接神经单元组成的多层结构构成。架构中包含隐藏层，使用激活函数如 ReLU（修正线性单元）来引入非线性，从而使网络能够从输入数据中学习复杂的模式。
  - 损失函数：用于分类任务的交叉熵损失函数，确保模型能够有效区分不同的类别。
2. 训练过程：
  - DNN 使用反向传播和迷你批次随机梯度下降算法进行训练。所使用的优化器是 Adam，这是适用于深度网络训练的，因为它能够动态调整学习率。
    
BERT-ResidualModel 方法
1. BERT 集成：
  - BERT-ResidualModel 结合了BERT的上下文嵌入与残差学习框架的优点。本实验将语音转成文字，再进行文本分析。
  - BERT：BERT在大语料库上预训练，以理解单词出现的上下文，捕捉复杂的语义关系。这个预训练模型提供了一个强大的特征提取器，用于语言理解任务。
  - 残差连接：这些连接被引入到BERT的架构中，以稳定训练并减轻梯度消失问题，使得模型能够在不丢失信息的情况下进行更深的训练。
2. 模型设计：
  - 该模型使用多语言BERT的变压器层作为基础，添加了六层的残差连接。通过深度网络，从而对于推理和上下文理解任务至关重要。
  - 在BERT的基础上集成了额外的任务特定输出层，例如用于情感分析或意图分类的分类头。
3. 训练过程：
  - BERT-ResidualModel 在特定任务的语料库上进行微调，其中BERT的权重与残差连接一起更新，以增强任务性能。
  - 使用学习速率预热和线性学习速率衰减等技术来优化训练过程。
