---
title: GPT
mathjax: true
date: 2025-05-21 12:02:14
categories: LLM
tags: LLM
---


```python
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

# output
torch.Size([32, 65])
tensor(4.8786, grad_fn=<NllLossBackward0>)
```
```python
loss = F.cross_entropy(logits, targets)
```
其实计算的是
$$\text{CrossEntropy}(p,y) = -log(p_y)$$

其中：
$p$ 是 softmax 后的概率分布  
$y$ 是 ground-truth label (目标token)  
$p_y$ 是对应ground-truth 类别的概率  

对于这个Loss如果我们想估计一下是什么水平，那就对比随机猜的情况,，这个可以作为baseline
$$ \text{loss} = -\text{log}(\frac{1}{65}) = \text{log}(65) \approx 4.17 \$$

如果模型刚初始化，或者毫无学习能力，其交叉熵损失函数就会非常接近这个值

### why loss=4.87 > 4.17
这是因为：模型还没开始训练（刚初始化），预测可能不均匀分布，而是更糟糕的“错误分布”，这会导致预测目标token概率 $p_y \lt fra$