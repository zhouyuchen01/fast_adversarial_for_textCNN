# fast_adversarial_for_TextCNN
# 复现流程
## 1. 代码部分主要修改
### 1.1 模型部分Model/TextCNN.py/forward
```python
def forward(self, x, attack=None):
    out = self.embedding(x[0])
    if attack is not None:
        out = out + attack
    out = out.unsqueeze(1)
    ...
    out = self.fc(out)
    return out
```
### 1.2 训练部分/adv_train_eval.py/train
#### 1.2.1 PGD
```python
for _ in range(PGD_OPT_NUM):
    outputs = model(trains, delta[:trains[0].size(0)])
    model.zero_grad()
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()

    grad = delta.grad.detach()
    delta.data.uniform_(-epsilon, epsilon)
    delta.data = delta + alpha_1 * torch.sign(grad)
    delta.data[:trains[0].size(0)] = clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
    delta.grad.zero_()
```
#### 1.2.2 FGSM
```python
outputs = model(trains, delta[:trains[0].size(0)])
model.zero_grad()
loss = F.cross_entropy(outputs, labels)
loss.backward()
optimizer.step()

grad = delta.grad.detach()
delta.data.uniform_(-epsilon, epsilon)
delta.data = delta + alpha_1 * torch.sign(grad)
delta.data[:trains[0].size(0)] = torch.clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
delta.grad.zero_()

outputs = model(trains, delta[:trains[0].size(0)])
model.zero_grad()
loss = F.cross_entropy(outputs, labels)
loss.backward()
optimizer.step()

grad = delta.grad.detach()
delta.data.uniform_(-epsilon, epsilon)
delta.data = delta + alpha_1 * torch.sign(grad)
delta.data[:trains[0].size(0)] = torch.clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
delta.grad.zero_()
```
#### 1.2.3 Free
```python
for _ in range(FREE_OPT_NUM):
    outputs = model(trains, delta[:trains[0].size(0)])
    model.zero_grad()
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()

    grad = delta.grad.detach()
    delta.data = delta + epsilon * torch.sign(grad)
    delta.data[:trains[0].size(0)] = torch.clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
    delta.grad.zero_()
```
## 2. 复现过程
```python
python run.py train/test FGSM/PGD/FREE/baseline
```
## 3. 性能分析报告

|         |Precision| Recall| F1-score|
|   ----  |  ----   | ----  |   ----  |
|BaseLine |   0.9999| 0.9999|   0.9999|
|PGD      |   0.9999| 0.9999|   0.9999|
|Free     |   0.9999| 0.9999|   0.9999|
|FGSM     |   0.9999| 0.9999|   0.9999|
