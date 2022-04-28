# fast_adversarial_for_TextCNN
# 复现流程
# 1. 代码部分主要修改
## 1.1 模型部分Model/TextCNN.py/forward
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
## 1.2 训练部分/adv_train_eval.py/train
### 1.2.1 PGD
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
### 1.2.2 FGSM
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
### 1.2.3 Free
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
## 1.3 可视化/adv_train_eval.py/evaluate&plot_confusion_matrix
```python
if test:
    
    report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    #plot confusion matrix
    plot_confusion_matrix(confusion, config.class_list, 'confusion_matrix.png', title='confusion matrix')
    return acc, loss_total / len(data_iter), report, confusion
```
```python
def plot_confusion_matrix(cm, classes, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap='summer')
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()
```


# 2. 复现过程
```python
python run.py --mode [FGSM/PGD/FREE/Baseline] --seed [int]
```
# 3. 性能分析报告

## 3.1 性能对比
|         |Precision| Recall| F1-score| Train_time| Train_time per Iter|
|   ----  |  ----   | ----  |   ----  |     ----  |     ----------     |
|BaseLine |   0.9119| 0.9118|   0.9117|    0:12:08|   0.1193s(6100 It) |
|PGD      |   0.9079| 0.9080|   0.9077|    0:42:55|   0.4858s(5300 It) |
|Free     |   0.8852| 0.8845|   0.8844|    0:48:08|   0.4443s(6100 It) |
|FGSM     |   0.9166| 0.9165|   0.9163|    0:37:46|   0.2490s(9100 It) |
## 3.2 Confusion Matrix

## 3.3 实验结论
### 发现与结论
+单次实验（seed==1），Precision/Recall/F1-score：FGSM>BaseLine>PGD>Free
+单次实验（seed==1），Train_time per Iter(训练效率)：BaseLine>FGSM>Free>PGD
+如何选择：考虑鲁棒性，优先选择FGSM；考虑训练效率，则优先BaseLine。

### 局限性
+限于时间原因，沿用了“超过1000batch效果无法提升，则提前结束训练”的设置，因此PGD、Free、FGSM方法相对BaseLine并无明显提升。
+PGD、Free、FGSM三类对抗算法性能优劣会受到超参数、随机数、样本、具体应用问题等诸多因素的影响，因此上述发现仅可作为参考；三类对抗算法优劣还需更多实验的检验。
