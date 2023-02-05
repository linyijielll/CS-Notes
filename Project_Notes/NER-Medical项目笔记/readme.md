## transformer版本问题

注意transformers在不同版本对bert-base-chinese的返回值的变动, **较新的版本返回一个特殊的字典而不是一个元组**

在4.x.x版本中（本项目使用的版本），我们需要使用

```python
embeds, _ = self.bert(sentence, return_dict=False)
```

在3.x.x版本中，我们可以直接使用

```python
embeds, _ = self.bert(sentence)
```



## 一个小疑惑

在这个项目中，我采用 BERT+BiLSTM+CRF 的结构，如果forward方式为

```python
with torch.no_grad():
    embeds, _ = self.bert(sentence, return_dict=False)
out, _ = self.lstm(embeds)
out = self.dropout(out)
feats = self.linear(out)
```

即固定住bert不训练，则模型训练正常，可以正确预测

如果让bert参与训练(代码如下)

```
embeds, _ = self.bert(sentence, return_dict=False)
out, _ = self.lstm(embeds)
out = self.dropout(out)
feats = self.linear(out)
```

训练损失降低较慢，且预测不正常，几乎全部预测为“Other”


