### 标签转数字

```python
class_name = ['a', 'b', 'c', 'd']
label = ['a', 'a', 'd', 'c', 'b']
label2 = []
for i in range(len(label)):
    label2.append(class_name.index(label[i]))
print(label2)

# output: [0, 0, 3, 2, 1]
```

