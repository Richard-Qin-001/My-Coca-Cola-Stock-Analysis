*** 很久不写Python了，对基本的列表都有些生疏， 重新梳理一下吧 ***

Python 中列表是 最常用且功能强大的数据结构之一。它是一种 有序、可变（可修改）的集合，用于存储任意类型的数据项。

列表的创建创建列表有几种简单的方式：

使用方括号 []：这是最常见的方式。

```Python
my_list = [1, 'hello', 3.14, True]
empty_list = []
```

使用 list() 构造函数：可以从其他可迭代对象（如元组、字符串、范围）创建列表。

```Python
list_from_tuple = list((1, 2, 3))  # [1, 2, 3]
list_from_string = list("abc")      # ['a', 'b', 'c']
```

列表元素的访问列表是 有序的，这意味着每个元素都有一个确定的位置（索引）。

正向索引：从 $0$ 开始，依次递增。

```Python
data = ['a', 'b', 'c', 'd']
print(data[0])  # 输出: a
print(data[2])  # 输出: c
```

负向索引：从列表末尾开始，-1 表示最后一个元素。

```Python
data = ['a', 'b', 'c', 'd']
print(data[-1]) # 输出: d
print(data[-3]) # 输出: b
```

切片（Slicing）：使用 [start:end:step] 语法获取列表的一个子集。

start（包含）：切片的起始索引，默认为 $0$。

end（不包含）：切片的结束索引，默认为列表长度。

step（步长）：跳过的元素数量，默认为 $1$。

```Python
numbers = [10, 20, 30, 40, 50]
print(numbers[1:4])   # 输出: [20, 30, 40] (索引 1 到 3)
print(numbers[2:])    # 输出: [30, 40, 50] (索引 2 到最后)
print(numbers[:3])    # 输出: [10, 20, 30] (开头到索引 2)
print(numbers[::2])   # 输出: [10, 30, 50] (每隔一个取一个)
print(numbers[::-1])  # 输出: [50, 40, 30, 20, 10] (反转列表)
```

列表元素的修改、添加和删除由于列表是 可变的，你可以直接修改、添加或删除其中的元素。1. 修改元素直接通过索引或切片赋值来修改元素。

```Python
list_a = [10, 20, 30]
list_a[1] = 25  # 修改索引 1 的元素
print(list_a)   # 输出: [10, 25, 30]

list_a[0:2] = [5, 15]  # 通过切片修改多个元素
print(list_a)          # 输出: [5, 15, 30]
```

2. 添加元素

append(item)：在列表 末尾 添加单个元素。

```Python
list_b = ['a', 'b']
list_b.append('c')
print(list_b)  # 输出: ['a', 'b', 'c']
```

insert(index, item)：在指定 索引位置 插入单个元素。

```Python
list_b.insert(1, 'x')
print(list_b)  # 输出: ['a', 'x', 'b', 'c']
```

extend(iterable)：将另一个 可迭代对象（如另一个列表）的所有元素添加到列表 末尾。

```Python
list_b.extend([1, 2])
print(list_b)  # 输出: ['a', 'x', 'b', 'c', 1, 2]
```

使用 + 运算符或切片：+ 运算符会创建一个 新的 列表。

```Python
list_c = [1, 2] + [3, 4]  # [1, 2, 3, 4]
```

3. 删除元素

del 语句：根据 索引 或 切片 删除元素，也可以删除整个列表。

```Python
list_d = [10, 20, 30, 40]
del list_d[1]       # 删除索引 1 的元素 (20)
print(list_d)       # 输出: [10, 30, 40]
del list_d[1:]      # 删除从索引 1 开始的所有元素
print(list_d)       # 输出: [10]
# del list_d        # 删除整个列表，此后 list_d 不再存在
```

pop([index])：删除并 返回 指定索引（默认为最后一个）的元素。

```Python
list_e = ['a', 'b', 'c']
item = list_e.pop(1)  # 删除并返回 'b'
print(item)           # 输出: b
print(list_e)         # 输出: ['a', 'c']
```

remove(value)：删除列表中 第一个 匹配到的 值。

```Python
list_f = [1, 2, 3, 2]
list_f.remove(2)    # 删除第一个 2
print(list_f)       # 输出: [1, 3, 2]
```

clear()：清空列表，使其变为空列表。

```Python
list_g = [1, 2, 3]
list_g.clear()
print(list_g)       # 输出: []
```

列表的其他常用操作len(list)：返回列表的元素个数（长度）。

```Python
print(len(['a', 'b', 'c']))  # 输出: 3
```

count(value)：返回列表中指定 值 出现的次数。

```Python
print([1, 2, 2, 3].count(2))  # 输出: 2
```

index(value, [start, end])：返回指定 值 第一次出现的 索引。如果值不存在会引发 ValueError。

```Python
print(['a', 'b', 'c'].index('b'))  # 输出: 1
```

sort(key=None, reverse=False)：就地（in-place）对列表元素进行排序。reverse=True 为降序。

```Python
num_list = [3, 1, 4]
num_list.sort()
print(num_list)      # 输出: [1, 3, 4]
```

sorted(list)：返回一个 新的 排序后的列表，不修改 原列表。

```Python
num_list = [3, 1, 4]
new_list = sorted(num_list)
print(new_list)      # 输出: [1, 3, 4]
print(num_list)      # 输出: [3, 1, 4] (原列表不变)
```

reverse()：就地 反转列表元素的顺序。

```Python
rev_list = [1, 2, 3]
rev_list.reverse()
print(rev_list)      # 输出: [3, 2, 1]
```

in 运算符：检查某个值是否在列表中。

```Python
print('b' in ['a', 'b', 'c'])  # 输出: True
```

列表的拷贝直接使用 = 赋值只是创建了一个 引用（浅拷贝），修改新列表会影响原列表。

```Python
list1 = [1, 2]
list2 = list1  # 引用
list2[0] = 99
print(list1)   # 输出: [99, 2] (list1 也被修改了)
```

要创建真正的 副本，可以使用以下方法：

切片 [:]：创建浅拷贝。

```Python
list_a = [1, 2]
list_b = list_a[:]
list_b[0] = 99
print(list_a)  # 输出: [1, 2] (原列表未受影响)
```

list() 构造函数：创建浅拷贝。

```Python
list_c = [1, 2]
list_d = list(list_c)
```

copy() 方法：创建浅拷贝。

```Python
list_e = [1, 2]
list_f = list_e.copy()
```