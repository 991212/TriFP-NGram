import matplotlib.pyplot as plt
import numpy as np

sequence_length_set = set()
dataset = 'IC50'
with open(f'../dataset/{dataset}/original data','r') as file:
    count = 0
    for row in file:
        if count == 0:
            count += 1
            continue
        row = row.strip().split('\t')
        sequence = row[6]
        sequence_length_set.add(int(row[7]))
        count += 1

sequence_length_list = list(sequence_length_set)
total_count = len(sequence_length_list)
print("The number of proteins is ",total_count)
sorted_lengths = sorted(sequence_length_list)
target_coverage = int(total_count * 0.8)

# 计算能够覆盖80%-90%数据的序列长度
cumulative_count = 0
target_length = None
for length in sorted_lengths:
    cumulative_count += 1
    if cumulative_count >= target_coverage:
        target_length = length
        break

print("The length that covers 80% of the data:", target_length)

bin_size = 500  # 柱状图的宽度
sequence_max_length = max(sequence_length_list)
print(f"The max length of proteins' sequence({dataset}):", sequence_max_length)
print(f"The mean length of proteins' sequence({dataset}):", round(sum(sequence_length_list) / len(sequence_length_list)))

bins = range(0, (sequence_max_length // bin_size + 2) * bin_size, bin_size)  # 生成bins边界列表
hist, bin_edges = np.histogram(sequence_length_list, bins=bins)  # 计算每个bin中的数量

plt.bar(bin_edges[:-1], hist, width=bin_size, align='edge', edgecolor='k')  # 绘制柱状图，条形的中心移动到bin的中点

xticks_values = range(0, (sequence_max_length // 1000 + 1) * 1000, 1000)  # 设置横轴刻度位置和标签
plt.xticks(xticks_values, [str(value) for value in xticks_values])

plt.tick_params(axis='x', which='both', direction='in')  # 修改刻度线样式，避免刻度线向下突出

for i in range(len(hist)):
    x = bin_edges[i] + bin_size / 2  # 计算柱状图中心的x坐标
    y = hist[i]  # 获取柱状图的高度
    plt.text(x, y, str(y), ha='center', va='bottom')  # 在柱状图中心位置添加高度

plt.title(f'{dataset}')
plt.xlabel('length of sequences(chars)')
plt.ylabel('number of proteins')

plt.xlim([0, bins[-1]])
plt.show()
