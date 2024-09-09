import matplotlib.pyplot as plt
import numpy as np

smiles_length_dic = {}
dataset = 'Ki'
with open(f'../dataset/{dataset}/original data','r')as file:
    count = 0
    for row in file:
        if count==0:
            count+=1
            continue
        row = row.strip().split('\t')
        smile = row[4]
        smiles_length_dic[smile] = row[5]

        count += 1

smiles_length_list = [int(value) for value in smiles_length_dic.values()]
print(f"The numbers of compounds' SMILES({dataset}):",len(smiles_length_list))
# print(sorted(smiles_length_list))

bin_size = 30
smile_max_length = max(smiles_length_list)
print(f"The max length of compounds' SMILES({dataset}):",smile_max_length)
print(f"The mean length of compounds' SMILES({dataset}):",round(sum(smiles_length_list)/len(smiles_length_list)))

bins = range(0, (smile_max_length // bin_size + 2) * bin_size, bin_size)  # 生成bins边界列表
hist, bin_edges = np.histogram(smiles_length_list, bins=bins)  # 计算每个bin中的数量

plt.bar(bin_edges[:-1], hist, width=bin_size, align='edge', edgecolor='k')  # 绘制柱状图，条形的中心移动到bin的中点

xticks_values = range(0, (smile_max_length // 100 + 1) * 100, 100)  # 设置横轴刻度位置和标签
plt.xticks(xticks_values, [str(value) for value in xticks_values])

plt.tick_params(axis='x', which='both', direction='in') # 修改刻度线样式，避免刻度线向下突出

plt.title(f'{dataset}')
plt.xlabel('length of SMILES(chars) ')
plt.ylabel('number of compounds')

plt.xlim([0, bins[-1]])
plt.show()
