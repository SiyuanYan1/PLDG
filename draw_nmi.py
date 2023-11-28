
import matplotlib.pyplot as plt

# 读取数据
with open("results/epvt_c4_l1_d/nmi.txt") as f:
    lines = f.readlines()

# 解析数据
class_labels = []
domain_labels = []
prev_assignment = []
epochs = []

for line in lines:
    parts = line.strip().split(', ')
    epoch = int(parts[0].split(': ')[1])
    nmi_class = float(parts[1].split(': ')[1])
    nmi_domain = float(parts[2].split(': ')[1])
    nmi_prev = float(parts[3].split(': ')[1])
    epochs.append(epoch)
    class_labels.append(nmi_class)
    domain_labels.append(nmi_domain)
    prev_assignment.append(nmi_prev)

# 绘制折线图
plt.plot(epochs, class_labels, marker='^', label='Class Labels')
plt.plot(epochs, domain_labels, marker='o', label='Domain Labels')
plt.plot(epochs, prev_assignment, marker='s', label='Previous Assignment')

# 添加图例和标签
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('NMI')
plt.title('NMI Comparison')

# 显示图形
plt.show()
