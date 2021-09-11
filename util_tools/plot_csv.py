import matplotlib.pyplot as plt
file = '/scratch/ssolit/StreamingPerception/f_1M_init_test1/resnet18/init/train_log.txt'
save_dir = ''

x = []
y = []

f = open(file, 'r')
for line in f:
    datapt = line.split(',')
    x.append(float(datapt[0]))
    y.append(float(datapt[1]))

print(x)
plt.plot(x, y)
plt.xlabel('Epoch')
plt.ylabel('Acc1')

plt.savefig(save_dir + 'plot.png')
