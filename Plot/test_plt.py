import matplotlib.pyplot as plt
import numpy as np

xlen = 200
ylen = 100
x = np.arange(xlen)
y = np.arange(ylen)

x1 = np.arange(xlen+1)
y1 = np.arange(ylen+1)

data = np.random.rand(ylen,xlen)
data[0] = 0
data[-1] = 0
data[:,0] = 0
data[:,-1] = 0

fig, axs = plt.subplots(2,1)
axs[0].pcolor(x1,y1,data,shading='flat')
axs[1].pcolor(x,y,data,shading='nearest')
plt.savefig('test.png', dpi=1200)
