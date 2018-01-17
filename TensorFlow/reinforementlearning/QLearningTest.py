'''
使用一个迷宫小游戏模拟qlearning过程：
迷宫大小n*n（n=7)
i   ###
# #  ##
#### ##
#     #
# # # #
### # #
##### o
i为入口o为出口
'''
import mpl_toolkits.mplot3d
import numpy as np
import matplotlib.pyplot as plt

x, y = np.mgrid[-2:2:50j, -2:2:50j]
z = x * np.exp(-x ** 2 - y ** 2)

ax = plt.subplot(111, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
