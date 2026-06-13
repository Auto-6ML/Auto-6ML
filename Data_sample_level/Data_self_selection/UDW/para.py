#!/usr/bin/env python

# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import mpl_toolkits.mplot3d
import xlrd #读取excel的库
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.ticker as ticker
wb = xlrd.open_workbook("excel/freebase.xlsx")
sheet1 = wb.sheet_by_index(0)
acc = (wb.sheet_by_index(0)).col_values(2)
a = (wb.sheet_by_index(0)).col_values(3)
b = (wb.sheet_by_index(0)).col_values(4)
# a,b = np.mgrid[0.1:1:0.1, 0.1:1:0.1]
acc = acc[1:]
# a = a[1:]
# b = b[1:]
a = [1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7]
# a = [10**-3,10*-2,10**-1,10**0,10**1,10**2,10**3,10**-3,10*-2,10**-1,10**0,10**1,
#      10**2,10**3,10**-3,10*-2,10**-1,10**0,10**1,10**2,10**3,10**-3,10*-2,10**-1,10**0,10**1,10**2,10**3,
#      10**-3,10*-2,10**-1,10**0,10**1,10**2,10**3,10**-3,10*-2,10**-1,
#      10**0,10**1,10**2,10**3,10**-3,10*-2,10**-1,10**0,10**1,10**2,10**3]
b = [1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7]
plt.figure(figsize = (6, 5))
#三维图形
# matplotlib.font_manager._rebuild()
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 10}
plt.rc('font', **font)
# plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams["font.family"] = 'Arial Unicode MS'

# import matplotlib as mpl
# print(mpl.font_manager.get_cachedir())

ax = plt.subplot(projection='3d')
acc = (np.array(acc)).reshape((7,7))
a = (np.array(a)).reshape((7,7))
b = (np.array(b)).reshape((7,7))
ax.plot_surface(a,b,acc*100,rstride=1, cstride=1, color = 'white', cmap = plt.cm.coolwarm, alpha=0.7, linewidth = 50 )
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.zaxis.set_tick_params(labelsize=10)

# plt.axis('off')
# ax.get_xaxis().set_visible(False)
# ax.set_ylim(10**-2, 10**2)
# ax.set_yscale('log')
# 设置y轴刻度，使其以10的幂次方来显示
# ax.set_ylim(10**-3, 10**3)
# ax.set_yscale('log')
# ax.set_xlim(10**-3, 10**3)
# ax.set_xscale('log')
ax.set_xticklabels(["0",r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$',r'$10^{2}$',r'$10^{3}$'],fontsize=10)
ax.set_yticklabels(["0",r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$',r'$10^{2}$',r'$10^{3}$'],fontsize=10)
# ax.set_xticklabels(["0","0.001","0.01","0.1","1","10","100","1000",],fontsize=17)
# ax.set_yticklabels(["0","0.001","0.01","0.1","1","10","100","1000",],fontsize=17)
#设置坐标轴标签
#ax.view_init(elev=30,azim=125)  #
# ax.view_init(elev=30,azim=320)   #
# ax.view_init(elev=30,azim=240)  #
# ax.view_init(elev=30,azim=240)   #
# ax.view_init(elev=30,azim=240)
# ax.view_init(elev=30,azim=220)  #
# ax.view_init(elev=30,azim=220)   # arxiv  produ photo computer cora cite pub
ax.view_init(elev=30,azim=220)
# plt.zticks(np.linspace(29, 32, 5))
# ax.set_xticks([1,2,3,4,5,6,7])
# ax.set_yticks([1,2,3,4,5,6,7])
# ax.view_init(elev=30,azim=40) #maga
ax.set_xlabel(r'$\lambda$', fontsize = 12)
ax.set_ylabel(r'$\eta$', fontsize = 12)
ax.set_zlabel('Macro-F1(%)', fontsize = 12)
# ax.set_xticks(size = 18)
# ax.set_zlim(60, 95)  #acm
# ax.set_zlim(30, 60)  #imdb
# ax.set_zlim(80, 95) #dblp
ax.set_zlim(50, 65)   #freebase
# ax.set_zlim(40, 82.5) #pubmed
# ax.set_zlim(40, 85)   #cora
# ax.set_zlim(20, 93.5) #photo
# ax.set_zlim(84, 89)  #computers
# plt.title('Photo', fontsize = 20)
plt.savefig("freebase_para_test.pdf", dpi = 300, bbox_inches = 'tight')
plt.show()
# In[ ]:




