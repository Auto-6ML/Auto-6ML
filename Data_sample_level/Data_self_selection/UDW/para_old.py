import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# import matplotlib.font_manager
#
# print(matplotlib.matplotlib_fname())

# import matplotlib.style as style
# style.use('dark_background')
# 生成数据
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.ticklabel_format(style='sci', scilimits=(-4,4), axis='x')
plt.style.use('default')  #ggplot, bmh
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 20}
plt.rc('font', **font)
# plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
fig,axes=plt.subplots(1,1,figsize=(9, 6))

plt.grid(axis="x")
def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1] - src[0])) * (dst[1] - dst[0]) + dst[0]

source_scale = (0, 50)# Scale values between 100 and 600
destination_scale = (40, 50)
# source_scale1 = (600, 6000)# Scale values between 100 and 600
# destination_scale1 = (140, 160)



#Corafull
data_y1 = [80.3, 89.6, 91.2, 91.6, 90.7, 90.3, 90.1] #lamda
data_y2 = [82.1, 84.6, 91.6, 90.2, 88.1, 87.9, 86.0] #DBLP
# data_y3 = [92.0, 92.1, 92.2, 92.6, 92.5, 92.6, 92.1] #yelp
# # data_y3 = [a * 100 for a in data_y3]
# data_y4 = [83.2, 84.0, 88.2, 90.9, 87.2, 84.7, 81.6] #HDMI
# data_y5 = [303,	350,	594,	720] #HeCo
# # data_y5 = [a * 100 for a in data_y5]
# data_y6 = [520,	883,	4380,	4893] #CKD
# data_y7 = [70,	83,	234,	330]

# data_scaledy5 = []
# for x in data_y5:
#     if x < 120:
#         data_scaledy5.append(x)
#     elif 120<x<600:
#         data_scaledy5.append(scale(x, source_scale, destination_scale))
#     elif 600<x:
#         data_scaledy5.append(scale(x, source_scale1, destination_scale1))
#
# data_scaledy6 = []
# for x in data_y6:
#     if x < 120:
#         data_scaledy6.append(x)
#     elif 120<x<600:
#         data_scaledy6.append(scale(x, source_scale, destination_scale))
#     elif 600<x:
#         data_scaledy6.append(scale(x, source_scale1, destination_scale1))
#
# data_scaledy4 = []
# for x in data_y4:
#     if x < 120:
#         data_scaledy4.append(x)
#     elif 120<x<600:
#         data_scaledy4.append(scale(x, source_scale, destination_scale))
#     elif 600<x:
#         data_scaledy4.append(scale(x, source_scale1, destination_scale1))
#
#
# data_scaledy7 = []
# for x in data_y7:
#     if x < 120:
#         data_scaledy7.append(x)
#     elif 120<x<600:
#         data_scaledy7.append(scale(x, source_scale, destination_scale))
#     elif 600<x:
#         data_scaledy7.append(scale(x, source_scale1, destination_scale1))

# data_y7 = [58.9, 58.1,53.8,46.7,39.1,33.1] #gca
x = [1,2,3,4,5,6,7] # 横坐标数据为从0到10之间，步长为0.1的等差数组
# axes.ticklabel_format(style='sci', scilimits=(-4,4), axis='x')
# x = [0.001,0.01,0.1,1,10,100,1000]
axes.set_xticks([1,2,3,4,5,6,7])
# axes.set_xticks([0.001,0.01,0.1,1,10,100,1000])
# axes.set_xticks([1.5,3,4.5,6,7.5,9,10.5])
# axes.set_yticks(fontsize=14)
axes.tick_params(labelsize=12)
# axes.set_ylabel('Time (second)', size=16)
# axes.set_xticklabels(["ACM","IMDB","DBLP","Amazon"],fontsize=14)
axes.set_xticklabels([r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$',r'$10^{2}$',r'$10^{3}$'],fontsize=16)

axes.set_yticks([40,50,60,70,80,90,100])
axes.set_yticklabels(["40",  "50",  "60", "70", "80", "90", "100"],fontsize=18)
axes.set_ylabel('Micro-F1 (%)', size=20)
# axes.set_xlabel('$\lambda$', size=20)
# axes.set_ylim(65,95) #photo

axes.set_ylim(70, 100)

# 生成图形
axes.plot(x, data_y1,label="$\lambda$",marker='P',color='c')
axes.plot(x,data_y2, label="$\eta$",marker='p',color='b')
# axes.plot(x, data_y3, label="DBLP",marker='<',color='g')
# axes.plot(x, data_y4, label="Amazon",marker='>',color='r')
# axes.plot(x, data_scaledy5, label="HeCo",marker='^',color='slategray')
# axes.plot(x, data_scaledy6, label="CKD",marker='*',color='indianred')
# axes.plot(x, data_scaledy7, label="STENCIL",marker='x',color='olive')
# axes.plot(x, data_y7, label="Contrast-Reg",marker='+')

axes.legend(fontsize=15, loc = 'lower right')
# axes.set_title('CoraFull',fontsize=18)
plt.savefig("Parameter_lambda.pdf", dpi = 300, bbox_inches = 'tight')
# 显示图形
plt.show()
