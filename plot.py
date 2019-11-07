import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

data_ncc =np.loadtxt("C:/Users/xwen2/Desktop/DIRNet/result/data.txt")  

steps = data_ncc[:, 0]
training_ncc = data_ncc[:, 1]
validation_ncc = data_ncc[:, 2]

fig = plt.figure(figsize = (7, 5))       
ax1 = fig.add_subplot(1, 1, 1) 

#pl.plot(steps,training_ncc,'g-',label='Dense_Unet(block layer=5)')

p2 = pl.plot(steps,training_ncc,'r-', label = 'training')
pl.legend()

p3 = pl.plot(steps,validation_ncc, 'b-', label = 'validation')
pl.legend()

pl.xlabel('Iterations')
pl.ylabel('ncc)')
plt.title('Registration-Net training vs validating curve')
plt.savefig("C:/Users/xwen2/Desktop/DIRNet/result/batch32_1.png")

