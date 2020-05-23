import numpy as np
from matplotlib import pyplot as plt
from func import anti_muar, open_reader_xcr, drawing_image_new

# x = np.linspace(0, 0.5, 100)
# y = np.random.uniform(1, 5, 100)
# z = np.array([x, y])
# # print(z)
# a = []
# for i in range(100):
#     a.append((x[i], y[i]))
# print(len(a))
# a = np.array(a)
#
# plt.plot(a[:, 0], a[:, 1])
# plt.show()

image = open_reader_xcr('h400x300.xcr')
width, height = 400, 300
matrix_pix = np.array(image).reshape(height, width)
image = drawing_image_new(matrix_pix, width, height)
image_new = anti_muar(image)
image_new.show()