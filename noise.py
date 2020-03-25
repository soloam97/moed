from func import *
import numpy as np


def derivative(matrix_pix, w, h):
    data = []
    for i in range(w):
        row = []
        for j in range(h - 1):
            row.append(matrix_pix[i][j + 1] - matrix_pix[i][j])
        data.append(row)

    return data


def image_conv(input_mass, control_mass, w, h, m):
    data_conv = []
    for i in range(w):
        temp = convolution(normalize_pixels(input_mass[i], 255), control_mass)
        data_conv.append(temp[m:(h + m)])

    return data_conv


def main():
    image = open_reader_xcr('h400x300.xcr')
    w, h = 300, 400
    matrix_pix = np.array(image).reshape(w, h)
    pix = from_2d_to_1d(matrix_pix, w, h)

    # Создаем изображение 400x300 (поворот на 90 градусов)
    input_image = drawing_image_new(pix, w, h).transpose(Image.ROTATE_90)
    input_image.show()

    data_diff = derivative(matrix_pix, w, h)
    pix = from_2d_to_1d(data_diff, w, h - 1)

    # Создаем изображение 400x300 (поворот на 90 градусов)
    input_image = drawing_image_new(pix, w, h - 1).transpose(Image.ROTATE_90)
    input_image.show()

    C = ampl_spectr(data_diff[0], 399)
    plt.plot(range(len(C)), C)
    plt.title('Спектр')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.show()

    fs = 400
    dt = 1 / fs
    m, fc1, fc2 = 64, 100, 130
    input_mass = matrix_pix
    control_mass = bend_stop_filter(m, dt, fc1, fc2)

    conv_matrix_pix = image_conv(input_mass, control_mass, w, h, m)
    conv_pix = from_2d_to_1d(conv_matrix_pix, w, h)

    # Создаем изображение 400x300 (поворот на 90 градусов)
    conv_image = drawing_image_new(conv_pix, w, h).transpose(Image.ROTATE_90)
    conv_image.show()


if __name__ == '__main__':
    main()
