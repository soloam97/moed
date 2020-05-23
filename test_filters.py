from matplotlib import pyplot as plt
from func import *
from PIL import *
from PIL import Image, ImageDraw


# Деконволюция изображения построчно
def main():
    # Загружаем изображение
    file = 'image_model_noise/MODEL.jpg'
    image = read_jpg_gray(file)
    w, h = image.size[0], image.size[1]
    image.show()

    # гистограмма для входного изображения
    hist_image_x, hist_image_y, hist_image_pix = image_histogram(image)


    # Добавление импульсного шума
    Pa, Pb = 0.01, 0.01
    # image_impulse_noise = add_impulse_noise(file, Pa, Pb)
    image_impulse_noise = add_gauss_noise(file, mu=0.2, level=10)
    image_impulse_noise = add_impulse_noise(image_impulse_noise, Pa, Pb)
    image_impulse_noise.show()

    # гистограмма для зашумленного изображения
    hist_impulse_x, hist_impulse_y, hist_impulse_pix = image_histogram(image_impulse_noise)

    fig = plt.figure()

    # plt.subplot(1, 2, 1)
    plt.plot(hist_image_x, hist_image_y)
    plt.title('Model')

    # plt.subplot(1, 2, 2)
    # plt.plot(hist_impulse_x, hist_impulse_y)
    # plt.title('Model+Noise')
    # plt.savefig('Гистограмма Импульсный')
    plt.show()


    pixels = image.load()  # Загружаем значения пикселей в массив pixels
    # Создаем двумерный массив numpy
    pix = []
    for i in range(w):
        for j in range(h):
            pix.append(pixels[i, j])
    matrix_pix = np.array(pix).reshape(w, h)
    print(len(matrix_pix))

    m, dt, fc = 32, 0.0004, 150
    control = high_pass_filter(m, dt, fc=150)

    data_conv = image_conv(matrix_pix, control, w, h, m)

    pix1 = from_2d_to_1d(data_conv, w, h)
    image_filtered = drawing_image_new(pix1, w, h)

    # гистограмма для зашумленного изображения
    hist_filtered_x, hist_filtered_y, hist_filtered_pix = image_histogram(image_filtered)


    plt.plot(hist_filtered_x, hist_filtered_y)
    plt.title('Filtered')
    plt.show()

    # Вывод отфильтрованного изображения
    data_conv = np.rot90(data_conv, axes=(-2, -1))
    plt.imshow(data_conv, cmap='gist_gray', origin='lower')
    plt.title(f'After hpf')
    plt.show()

    # Пороговое преобразование
    for i in range(w * h):
        if pix1[i] > 10:
            pix[i] = 0
        else:
            pix1[i] = 255
    image_new = drawing_image_new(pix1, w, h)
    image_new.show()



if __name__ == '__main__':
    main()

'''
    1) Считываем dat файл изображения
    2) Считываем dat файл с ядром функции
    3) Выводим изображение
    4) Дополняем нулями функицю ядра
    5) Построчно производим Фурье преобразование изображения
    6) Производим преобразование Фурье функции ядра
    7) Считаем действительные и мнимые части функции ядра и каждой строки изображения
    8) Производим построчно комплексное деление изображения и функции ядра
    9) Считаем модули
    9) Берем обратное преобразование Фурье и получаем отфильтрованное изображение от размытия
'''