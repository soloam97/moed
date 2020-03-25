from func import *
from PIL import *
from matplotlib import pyplot as plt


def main():
    # Считываем данные из файла h400x300.xcr
    pixels = open_reader_xcr('h400x300.xcr')
    w, h = 300, 400
    print(pixels)

    matrix_of_pixels = []
    i = 0
    for col in range(h):
        temp = []
        for row in range(w):
            temp.append(int(pixels[i]))
            i += 1
        matrix_of_pixels.append(temp)

    print(matrix_of_pixels)

    matrix_of_derivative = []
    for col in range(h):
        temp = []
        for row in range(1, w):
            temp.append(matrix_of_pixels[col][row] - matrix_of_pixels[col][row - 1])
        matrix_of_derivative.append(temp)
    print(matrix_of_derivative)

    derivative_1d = []
    for i in range(h):
        for j in range(w - 1):
            if matrix_of_derivative[i][j] < 0:
                matrix_of_derivative[i][j] = 0
            derivative_1d.append(matrix_of_derivative[i][j])

    image = drawing_image_new(derivative_1d, w - 1, h)
    image = image.transpose(Image.ROTATE_90)
    image.show()

    pixels = image.load()


    specrt = ampl_spectr(matrix_of_derivative[0], len(matrix_of_derivative[0]))

    plt.plot(range(len(specrt)), specrt)
    plt.show()



if __name__ == '__main__':
    main()

