import matplotlib.pyplot as plt

def show(img, is_gray = False):
    plt.figure()
    if(is_gray):
        plt.imshow(img, cmap = 'gray')
    else:
        plt.imshow(img)
    plt.show()

