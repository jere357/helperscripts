from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import rgb_to_grayscale
import matplotlib.pyplot as plt
from math import floor


def display_stripe(img):
    plt.imshow(img.permute(1, 2, 0), cmap='gray')
    plt.box(False)
    plt.axis('off')
    plt.show()


def display_img(img):
    plt.imshow(img.permute(1, 2, 0), cmap='gray')
    plt.show()


def extract_stripes(image, num_stripes, stripe_width=10):
    #definiraj listu trakica rangeva [100, 200, 300] and then the slicing looks like:[:,:,100, 100+stripe_width] and so ons
    stripes = []
    range_step = floor(image.shape[2] / num_stripes)
    print(range_step)
    for i in range(1, image.shape[1], range_step):
        #last stripe might be out of bounds so just do this
        if i+stripe_width > image.shape[2]:
            stripes.append(image[:, :, -stripe_width: -1])
            break
        else:
            stripes.append(image[:, :, i : i + stripe_width])
        display_stripe(stripes[-1])
    return stripes


if __name__ == '__main__':
    image = Image.open('images/antonio1.png')
    x = TF.to_tensor(image)
    x.unsqueeze(0)
    x_gs = rgb_to_grayscale(x)
    trakica = x[:, :, 400:405]
    trakica_gs = x_gs[:, :, 400:405]
    """
    display_img(x)
    display_img(x_gs)
    display_stripe(trakica)
    display_stripe(trakica_gs)
    """
    print(x.shape)
    # print(trakica.shape)
    stripes = extract_stripes(x, 9, )
