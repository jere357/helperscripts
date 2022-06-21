from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import rgb_to_grayscale
import matplotlib.pyplot as plt
from math import floor
import torch

def pad_stripes(stripes, padding_color = 'black'):
    padded_stripes = []
    matrix_dimensions = stripes[0].shape
    for stripe in stripes:
        print(stripe.shape)
        if(padding_color == 'black'):
            padding_top = torch.zeros((int(matrix_dimensions[0]), floor(matrix_dimensions[1]/2), floor(matrix_dimensions[2])))
            padding_bot = torch.zeros((int(matrix_dimensions[0]), floor(matrix_dimensions[1]/2), floor(matrix_dimensions[2])))
        elif(padding_color == 'white'):
            padding_top = torch.ones((int(matrix_dimensions[0]), floor(matrix_dimensions[1]/2), floor(matrix_dimensions[2])))
            padding_bot = torch.ones((int(matrix_dimensions[0]), floor(matrix_dimensions[1]/2), floor(matrix_dimensions[2])))
        padded_stripes.append(torch.cat((padding_top, stripe, padding_bot), dim=1))
        #print(padded_stripe.shape)
    return padded_stripes


def display_stripe(img):
    plt.imshow(img.permute(1, 2, 0), cmap='gray')
    plt.box(False)
    plt.axis('off')
    plt.show()


def display_img(img):
    plt.imshow(img.permute(1, 2, 0), cmap='gray')
    plt.show()


def extract_stripes(image, num_stripes, stripe_width=10):
    stripes = []
    range_step = floor(image.shape[2] / num_stripes)
    for i in range(1, image.shape[1], range_step):
        if i+stripe_width > image.shape[2]:
            stripes.append(image[:, :, -stripe_width-1: -1])
            break
        else:
            stripes.append(image[:, :, i : i + stripe_width])
        #display_stripe(stripes[-1])
    return stripes



if __name__ == '__main__':
    image = Image.open('images/antonio1.png')
    x = TF.to_tensor(image)
    x.unsqueeze(0)
    stripes = extract_stripes(x, 9, )
    padded_stripes = pad_stripes(stripes)
    display_stripe(stripes[0])
    display_stripe(padded_stripes[0])
