from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import rgb_to_grayscale
import matplotlib.pyplot as plt
from math import floor
import seaborn as sns
import torch
from torch import nn
import numpy as np
from torch.nn.functional import conv2d as conv2d
from numpy import diff

def pad_stripes(stripes, padding_color = 'white'):
    padded_stripes = []
    matrix_dimensions = stripes[0].shape
    for stripe in stripes:
        #print(stripe.shape)
        if(padding_color == 'black'):
            padding_top = torch.zeros((int(matrix_dimensions[0]), floor(matrix_dimensions[1]), floor(matrix_dimensions[2])))
            padding_bot = torch.zeros((int(matrix_dimensions[0]), floor(matrix_dimensions[1]), floor(matrix_dimensions[2])))
        elif(padding_color == 'white'):
            padding_top = torch.ones((int(matrix_dimensions[0]), floor(matrix_dimensions[1]), floor(matrix_dimensions[2])))
            padding_bot = torch.ones((int(matrix_dimensions[0]), floor(matrix_dimensions[1]), floor(matrix_dimensions[2])))
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


def load_image(path):
    image = Image.open(path)
    img_tensor = TF.to_tensor(image)
    img_tensor.unsqueeze(0)
    return img_tensor


def autocorrelate(input, weights):
    result = conv2d(input, weight=weights)
    return result.item()


def autocorrelate_padded(input, weights):
    result_array = []
    #print(input.shape)
    #print(weights.shape)
    for i in range(weights.shape[2]*2):
        sub_stripe = input[:,i:i+weights.shape[2],:]
        result_array.append(autocorrelate(sub_stripe, weights))
    #plt.plot(result_array)
    #plt.show()
    return result_array
"""
def get_range(a, b, offset):
    return list(range((offset + a): (offset + b)))
"""
if __name__ == '__main__':
    #sns.set()
    image_tensor = load_image('images/fejk_polica_ali_5.jpg')
    #TODO: n-1 stripeova iz nekog razloga idk pogl posli
    stripes = extract_stripes(image_tensor, num_stripes = 4, stripe_width=30)
    padded_stripes = pad_stripes(stripes, padding_color='black')
    image_results = []
    for stripe, stripe_padded in zip(stripes, padded_stripes):
        display_stripe(stripe)
        display_stripe(stripe_padded)
        result = autocorrelate_padded(stripe_padded, stripe.unsqueeze(0))
        image_results.append(result)
        x = list(range(len(result)))
        dydx = diff(result)/diff(x)
        fig ,(ax1,ax2, ax3) = plt.subplots(1, 3)
        x = list(range(len(result)))[1:]
        auu = np.argwhere([round(number) for number in dydx] == 0)
        ax3.hist([round(number) for number in dydx])
        fig.suptitle('stripe i rezultat autokorelacije')
        ax1.imshow(stripe.permute(1,2,0))
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax2.plot(result, range(len(result)))
        ax2.set_yticklabels([])
        fig.show()
    #x = list(range(len(image_results)))
    dydx = diff(image_results)/diff(x)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('polica i graf korelacije po stripeovima')
    ax1.imshow(image_tensor.permute(1,2,0), aspect='auto')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax3.plot(x,dydx)
    for i, stripe_result in enumerate(image_results):
        ax2.plot(np.array(stripe_result) + 5000*i, range(len(result)))
        ax2.set_yticklabels([])
    fig.show()


    #display_stripe(padded_stripes[0])
    #print(padded_stripes[0].shape)
    #print(image_tensor.shape[1:])
    #display_stripe(stripes[2])
    """
    a = conv_layer.weight.detach().numpy()[0][0]
    b =  conv_layer.weight.detach().numpy()[0][1]
    c = conv_layer.weight.detach().numpy()[0][2]
    plt.plot(a)
    plt.show()
    plt.plot(b)
    plt.show()
    plt.plot(c)
    plt.show()
    """
