Dear Google recruiting team, my name is Jeronim, I'm currently working on my a master's degree in computer science with plans of enrolling in a phd program at the Faculty of electrical engineering and computing, Zagreb, which as far as Croatia is concerned is the best university for computer science students.
I will try to keep this letter as short as i can but I'm not sure if i will succeed.

#ZASTO JE GUGLE SAVRSENO MISTO ZA MENE
#ZASTO SAN JA SAVRSEN LIK ZA GUGL
I am writing to express my strong interest in the student researcher position at Google. As a knowledgeable and ambitious computer science student, I am convinced that Google is the ideal place for me to further my research and achieve my goals as a computer vision researcher. 

For the past year of my life i have worked as a young researcher with the image processing group at FER, Zagreb on a project which was a collaboration between academia and industry with a company called Cloudonia. The team was composed of three professors and four other researchers. We have helped them develop solutions for inventory management in retail environments using computer vision.

The paragraphs below briefly explain a part of my research and represent my ideas for future work if I am to become a student researcher.

For the past 6 months i have worked on shelf detection. In our case, the main goal of shelf detection in their pipeline is to determine the ordinal number of the shelf on which the product is placed on (most bottom shelf (first), middle shelf (second), ...). We have tried to apply some traditional computer vision methods to this problem but none of them proved to be robust enough to be used in an "in-the-wild" environment such as a store. Exact and precise localization of the shelf bounding boxes wasn't necessary.

For that reason mAP wasn't exactly the right metric for this specific use case of object detection. I have proposed a new LoGT [1] evaluation metric. LoGT represents the predicted bounding box as a line that is going through the middle of that bbox. If that line crosses the ground truth bounding box i declare loss to be 0, otherwise it's 1. keep in mind this is purely an evaluation metric and training is still done using CIoU.
This is a very simple 1-class object detection problem where even the simplest yolov network configuration achieves great results but requires a "large" amount of parameters.
Using my previously proposed metric I managed to prune the memory footprint from 1.4m parameters (yolov5n, the smallest config you can find on the ultralytics yolov5 github repo) to a tiny yolov5femto model of 30k parameters, with the same FPN based architecture but significantly less filters in every convolutional layer, achieving the same results.
I feel like this kind of deep learning research can go two ways, one is trying to reduce the complexity while achieving similair results which i have already done; and the other is increasing network complexity while trying to achieve better results.
I have recently began work on a new method of object detection in general which will attempt to improve the results of basic RGB object detection models but at the cost of more parameters. I propose that instead of using only RGB images as inputs to object detection network, i would like to construct additional channel features so that images can become something even more than just 3 channels. Some of "channel features" i think are worth testing out:

1. estimated depth RGB->RGBD (1000x1000x3 input ->1000x1000x4 input), as part of my research i've read many papers that focus on networks for monocular depth estimation such as this one http://yaksoy.github.io/highresdepth/ and i think it would be interesting to use estimated depth as a channel feature in object detection; also open for consideration is applying simple image thresholding to this depth map, the threshold can be a hyperparameter or maybe even a learnable parameter in the network somehow, that would require a slightly different approach.

2. canny edge feature RGB->RGBC, applying the canny edge algorithm on an image and using that grayscale image as a feature, this is my least favorite feature of the 3 but i think it still has enough potential to warrant experiments.

3. applying the fourier transform to images to construct potentially powerful channel features. As part of my research i have also read serveral papers that use NeRFs. Inspired by the way Mildenhall et al. use positional encoding in chapter 5.1 (https://arxiv.org/pdf/2003.08934.pdf) i would like to try out a similair approach - but for object detection.


If you made it this far into this cover essay i congratulate you and hope to hear from you soon. ĆB, JM

[1] - Line over ground truth metric; source code: https://github.com/jere357/yolov5-RGBD/blob/39ad3cfa5782b5c1aba1cda3b47b7ae2ac9d1b2d/val_jere.py#L524 fell free to contact me by mail if you wanna talk about it - keep in mind this is all still a work in progress :)
