Dear Google recruiting team,
I am writing to express my strong interest in the student researcher position at Google. My name is Jeronim, I'm currently working on my master's degree in computer science with plans of enrolling in a Ph.D. program at the Faculty of electrical engineering and computing (FER), Zagreb, the foremost computer science institution in Croatia. As a highly ambitious computer science student, I'm convinced that Google is the place to be if I want my research to make an impact in the computer vision field

I will try to keep this letter as short as I can but I'm not sure if I will succeed.

Over the past six years, I have developed a keen interest in artificial intelligence, which has driven my fascination for exploring the vast and dynamic field of computer vision. Through my academic pursuits, I have gained a solid foundation in computer science, deep learning, and linear algebra. A further testament to my enthusiasm must be my hobby projects involving computer vision, such as but not limited to, a collaboration with my academic painter friend to generate images based on her paintings. This was one of several huggingface sprints that I have taken part in and I plan to continue to do so in the future. I feel that I would be a valuable addition to Google's research team.  

For the past year, I have worked as a young researcher with the image processing group at FER, Zagreb on a project which was a collaboration between academia and industry with a company called Cloudonia. The team was composed of three professors and four other researchers. We have helped them develop solutions for inventory management in retail environments using computer vision.


The paragraphs below briefly explain a part of my research and represent my ideas for future work if I am to become a student researcher.

For the past 6 months, I have worked on shelf detection. In our case, the main goal of shelf detection in their pipeline is to determine the ordinal number of the shelf on which the product is placed on (most bottom shelf (first), middle shelf (second), ...). We have tried to apply some traditional computer vision methods to this problem but none of them proved to be robust enough to be used in an "in-the-wild" environment such as a store. Exact and precise localization of the shelf bounding boxes wasn't necessary.

For that reason mAP wasn't exactly the right metric for this specific use case of object detection. I have proposed a new LoGT [1] evaluation metric. LoGT represents the predicted bounding box as a line that is going through the middle of that bbox. If that line crosses the ground truth bounding box I declare the loss to be 0 otherwise, it's 1. Keep in mind this is purely an evaluation metric and training is still done using CIoU.
This is a very simple 1-class object detection problem where even the simplest yolov network configuration achieves great results but requires a "large" amount of parameters.
Using my previously proposed metric I managed to prune the memory footprint from 1.4m parameters (yolov5n, the smallest config you can find on the ultralytics yolov5 github repo) to a tiny yolov5femto model of 30k parameters, with the same FPN based architecture but significantly fewer filters in every convolutional layer, achieving the same results.
I feel like this kind of deep learning research can go two ways, one is trying to reduce the complexity while achieving similar results which I have already done; the other is increasing network complexity while trying to achieve better results.
I have recently begun work on a new method of object detection in general which will attempt to improve the results of basic RGB object detection models but at the cost of more parameters. I propose that instead of using only RGB images as inputs to object detection network, I would like to construct additional channel features so that images can become something even more than just 3 channels. I believe that this idea can lay ground for some very exciting computer vision research. Some of "channel features" I think are worth testing out:

1. estimated depth RGB->RGBD (1000x1000x3 input ->1000x1000x4 input), as part of my research I've read many papers that focus on networks for monocular depth estimation such as this one http://yaksoy.github.io/highresdepth/ and I think it would be interesting to use estimated depth as a channel feature in object detection; also open for consideration is applying simple image thresholding to this depth map, the threshold can be a hyperparameter or maybe even a learnable parameter in the network somehow, that would require a slightly different approach.

2. canny edge feature RGB->RGBC, applying the canny edge algorithm on an image and using that grayscale image as a feature, this is my least favorite feature of the 3 but I think it still has enough potential to warrant experiments.

3. applying the Fourier transform to images to construct potentially powerful channel features. As part of my research, I have also read several papers that use NeRFs. Inspired by the way Mildenhall et al. use positional encoding in chapter 5.1 (https://arxiv.org/pdf/2003.08934.pdf) I would like to try out a similar approach - but for object detection.

I feel like a similar approach is being done by controlnet [2] the way they kinda nudge the network towards what they want it to do - i think same results could be achieved in object detection networks as well

If you made it this far into this cover essay I congratulate you and hope to hear from you soon. ĆB, JM

[1] - Line over ground truth metric; source code: https://github.com/jere357/yolov5-RGBD/blob/39ad3cfa5782b5c1aba1cda3b47b7ae2ac9d1b2d/val_jere.py#L524 fell free to contact me by mail if you wanna talk about it - please keep in mind this is all still a work in progress :)
[2] - Lvmin Zhang and Maneesh Agrawala, Adding Conditional Control to Text-to-Image Diffusion Models https://arxiv.org/pdf/2302.05543.pdf
