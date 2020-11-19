# Paper_notes
This repository contains the paper notes for medical image, computer vision or any other area.


## Rethinking 
1. [Rethinking the hyperparameters for fine-tuning](https://arxiv.org/pdf/2002.11770.pdf) <br>
   Content: 本文讲了超参数的选择对fine-tune的影响，主要包含学习率、动量和正则项。其中一点，pretrained model的数据集domain (source domain) 与fine-tune数据集domain (target domain)的差距会影响着超参数的选择，尤其是学习率和动量。当source domain和target domain差距很大时，则要用较小的动量，even m=0；反之，则较大的动量（0.9）。这意味着source domain和target domain之间存在着一定的gap，所以才要根据这个gap给定不同的动量来fine-tune。后面还讲到了最优的学习率的结论，也和上述相似。<br>
   idea：那么，我们在做医疗图像的任务时，我们会用pretrained on ImageNet的网络，但是医疗图像和自然图像实在有比较大的差异。<br>
         (1) 这个gap该如何填充？<br>
         (2) 我们应该找一个医疗图像的数据集重新训练一下，缩小这个gap？最后在我们的数据集上fine-tune？<br>
         (3) 还是应该做一个医疗图像自己的pretrained model？<br>
         还有一个问题，还是说我们应该投入精力在hyperparameters的search上？做自动的超参数查找？<br>
         
## Seeded segmentation
首先，seeded segmentation在医疗图像真的很重要吗？

1. 

## Transformer for Computer Vision

越来越多的工作利用transformer来解决CV的问题，例如目标检测、face anti-spoofing和图像分类等等。利用了transformer能够学习不同patch间联系的能力，因此，在医疗图像上是否也可以使用？
