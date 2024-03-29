# Paper_notes
This repository contains the paper notes for medical image, computer vision or any other area.
This repository is regarded as a notebook. Please feel free to correct me. 

Outlines:
* [Hierarchical Clustering](#HC)


## Rethinking 
1. [Rethinking the hyperparameters for fine-tuning](https://arxiv.org/pdf/2002.11770.pdf) <br>
   Content: 本文讲了超参数的选择对fine-tune的影响，主要包含学习率、动量和正则项。其中一点，pretrained model的数据集domain (source domain) 与fine-tune数据集domain (target domain)的差距会影响着超参数的选择，尤其是学习率和动量。当source domain和target domain差距很大时，则要用较小的动量，even m=0；反之，则较大的动量（0.9）。这意味着source domain和target domain之间存在着一定的gap，所以才要根据这个gap给定不同的动量来fine-tune。后面还讲到了最优的学习率的结论，也和上述相似。<br>
   idea：那么，我们在做医疗图像的任务时，我们会用pretrained on ImageNet的网络，但是医疗图像和自然图像实在有比较大的差异。<br>
         (1) 这个gap该如何填充？<br>
         (2) 我们应该找一个医疗图像的数据集重新训练一下，缩小这个gap？最后在我们的数据集上fine-tune？<br>
         (3) 还是应该做一个医疗图像自己的pretrained model？<br>
         还有一个问题，还是说我们应该投入精力在hyperparameters的search上？做自动的超参数查找？<br>
         
<!-- ## Seeded segmentation
首先，seeded segmentation在医疗图像真的很重要吗？
1.  -->

## Transformer for Computer Vision

越来越多的工作利用transformer来解决CV的问题，例如目标检测、face anti-spoofing和图像分类等等。利用了transformer能够学习不同patch间联系的能力，因此，在医疗图像上是否也可以使用？
1. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy)

2. [How to augment your ViTs? Consistency loss and StyleAug, a random style transfer augmentation](https://arxiv.org/pdf/2112.09260.pdf)
该论文提出如何通过一些CNN中数据增强的technique去提升ViTs的性能。主要包括基于风格迁移的数据增强，以及增强图像的预测结果的consistency loss。
augmented consistency loss: (1) classic version: Loss = CE(f(x_aug), label) (2) JSD version: Loss = CE(f(x_orig), label) + λ JSD(f(x_orig)|| f(x_aug1
)|| f(x_aug2))

3. [StyleSwin: Transformer-based GAN for High-resolution Image Generation](https://arxiv.org/pdf/2112.10762v1.pdf)
该论文提出了类似stylegan框架但由transformer block组成生成网络。其性能超过stylegan2。

4. [On Efficient Transformer and Image Pre-training for Low-level Vision](https://arxiv.org/pdf/2112.10175v1.pdf)
该论文提出了新的基于encoder-decoder-based transformer的架构。在不同的任务间，有不同的encoder-deocder pair，但是共享同一个transformer stage，从而利用其他任务达到预训练transformer stage的目的。

5. [The Nuts and Bolts of Adopting Transformer in GANs](https://arxiv.org/pdf/2110.13107.pdf?ref=https://githubhelp.com)<br/>
该论文提出了一个关于GAN的transformer的网络结构，减少了artifacts。<br/>

## Image Translation

//该论文提出利用一个预定义的mask让网络选择性的优化mask的部分。在SR中，我们也可以用mask将前景和背景分离，分开predict。

## Domain adaptation
1. [ONE-SHOT GENERATIVE DOMAIN ADAPTATION](https://arxiv.org/pdf/2111.09876.pdf)<br/>
该论文目标是实现一个zero-shot的DA生成模型。如图，先训练一个GAN学习source domain的图像生成；接着，利用一个adaptor将输入的feature映射到target domain的feature；而此时，判别器还不具备分类target domain的能力，因此在判别器后加一个分类器。该模型可以在大量数据的预训练模型上finetune，保持了原本模型所学得的基本pattern，还可以减少训练的时间。<br/>
<img src="./figs/one_shot_generative_DA.png " width = "500" />


## GAN
1. [Ensembling Off-the-shelf Models for GAN Training](https://arxiv.org/pdf/2112.09130.pdf) <br/>
在判别器中，ensemble多个pretrained model的特征，来提升其判别能力，从而提升生成图像的质量。<br/>
<img src="./figs/ensembling_gan_training.png"  />

2. [Ensembling with Deep Generative Views](https://arxiv.org/pdf/2104.14551.pdf)<br/>
与论文1相对应，针对生成器的结果进行ensemble，从而提升分类器的性能。该文中，先把图像映射到一个latent vector,针对这个vector加入一些小的扰动，再重新输入到图像中，得到不同的图片。接着，把这些生成的图像分别输入到分类其中，再将其结果进行ensemble，作为最终的结果。<br/>
<img src="./figs/ensembling_generative_views.png" width = "400" />

3. [GAN-Supervised Dense Visual Alignment](https://arxiv.org/pdf/2112.05143.pdf)<br/>
该论文中，希望网络学习到目标图像的形状的位置，如猫头的位置，从而进行下一步的图像编辑。这里为了猫的形状，利用了一个spatial transformer network 来学习输入图像的形变参数，从而根据形变参数获得位置关系。在训练STN时，固定生成器的参数，利用peceptual loss计算生成图像和目标图像的距离，使得STN学习到最好的形变参数。
<br/>
<img src="./figs/gan_supervised_dense_visual_alignment0.png" width = "500" />
<img src="./figs/gan_supervised_dense_visual_alignment.png" width = "500" />

4. [Understanding the role of individual units in a deep
neural network](https://www.pnas.org/content/pnas/117/48/30071.full.pdf)<br/>
该论文中，分析了VGG16/GAN网络中不同层参数学习到的信息，包括物体形状、区域、材质和颜色。<br/>
<img src="./figs/understanding_uints_dnn.png" width = "500" />

5. [Rewriting a Deep Generative Model](https://arxiv.org/pdf/2007.15646.pdf)<br/>
该论文认为网络的权重在学习一些跟pattern相关的规则，通过缩小目标pattern和输入区域的特征差异，从而达到替换目标pattern的目的。<br/>
<img src="./figs/rewriting_generative_model.png" width = "500" />

6. [Improving GAN Equilibrium by Raising Spatial Awareness](https://arxiv.org/pdf/2112.00718.pdf) <br/>
本文提出了利用GradCAM提取判别器的热力图，从而向生成器提供一些空间信息，最终提升GAN的平衡<br/>
<img src="./figs/improve_gan_equilibrium.png" width = "500" />

7. [Positional Encoding as Spatial Inductive Bias in GANs](https://arxiv.org/pdf/2012.05217.pdf) <br/>
在GAN的编码过程中，存在一些空间bias，而这些bias是由zero padding带来的。<br/>

8. [STYLEALIGN: ANALYSIS AND APPLICATIONS
OF ALIGNED STYLEGAN MODELS](https://openreview.net/pdf?id=Qg2vi4ZbHM9)(ICLR 2022)<br/>
现有的图像编辑的工作中，有许多基于StyleGAN进行fine-tune的方法，实质上就是迁移学习。该论文分析了这些方法是如何继承父模型的参数，在finetune过程中又是如何修改这些参数的。<br/>
 
## Self-Supervised Representation Learning

1. [Self-Distilled Self-Supervised Representation Learning](https://arxiv.org/pdf/2111.12958.pdf)
这篇论文介绍了如何提升互信息的upper bound。首先，两张图像的最大互信息肯定是他们的content C。其次，因为随着网络层数增加，其提到的特征就更加抽象，和content偏移也就更远，所以默认浅层的feature f1_l相比深层的feature f1_L更加接近content。因此，作者认为I(f1_L, f2_L) <= I(f1_l, f2_L), （I表示互信息），即深层之间的互信息 小于 浅层和深层间的互信息。进而认为，I(f1_l, f2_L)这便是互信息的上界。

2. [Are Large-scale Datasets Necessary for Self-Supervised Pre-training?](https://arxiv.org/pdf/2112.10740v1.pdf)
这篇论文应该是Masked auto-endocder那篇文章的延伸。在同样的框架上，同样的图片输入encoder，decoder分别输入两个不同的masked encoded features，并重构其masked的部分。重构后，两个不同的结果计算互信息损失，提升特征的相似性。该方法在COCO上pretrained的效果竟然比在imagenet的效果有比较大的提升。

3. [Demystifying Unsupervised Semantic Correspondence Estimation](https://arxiv.org/pdf/2207.05054v1.pdf)
![image](https://user-images.githubusercontent.com/16815652/178417059-5dab9f80-ac45-4ae7-99e5-0f6b25fb1abb.png)
![image](https://user-images.githubusercontent.com/16815652/178417125-cce2a531-828e-4bed-8629-082d07fc76c1.png)

## Data Bias

1. [MITIGATING THE BIAS OF CENTERED OBJECTS IN COMMON DATASETS](https://arxiv.org/pdf/2112.09195.pdf)<br>
该论文希望通过一种新的数据增强的方式缓解物体位于中心的bias。

## Basic Components for Training
1. [ADAPTIVELY CUSTOMIZING ACTIVATION FUNCTIONS FOR VARIOUS LAYERS](https://arxiv.org/pdf/2112.09442.pdf)<br>
该论文提出了一个自适应的激活函数，用于提高收敛速度和性能。例子如下：<br>
ASimoid : fAsimoid = biSigmoid(aiz + ci) + di<br>
ATanh : fAtanh = bitanh(aiz + ci) + di<br>
AReLU : fArelu = maximum(aiz + ci, biz + di) <br>

## Lifelong learning
1. [AN EMPIRICAL INVESTIGATION OF THE ROLE OF PRETRAINING IN LIFELONG LEARNING](https://arxiv.org/pdf/2112.09153.pdf)<br>
This paper found that lifelong learning methods should focus on learning generic features instead of simply focusing on alleviating catastrophic forgetting, as generic features appear to undergo minimal forgetting. 

## CLIP
1. [Align and Prompt: Video-and-Language Pre-training with Entity Prompts](https://arxiv.org/pdf/2112.09583.pdf)<br>
该论文介绍了一个视频文字的与训练框架。在预训练的过程中，通过对齐视频和文字中的多个实体，来提升在下游任务中的性能。相比现有的方法，它不需要检测器，这是关键的地方。

2. [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/pdf/2112.10741v1.pdf)<br>
该论文介绍了一个可以根据文字和mask编辑图像的网络。

3. [What is Where by Looking: Weakly-Supervised Open-World Phrase-Grounding without Text Inputs](https://arxiv.org/pdf/2206.09358.pdf)<br>
<img src="https://user-images.githubusercontent.com/16815652/180938977-1c0c7239-de23-49cf-acb0-476665eb881a.png" width = "500" />

4. [Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification](https://arxiv.org/pdf/2207.09519.pdf)<br>
<img src="https://user-images.githubusercontent.com/16815652/180939126-edb9f705-6235-42e8-a4ac-5073e6559ce7.png" width = "500" />
<img src="https://user-images.githubusercontent.com/16815652/180939321-068cc993-0982-4506-bb43-0ea1bdd7924e.png" width = "300" />
<br/>
本文提出了一个不需要训练，能够完成few-shot分类任务的框架。利用CLIP visual encoder提取特征，计算few-shot image features 和test image features的affinity map，再把affinity map对应到不同的类别，获得predicted label。

5. [Prototypical Contrastive Language Image Pretraining](https://arxiv.org/pdf/2206.10996.pdf)<br>
Push student representations towards their cross-modal prototypes.<br> 
Group the student representation to their within-modal centroid. <br>
Ensemble multiple teachers to guide student representation.<br>
<img src="https://user-images.githubusercontent.com/16815652/181210681-cf22a029-423c-4dc1-9df4-6d5cd573b8d8.png" width = "500" />
<img src="https://user-images.githubusercontent.com/16815652/181210758-c0774fcf-dc13-4364-9735-aa1dd8bfc2d7.png" width = "500" />

6. [Curriculum Learning for Data-Efficient Vision-Language Alignment](https://arxiv.org/pdf/2207.14525v1.pdf)<br>
构造不同难度的contrastive task使得visual和textual reprentation对齐。
<img src="./figs/curi_learning_for_CLIP.jpg" width="400"/>

7. [Learning Visual Representation from Modality-Shared Contrastive Language-Image Pre-training](https://arxiv.org/pdf/2207.12661.pdf)<br>
该论文提出了文本和图像共享更多的模型参数，会带来更大的性能提升。利用共享的参数，可以让两者的语义概念在特征空间更加接近。
<img src="./figs/MS_CLIP.jpg" width = "500" />

8. [Prompt-aligned Gradient for Prompt Tuning](https://arxiv.org/pdf/2205.14865.pdf)<br>
在CLIP中，我们常常会让prmopt模版可以学习，从而提升下游任务的性能。但是fine-tune的过程中，有可能会忘掉之前的预训练模型的知识。因为在fine-tune的过程中，计算fine-tune模型的预测p，并用ce loss来优化模型，其中有可能优化的时候偏离原始的知识。为了避免这个问题，计算了 p 和 基于原始预训练模型zero-shot prediction的KL散度（KL散度代表了与原始知识偏离的程度），比较KL散度的梯度G_{KL}和ce loss的梯度G_{ce}，如果小于1说明CE loss的优化的方向是正确的，反之亦然。<br>
<img src="./figs/prmopt_gradient.jpg" width = "500" />

9. [CYCLIP: Cyclic Contrastive Language-Image Pretraining](https://arxiv.org/pdf/2205.14459.pdf)<br>
CYCLIP is cyclic consistent between image-text pairs as the in-modal distances, d(Tcat, Tdog) ∼ d(Icat, Idog), and the cross-modal distances, d(Tcat, Idog) ∼ d(Icat, Tdog), are similar to each other unlike CLIP.<br>
<img src="./figs/cyclip1.jpg" width = "500" />
<img src="./figs/cyclip2.jpg" width = "500" />

10. [Open-world Semantic Segmentation via Contrasting and Clustering Vision-Language Embedding](https://arxiv.org/pdf/2207.08455.pdf)<br>
本文关键是提出了一个 online clustering loss, 将原图和增强的图像作为成对的cluster pixel pair，并最大化两种图像对应的cluster map的互信息，从而让模型学会哪些像素是同一个类。在预测的时候，将同一个cluster的pixel特征和不同类别文本匹配，即可获得对应cluster的类别<br>
<img src="./figs/ViL-Seg.jpg" width = "500" />

11. [Learning Hierarchy Aware Features for Reducing Mistake Severity](https://arxiv.org/pdf/2207.12646.pdf)<br>
It proposes a probabilistic approach using to learn hierarchy-aware features that respect the label hierarchy in the feature space and thereby make semantically meaningful mistakes<br>
<img src="./figs/HAF.jpg" width = "500" />
<img src="./figs/haf2.jpg" width = "200" />


## NLP
1. [Hypergraph Transformer: Weakly-Supervised Multi-hop Reasoning for Knowledge-based Visual Question Answering](https://aclanthology.org/2022.acl-long.29.pdf)<br>
<img src="https://user-images.githubusercontent.com/16815652/180947194-e42808f4-0b0e-4256-beed-d8f3dd3316ad.png" width = "500" />
Use hpyergraph in transformer for VQA task.


## NLP for Medical
1. [Explaining Chest X-ray Pathologies in Natural Language](https://arxiv.org/pdf/2207.04343v1.pdf)<br>
介绍了一个NLE的医疗图像数据集，从报告中提取关键信息。
<img src="https://user-images.githubusercontent.com/16815652/178416400-fe902faf-d149-411d-940f-68b8ae034301.png" width = "500" />


## Knowledge Distillation
1. [Pixel Distillation: A New Knowledge Distillation Scheme for Low-Resolution Image Recognition](https://arxiv.org/pdf/2112.09532.pdf)<br>
本文的动机是希望（1）在降低模型输入图像的分辨率的同时，还能保证低分辨率图像特征和高分辨率图像特征相似，另外（2）还希望能够达到压缩模型参数的目的。<br>
因此，为了动机（2）该论文提出了除了老师和学生模型以外，还提出加入一个助教模型，用于减少模型的参数；为了动机（1）本文提出了在学生和助教模型之间进行像素级的知识蒸馏，保证特征表示与高分辨率的特征一致。


## Medical Image Analysis
1. [Unified 2D and 3D Pre-training for Medical Image Classification and Segmentation](https://arxiv.org/pdf/2112.09356.pdf)<br>
该论文提出了打破2d和3d的gap，并且先在两种数据上先预训练，然后各自下游任务上再fine tune。<br>

2. [Exploring Contextual Relationships for Cervical Abnormal Cell Detection](https://arxiv.org/pdf/2207.04693v1.pdf)<br>
利用上下文信息帮助检测。

## RL
1. [Off-Policy Reinforcement Learning for Efficient
and Effective GAN Architecture Search](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520171.pdf)<br/>
如题，本文讲了如何利用RL搜索GAN的结构。<br/>

<h2 id="HC">Hierarchical Clustering</h2>

1. [Learning Hierarchical Graph Neural Networks for Image Clustering](https://openaccess.thecvf.com/content/ICCV2021/papers/Xing_Learning_Hierarchical_Graph_Neural_Networks_for_Image_Clustering_ICCV_2021_paper.pdf) 
(ICCV2021)<br/>
<img src="figs/hc1.jpg" width = "500" /><br/>

2. [Contrastive Multi-view Hyperbolic Hierarchical Clustering](https://arxiv.org/pdf/2205.02618.pdf)<br/>
<img src="figs/hc2.jpg" width = "400" /><br/>

3. [Hierarchically Clustered Representation Learning](https://arxiv.org/pdf/1901.09906.pdf)<br/>
<img src="figs/hc3.jpg" width = "400" /><br/>

4. [Hierarchical Clustering with Hard-batch Triplet Loss for Person Re-identification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zeng_Hierarchical_Clustering_With_Hard-Batch_Triplet_Loss_for_Person_Re-Identification_CVPR_2020_paper.pdf)<br/>
<img src="figs/hc4.jpg" width = "200" /><br/>

5. [Unsupervised Hierarchical Semantic Segmentation with Multiview Cosegmentation and Clustering Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Ke_Unsupervised_Hierarchical_Semantic_Segmentation_With_Multiview_Cosegmentation_and_Clustering_Transformers_CVPR_2022_paper.pdf)<br/>
<img src="figs/hc5.jpg" width = "350" /><br/>


