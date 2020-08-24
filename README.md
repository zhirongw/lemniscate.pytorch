## Unsupervised Feature Learning via Non-parameteric Instance Discrimination

This repo constains the pytorch implementation for the CVPR2018 unsupervised learning paper [(arxiv)](https://arxiv.org/pdf/1805.01978.pdf).

## Updated Pretrained Model

An updated instance discrimination model with memory bank implementation and with nce-k=65536 negatives is provided.
The updated model is trained with Softmax-CE loss as in CPC/MoCo instead of the original NCE loss.

- [ResNet 50](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/lemniscate/lemniscate_resnet50_update.pth) (Linear ImageNet Acc 58.5%)


**Oldies**: original releases of ResNet18 and ResNet50 trained with 4096 negatives and the NCE loss.
Each tar ball contains the feature representation of all ImageNet training images (600 mb) and model weights (100-200mb).
You can also get these representations by forwarding the network for the entire ImageNet images.

- [ResNet 18](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/lemniscate/lemniscate_resnet18.pth) (top 1 nearest neighbor accuracy 41.0%)
- [ResNet 50](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/lemniscate/lemniscate_resnet50.pth) (top 1 nearest neighbor accuracy 46.8%)


## Highlight

- We formulate unsupervised learning from a completely different non-parametric perspective.
- Feature encodings can be as compact as 128 dimension for each image.
- Enjoys the benefit of advanced architectures and techniques from supervised learning.
- Runs seamlessly with nearest neighbor classifiers.

## Nearest Neighbor

Please follow [this link](http://zhirongw.westus2.cloudapp.azure.com/nn.html) for a list of nearest neighbors on ImageNet.
Results are visualized from our ResNet50 model, compared with raw image features and supervised features.
First column is the query image, followed by 20 retrievals ranked by the similarity.

## Usage

Our code extends the pytorch implementation of imagenet classification in [official pytorch release](https://github.com/pytorch/examples/tree/master/imagenet). 
Please refer to the official repo for details of data preparation and hardware configurations.

- supports python27 and [pytorch=0.4](http://pytorch.org)

- if you are looking for pytorch 0.3, please switch to tag v0.3

- clone this repo: `git clone https://github.com/zhirongw/lemniscate.pytorch`

- Training on ImageNet:

  `python main.py DATAPATH --arch resnet18 -j 32 --nce-k 4096 --nce-t 0.07  --lr 0.03 --nce-m 0.5 --low-dim 128 -b 256 `

  - parameter nce-k controls the number of negative samples. If nce-k sets to 0, the code also supports full softmax learning.
  - nce-t controls temperature of the distribution. 0.07-0.1 works well in practice.
  - nce-m stabilizes the learning process. A value of 0.5 works well in practice.
  - learning rate is initialized to 0.03, a bit smaller than standard supervised learning.
  - the embedding size is controlled by the parameter low-dim.

- During training, we monitor the supervised validation accuracy by K nearest neighbor with K=1, as it's faster, and gives a good estimation of the feature quality.

- Testing on ImageNet:

  `python main.py DATAPATH --arch resnet18 --resume input_model.pth.tar -e` runs testing with default K=200 neighbors.

- Training on CIFAR10:

  `python cifar.py --nce-k 0 --nce-t 0.1 --lr 0.03`


## Citation

```
@inproceedings{wu2018unsupervised,
  title={Unsupervised Feature Learning via Non-Parametric Instance Discrimination},
  author={Wu, Zhirong and Xiong, Yuanjun and Stella, X Yu and Lin, Dahua},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

## Contact

For any questions, please feel free to reach 
```
Zhirong Wu: xavibrowu@gmail.com
```
