# cycleGAN_with_segmentation

cycleGAN model trained together with semantic segmentation network that helps preserve latent embeddings in images from each domain.  
Images from CITYSCAPES and GTA5 dataset look very different- most GTA images have a wide range of sky yet most scenes in CITYSCAPES are collected in urban areas with trees and buildings blocking the sky.  
After doing domain transfer, upper parts of buildings in CITYSCAPES images are erased and replaced with sky, which is not I want.  
Forcing the generator network to preserve the structure of objects in scenes by explicitly training segmentation network on paired data might help prevent such cases...  

### TODO list
- [x] Add save/load model
- [x] Add segmentation network
- [ ] multithread processing
- [x] add loss plot with tensorboardX
- [ ] train in higher resolution
- [ ] hyperparameter tuning
- [ ] clean up dataset
- [ ] collect and train with coarse-labelled images

## Results up to date
GTA5 images            |   After domain transfer
:-------------------------:|:-------------------------:
![](logs/testB/030_00628.jpg)  |  ![](logs/BA/030_00628.jpg) 
![](logs/testB/044_00134.jpg)  |  ![](logs/BA/044_00134.jpg)
![](logs/testB/051_01024.jpg)  |  ![](logs/BA/051_01024.jpg)
![](logs/testB/068_02196.jpg)  |  ![](logs/BA/068_02196.jpg)


CITYSCAPES images      |   After domain transfer
:-------------------------:|:-------------------------:
![](logs/testA/bochum_000000_000885_leftImg8bit.jpg) | ![](logs/AB/bochum_000000_000885_leftImg8bit.jpg) 
![](logs/testA/munster_000062_000019_leftImg8bit.jpg) | ![](logs/AB/munster_000062_000019_leftImg8bit.jpg) 
![](logs/testA/stuttgart_000153_000019_leftImg8bit.jpg) | ![](logs/AB/stuttgart_000153_000019_leftImg8bit.jpg) 
![](logs/testA/ulm_000088_000019_leftImg8bit.jpg) | ![](logs/AB/ulm_000088_000019_leftImg8bit.jpg) 


