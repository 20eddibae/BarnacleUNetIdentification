#Barnacle Segmentation

Two approaches were taken on this problem, one one using OpenCV tools for a more manual, direct segmentation approach while the U-Net segmentation used existing deep learning architectures and adjusted the data creation for training the model. 

### General project structure

```
BarnacleUNetIdentification/
├── opencv-barnacle/
├── Pytorch-UNet/
```


Each folder has the respective approaches with root README files that describe the approaches taken and more detailed project documentation. 


The source of the UNet architecture was - https://github.com/milesial/Pytorch-UNet, to which I created metadata creation scripts and trained the model on the barnacle data