# Ideas

### Ground identification and removal
Run trough several images and find what is the ground without clouds. This should be updated by itself when snow changes what the ground is. This preprocessing gives a ground, so that when a new picture is added, the parts of this picture that is similar to that ground is removed.

### U-Net, Res-Net, ImageNet
What is U-Net, read up on this.

### Images at night
Several approaches:
- Let the model learn that there is no information in the image.
- Do some different preprocessing on the night images, than the other images.
- Add a feature in the network, where it is annotated whether the image is night.
- Don't use the image, or use some default output, when the mean illumination in the image is below some threshold.

