# pytorch-sandbox
Sandbox repo for pytorch sandbox


# Setting up CUDA and cuDNN on WSL Ubuntu 22.04

This is not needed if you are using PyTorch 2.0.0. This is only needed if you are using PyTorch 2.0.1.

Run the following command to set up CUDA and cuDNN and the WSL:

```commandline
make install-cuda-wsl
```

## References
* https://stackoverflow.com/questions/76327419/valueerror-libcublas-so-0-9-not-found-in-the-system-path
* https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-ubuntu
* https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#cudnn-package-manager-installation-overview
* https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202

# PyTorch 2.0.1 issues

There were issues with PyTorch 2.0.1 where not all dependencies were included in the wheel file. Hence, we had to 
downgrade to 2.0.0.

Run the following command to install PyTorch 2.0.0:

```commandline
poetry add torch@2.0.0 torchaudio@2.0.1 torchvision@0.15.1
```

## References
* https://github.com/pytorch/pytorch/issues/100974
* https://github.com/sanchit-gandhi/whisper-jax/issues/108