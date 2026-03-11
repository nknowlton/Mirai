"""Custom ResNet backbone used in the Mirai architecture.

This module exposes :class:`CustomResnet`, a thin wrapper around
``ResNet`` that allows the block configuration and pretrained weights
to be specified at runtime.

Parameters passed through ``args``
-------------------------------
block_layout: list of ``int``
    Defines the number of blocks per stage of the network.
pretrained_imagenet_model_name: str
    Name of the torchvision ResNet variant to load weights from.
pretrained_on_imagenet: bool
    If ``True`` the backbone is initialised with ImageNet weights.
"""

from torch import nn

from onconet.models.factory import RegisterModel, load_pretrained_weights, get_layers
from onconet.models.default_resnets import load_pretrained_model
from onconet.models.resnet_base import ResNet

@RegisterModel("custom_resnet")
class CustomResnet(nn.Module):
    def __init__(self, args):
        super(CustomResnet, self).__init__()
        # Build the underlying ResNet with the user-provided block layout
        layers = get_layers(args.block_layout)
        self._model = ResNet(layers, args)
        model_name = args.pretrained_imagenet_model_name
        if args.pretrained_on_imagenet:
            # Optionally initialise with ImageNet weights
            load_pretrained_weights(
                self._model, load_pretrained_model(model_name)
            )

    def forward(self, x, risk_factors=None, batch=None):
        # Delegate to the underlying ResNet. ``batch`` is unused but kept for
        # API compatibility with other models in this package.
        return self._model(x, risk_factors=risk_factors, batch=None)

    def cuda(self, device=None):
        # Ensure the wrapped model resides on the correct device
        self._model = self._model.cuda(device)
        return self
