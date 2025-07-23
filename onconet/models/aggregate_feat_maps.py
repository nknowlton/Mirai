"""Aggregate patch-level feature maps into image-level predictions.

This module contains :class:`CustomBlock_Agg`, which attaches additional
ResNet blocks to a patch model and combines their outputs. The model can
either recompute features or operate on precomputed hidden tensors when
``args.use_precomputed_hiddens`` is set.

Key constructor arguments
-------------------------
patch_snapshot : str
    Path to a saved patch model to load.
block_layout : list of ``int``
    Number of blocks for each added ResNet stage.
use_precomputed_hiddens : bool
    If ``True`` skip the patch network and expect hidden tensors as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pdb
from onconet.models.factory import RegisterModel, load_model, get_layers
from onconet.models.resnet_base import ResNet


@RegisterModel("custom_agg")
class CustomBlock_Agg(nn.Module):
    def __init__(self, args):
        '''
            Given some a patch model, add add some FC layers and a shortcut to make whole image prediction
       '''
        super(CustomBlock_Agg, self).__init__()

        self.args = args
        if not args.use_precomputed_hiddens:
            # Load the patch-level feature extractor from a snapshot
            self.feat_extractor = load_model(args.patch_snapshot, args, False)
        agg_layers = get_layers(args.block_layout)
        # ResNet stack used to aggregate the extracted feature maps
        self._model = ResNet(agg_layers, args)


    def forward(self, x, risk_factors=None):
        '''
            param x: a batch of image tensors, in the order of:

            returns hidden: last hidden layer of model
        '''
        x = x.data
        if not self.args.use_precomputed_hiddens:
            # Extract patch-level feature maps before aggregation
            _, _, x = self.feat_extractor(x, risk_factors)
        x = x.data
        # Run the aggregation ResNet on the feature maps
        logit, hidden, x = self._model(x, risk_factors)
        return logit, hidden, x

    def cuda(self, device=None):
        # Move both the patch feature extractor and aggregation model to GPU
        self.feat_extractor = self.feat_extractor.cuda(device)
        self._model = self._model.cuda(device)
        return self
