# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 19:56:27 2022

@author: mobas
"""

import numpy as np
import torch
import torch.nn as nn

# build two model
widenet = nn.Sequential(
    nn.Linear(2, 4),
    nn.Linear(4, 3),
    )

deepnet = nn.Sequential(
    nn.Linear(2, 2),
    nn.Linear(2,2),
    nn.Linear(2,3),
    )
print(widenet)
print(" ")
print(deepnet)
print("Named Parameter")
for p in widenet.named_parameters():
    print(p) 
    
# Count number of parameters
numNodeInWide = 0
for p in widenet.named_parameters():
    if 'bias' in p[0]:
        numNodeInWide+=len(p[1])
        
numNodeInDeep = 0
for paramName, paramVect in deepnet.named_parameters():
    if "bias" in paramName:
        numNodeInDeep+=len(paramVect)
        
print("Node in wide network ", numNodeInWide)
print("Node in deep network ", numNodeInDeep)


# =============================================================================
# Just parameter
# =============================================================================
for p in widenet.parameters():
    print(p)
    
# =============================================================================
# Trainable parameter
# =============================================================================
trainP = 0
for p in widenet.parameters():
    if p.requires_grad:
        print("This p is %s parameter" %p.numel())
        trainP+=p.numel()
print("Total train para: ",trainP)
# =============================================================================
# Library Nice function to get model summary
# =============================================================================
from torchsummary import summary
summary(widenet,(1,2))

# When Python is reporting sizes, it uses "-1" to indicate "fill in whatever is appropriate here." So it's not any specific number, but instead depends on your data (in this case the batch size).
