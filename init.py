import torch
from torch.utils.cpp_extension import load

jonlib=load("jonlib", sources=["test.cpp", "kernel.cu"], verbose=True)
x=torch.Tensor([1,2,3]).cuda()
a=torch.cuda.FloatTensor(5).normal_()
b=torch.cuda.FloatTensor(5).normal_()
c=torch.cuda.FloatTensor(5).normal_()
