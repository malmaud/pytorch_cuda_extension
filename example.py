import torch
from torch.utils.cpp_extension import load

my_extension = load("my_extension", sources=[
                    "extension.cpp", "kernel.cu"], verbose=True)
a = torch.cuda.FloatTensor(5).normal_()
b = torch.cuda.FloatTensor(5).normal_()
c = torch.cuda.FloatTensor(5).normal_()
d = my_extension.my_function(a, b, c)
# We have d ~= a*b+c
