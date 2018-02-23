#include <torch/torch.h>
#include "kernel.h"

using namespace at;


Tensor my_function(const Tensor& a,const Tensor& b, const Tensor &c)
{
    auto N = a.size(0);
    auto d = a.clone();
    call_custom_kernel(a.data<float>(), b.data<float>(), c.data<float>(), d.data<float>(), N);
    return d;
}

PYBIND11_MODULE(my_extension, m)
{
    m.def("my_function", &my_function, "Compute element-wise a*b + c");
}
