#include <torch/torch.h>
#include "test.h"

using namespace std;
using namespace at;

Tensor jkernel(const Tensor& in)
{
    auto out = in.clone();
    auto ptr = out.data<float>();
    auto N = out.size(0);
    kernel(ptr, N);
    return out;
}

Tensor jsum_entry(const Tensor& a,const Tensor& b, const Tensor &c)
{
    auto N = a.size(0);
    auto d = a.clone();
    jsum_host(a.data<float>(), b.data<float>(), c.data<float>(), d.data<float>(), N);
    return d;
}

PYBIND11_MODULE(jonlib, m)
{
    m.def("record_thread", &jkernel, "Record the thread index");
    m.def("jsum", &jsum_entry, "Do a JSUM");
    m.attr("author")= "Jon Malmaud";
}
