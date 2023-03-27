//$ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) heng.cpp -o heng$(python3.7-config --extension-suffix)

#include <pybind11/pybind11.h>
#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(heng, m){
    py::class_<tensor>(m, "tensor", py::dynamic_attr())
        .def(py::init<const int>())
        .def_readwrite("nElem", &tensor::nElem)
        ;
}



