//$ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) heng.cpp -o heng$(python3.7-config --extension-suffix)

#include <pybind11/pybind11.h>
// #include "tensor.h"
// namespace py = pybind11;

// PYBIND11_MODULE(heng, m){
//     py::class_<tensor>(m, "tensor")
//         .def(py::init<const int>())
//         .def("add", &tensor::add);
// }


class Pet
{
public:
    Pet(const std::string &name):name(name){}
    void setName(const std::string &name_){name = name_;}
    const std::string &getName() const {return name;}

    std::string name;
    int age = 0;
private:
    int tag;
};

namespace py = pybind11;

PYBIND11_MODULE(heng, m){
    //py::dynamic_attr->make sure every attributes dynamic
    py::class_<Pet>(m, "Pet", py::dynamic_attr())
            .def(py::init<const std::string &>())
            .def("setName", &Pet::setName)
            .def("getName", &Pet::getName)
            //directly exposed name
            .def_readwrite("name", &Pet::name)
            //class description
            .def("__repr__",
                [](const Pet &a){
                    return "example.Pet named '"+a.name+"'>";
                });
}