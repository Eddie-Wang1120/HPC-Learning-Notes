#include <string>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct Pet{
    Pet(const std::string &name):name(name){}
    std::string name;
};

struct Dog:Pet{
    Dog(const std::string &name):Pet(name){}
    std::string bark() const {return "woof!";}
};

PYBIND11_MODULE(heng, m){
    py::class_<Pet>(m, "Pet")
            .def(py::init<const std::string &>())
            .def_readwrite("name", &Pet::name);

    py::class_<Dog, Pet>(m, "Dog")
            .def(py::init<const std::string &>())
            .def("bark", &Dog::bark);
    m.def("pet_store", [](){return std::unique_ptr<Pet>(new Dog("Molly"));});
}

