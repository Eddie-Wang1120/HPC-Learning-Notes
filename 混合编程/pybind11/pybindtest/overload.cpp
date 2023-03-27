#include <pybind11/pybind11.h>

namespace py = pybind11;

struct Pet{
    Pet(const std::string &name, int age):name(name), age(age){}
    
    void set(int age_){age = age_;}
    void set(const std::string &name_){name = name_;}

    std::string name;
    int age;
};

PYBIND11_MODULE(heng, m){
    py::class_<Pet>(m, "Pet")
            .def(py::init<const std::string, int>())
            //disambiguate by casting them to function pointers
            .def("set", static_cast<void (Pet::*)(int)>(&Pet::set), "Set the pet's age")
            .def("set", static_cast<void (Pet::*)(const std::string &)>(&Pet::set), "Set the pet's name");
}


