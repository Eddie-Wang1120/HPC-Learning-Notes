#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>

tensor::tensor(int Elem){
    this->nElem = Elem;
    this->data = (float *)malloc(nElem*sizeof(float));
    for(int i=0;i<Elem;i++){
        this->data[i] = 1;
    }
}

tensor* tensor::add(tensor* a){
    int n = a->nElem;
    tensor* res = new tensor(n);
    for(int i=0;i<n;i++){
        res->data[i] = a->data[i] + this->data[i];
    }
    return res;
}

void tensor::show(){
    for(int i=0;i<this->nElem;i++){
        printf("%f ", this->data[i]);
    }
    printf("\n");
}

// int main(){
//     tensor* a = new tensor(3);
//     tensor* b = new tensor(3);
//     tensor* c = a->add(b);
//     c->show();

// }

// Pet::Pet(const std::string &name_){
//     name = name_;
// }

// void Pet::setName(const std::string &name_){
//     name = name_;
// }

// const std::string &Pet::getName() const{
//     return name;
// }
