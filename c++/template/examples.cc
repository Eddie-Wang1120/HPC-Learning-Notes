#include <iostream>

using namespace std;

template <
    typename T>
bool equivalent(const T &a, const T &b){
    return !(a < b) && !(b < a);
}

template <
    typename T=int>
class bignumber {
 public:
    T _v;
    bignumber(T a) : _v(a){}
    inline bool operator<(const bignumber &b) const;
};

template <
    typename T>
bool bignumber<T>::operator<(const bignumber &b) const{
    return this->_v < b._v;
}

template <
    typename T>
class aTMP {
 public:
    void f1() {cout << "f1\n";}
    void f2() {cout << "f2\n";}
};

template <
    typename T,
    int i=1>
class numComputing {
 public:
    enum { retValue = i + numComputing<T, i-1>::retValue};
    static void f(){cout << "numComputing:" << i << endl;}
};

template <
    typename T>
class numComputing<T, 0> {
 public:
    enum { retValue = 0};
};

template <
    typename T>
class typeComputing {
 public:
    typedef volatile T* retType;
};

template <
    typename T>
class codeComputing {
 public:
    static void f() {T::f();}
};

int main(){
    bignumber<int> a(1), b(1);
    cout << equivalent(a, b) << endl;
    cout << equivalent<double>(1, 2.2) << endl;
    aTMP<int> atmp;
    atmp.f1();
    atmp.f2();

    typeComputing<int>::retType tpe=0;
    cout << "type size:" << sizeof(tpe) << endl;

    cout << "numcompute:" << numComputing<int, 500>::retValue << endl;

    codeComputing<numComputing<int, 99>>::f();

    return 0;
}