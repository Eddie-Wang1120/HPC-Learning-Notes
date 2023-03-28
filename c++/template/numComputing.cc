// Prime number computation by Erwin Unruh
template<int i> struct D { D(void*); operator int(); }; // 构造函数参数为 void* 指针
 
template<int p, int i> struct is_prime { // 判断 p 是否为素数，即 p 不能整除 2...p-1
    enum { prim = (p%i) && is_prime<(i>2?p:0), i-1>::prim };
};
template<> struct is_prime<0, 0> { enum { prim = 1 }; };
template<> struct is_prime<0, 1> { enum { prim = 1 }; };
 
template<int i> struct Prime_print {
    Prime_print<i-1> a;
    enum { prim = is_prime<i, i-1>::prim };
    // prim 为真时， prim?1:0 为 1，int 到 D<i> 转换报错；假时， 0 为 NULL 指针不报错
    void f() { D<i> d = prim?1:0; a.f(); } // 调用 a.f() 实例化 Prime_print<i-1>::f()
};
template<> struct Prime_print<2> { // 特例，递归终止
    enum { prim = 1 };
    void f() { D<2> d = prim?1:0; }
};
 
#ifndef LAST
#define LAST 10
#endif
 
int main() {
    Prime_print<LAST> a; a.f(); // 必须调用 a.f() 以实例化 Prime_print<LAST>::f()
}
