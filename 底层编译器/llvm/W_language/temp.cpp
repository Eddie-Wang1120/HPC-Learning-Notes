#include <iostream>
#include <string>
#include <wchar.h>
 
using namespace std;






int main()
{
    std::string tmp = " ";
    std::cin>>tmp;
    printf("123%s\n", tmp.c_str());
    printf("%s\n", tmp[tmp.length()-1]);
}