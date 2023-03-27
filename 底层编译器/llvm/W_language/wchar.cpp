#include <wchar.h>
#include <iostream>

wchar_t * MBCS2Unicode(wchar_t * buff, const char * str)

{
    wchar_t * wp = buff;
    char * p = (char *)str;
    while(*p)
{
        if(*p & 0x80)
{
            *wp = *(wchar_t *)p;
            p++;
        }
        else{
            *wp = (wchar_t) *p;
        }
        wp++;
        p++;
    }
    *wp = 0x0000;
    return buff;
}

 

char * Unicode2MBCS(char * buff, const wchar_t * str)
{

    wchar_t * wp = (wchar_t *)str;
    char * p = buff, * tmp;
    while(*wp){
        tmp = (char *)wp;
        if(*wp & 0xFF00){
            *p = *tmp;
            p++;tmp++;
            *p = *tmp;
            p++;
        }
        else{
            *p = *tmp;
            p++;
        }
        wp++;

    }

    *p = 0x00;

    return buff;

}

 

std::wstring str2wstr(std::string str)
{
    size_t len = str.size();
    wchar_t * b = (wchar_t *)malloc((len+1)*sizeof(wchar_t));
    MBCS2Unicode(b,str.c_str());
    std::wstring r(b);
    free(b);
    return r;
}

std::string wstr2str(std::wstring wstr)
{
    size_t len = wstr.size();
    char * b = (char *)malloc((2*len+1)*sizeof(char));
    Unicode2MBCS(b,wstr.c_str());
    std::string r(b);
    free(b);
    return r;
}

char * wwputs(const wchar_t * wstr)
{
    int len = wcslen(wstr);
    std::cout<<len<<std::endl;
    char * buff = (char *)malloc((len * 2 + 1)*sizeof(char));
    Unicode2MBCS(buff,wstr);
    return buff;
} 

char* wputs(std::wstring wstr)
{
    char* buff = wwputs(wstr.c_str());
    return buff;
}

 


int main(){
    std::string test = "我的世界";

    // printf("%s\n",test.c_str());
    // std::wstring wtest = str2wstr(test);

    // const wchar_t* tmp = wtest.c_str();

    // const char* temp = wputs(wtest);
    // std::string tstr = temp;
    // std::cout<<tstr.length()<<std::endl;


    // std::cout<<test<<std::endl;
    // std::wcout<<wtest<<std::endl;
    // std::cout<<test.length()<<std::endl;
    // std::wcout<<wtest.length()<<std::endl;
}