#include <cstdio>
#include <cstdint>
#include <string>
#include <tuple>

using namespace std;

tuple<uint64_t, string> func(uint32_t cnt){
    uint64_t val_a = 0;
    uint64_t val_b = 1;
    string text;
    char buff[24];
    text.reserve(cnt*2);
    text += "1";
    for(uint32_t i=0;i<cnt;i++){
        tie(val_a, val_b) = make_tuple(val_b, val_a+val_b);
        sprintf(buff, ",%lu", val_b);
        text += buff;
    }
    return {val_b, text};
}

int main(){
    uint32_t cnt = 10;
    uint64_t fib;
    string fib_str;
    tie(fib, fib_str) = func(cnt);
    printf("fib sum(%u): %lu\n", cnt, fib);
    printf("info: %s\n", fib_str.c_str());
    return 0;
}