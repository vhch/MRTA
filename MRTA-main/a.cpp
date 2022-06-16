#include <iostream>
#include <queue>
#include <vector>
using namespace std;

int main(void)
{
    int i=0;
    for(i=0; i<2; i++)
    {
        cout << i << endl;
        if(i==1)
        {
            break;
        }
    }
    cout << i << endl;
}