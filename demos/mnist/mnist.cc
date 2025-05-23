#define pb push_back
#define f first
#define s second
#define sz(i) (int)i.size()
#include "mnist.h"

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>
using namespace std;
/*
 _______________________________________
( If you don't fail at least 90% of the )
( time, you're not aiming high enough.  )
(                                       )
( - Alan Kay                            )
 ---------------------------------------
        o   ^__^
         o  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
*/
const int MAXN = 10000;
pair<vector<int> , vector<vector<float>>> GetMnist(std::string path = "mnist_test.csv"){ // label set , image set
    ifstream cin(path);
    string temp;
    getline(cin , temp);
    vector<vector<float>> ans;
    vector<int> labels;
    while(getline(cin , temp)){
        vector<float> currImage;
        stringstream ss(temp);
        string token;
        bool label = true;
        while(getline(ss , token , ',')){
            if(!label)
                currImage.pb(stof(token) / 255.0f);
            else
                labels.pb(stoi(token));
            label = false;
        }
        ans.pb(currImage);
    }
    cin.close();
    return {labels , ans};
}