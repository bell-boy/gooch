#include <bits/stdc++.h>
#define pb push_back
#define f first
#define s second
#define sz(i) (int)i.size()
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
pair<vector<int> , vector<vector<float>>> getMnist(){ // label set , image set
    cin.tie(0)->sync_with_stdio(0);
    ifstream cin("mnist_test.csv");
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
                currImage.pb(stof(token));
            else
                labels.pb(stoi(token));
            label = false;
        }
        ans.pb(currImage);
    }
    cin.close();
    return {labels , ans};
}

int main(){
    auto item = getMnist();
    cout << sz(item.s) << "\n";
}