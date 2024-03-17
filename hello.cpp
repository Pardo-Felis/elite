#include<bits/stdc++.h> 

using namespace std;

int main( )
{
    map<string,int> m;
    queue<string> q;
    int n,ans=10000;
    cin>>n;
    while(n--)
    {
        string name;
        int score;
        cin>>name>>score;
        map<string,int>::iterator ii=m.find(name);
        if(ii==m.end()) {
            m.insert(pair<string,int>(name,score+1000));
            ans=min(ans,score+1000);
            q.push(name);
        }
        else{
            pair<string,int> p;
            p.first=ii->first;
            p.second=ii->second+score;
            ans=min(ans,p.second);
            m.erase(ii);
            m.insert(p);
        }
    }
    bool flag=true;
    while(flag)
    {
        string qw=q.front();
        q.pop();
        map<string,int>::iterator iq=m.find(qw);
        if(iq->second==ans)
        {
            flag=false;
            cout<<iq->first<<endl;
            cout<<iq->second<<endl;
        }
    }
    return 0;
}
