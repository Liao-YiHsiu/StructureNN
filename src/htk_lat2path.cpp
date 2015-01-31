#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define BUFSIZE 1024

using namespace std;
typedef struct{
   double t;
   string W;
} Node;

typedef struct{
   int E;
   double a;
   double l;
   string d;
} Edge;

void genPath(vector<Node> &nodes, 
      vector< vector<Edge> > &edges, int N);

bool readLattice(char* fLattice, vector<Node> &nodes,
      vector< vector<Edge> > &edges);

void Usage(char* progName){
   cerr << "Usage: " << progName << " lattice_file N" << endl;
}

int main(int argc, char* argv[]){
   if(argc != 3)
      Usage(argv[0]);

   srand(time(NULL));
   int N = atoi(argv[2]);

   vector<Node> nodes;
   vector< vector<Edge> > edges;

   if(!readLattice(argv[1], nodes, edges))
      Usage(argv[0]);
   genPath(nodes, edges, N);

   return 0;
}

bool readLattice(char* fLattice, vector<Node> &nodes,
      vector< vector<Edge> > &edges){

   ifstream fin(fLattice);
   string line;
   int N, L;

   // read header
   while(getline(fin, line)){
      if(strncmp(line.c_str(), "N=", 2) == 0){
         sscanf(line.c_str(), "N=%d L=%d", &N, &L);
         break;
      }
   }

   nodes.resize(N);
   edges.clear();
   edges.resize(N);

   int I; double t; char W[BUFSIZE];
   // read nodes
   for(int i = 0; i < N; ++i){
      assert(getline(fin, line));
      sscanf(line.c_str(),
            "I=%d t=%lf W=%s", &I, &t, W);

      nodes[I].t = t;
      nodes[I].W = W;
   }

   Edge tmp_edge;
   // read edges
   int J, S, E; double a, l; char d[BUFSIZE]; 
   for(int i = 0; i < L; ++i){
      assert(getline(fin, line));
      sscanf(line.c_str(),
            "J=%d S=%d E=%d a=%lf l=%lf d=%s",
            &J, &S, &E, &a, &l, d);
      tmp_edge.E = E;
      tmp_edge.a = a;
      tmp_edge.l = l;
      tmp_edge.d = d;

      edges[S].push_back(tmp_edge);
   }

}

void genPath(vector<Node> &nodes, 
      vector< vector<Edge> > &edges, int N){

   int end = nodes.size() - 1;
   int next, now;
   for(int i = 0; i < N; ++i){
      int now = 0;
      while(now != end){
         next = rand()%edges[now].size();
         cout << edges[now][next].d << "->";
         now = edges[now][next].E;
      }
      cout << "end" << endl;
   }
   
}
