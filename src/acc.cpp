#include <fstream>
#include <iostream>
#include <stdlib.h>

using namespace std;

void Usage(const char* progName){
   cerr << "Usage: " << progName << " file1 file2" << endl;
   exit(-1);
}

int main(int argc, char* argv[]){
   if(argc != 3)
      Usage(argv[0]);

   ifstream fin1(argv[1]);
   ifstream fin2(argv[2]);

   string line1, line2;
   int counter = 0;
   int corr = 0;
   while(getline(fin1, line1) && getline(fin2, line2)){
      counter ++;
      if(line1.compare(line2) == 0)
         corr++;
   }

   cout << corr /(double)counter << endl;

   return 0;
}
