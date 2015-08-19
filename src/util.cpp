#include "util.h"
#include "kernel.h"

double frame_acc(const vector<uchar>& path1, const vector<uchar>& path2, bool norm){

   assert(path1.size() == path2.size());
   double corr = 0;
   for(int i = 0; i < path1.size(); ++i){
      corr += (path1[i] == path2[i]) ? 1:0;
   }

   if(norm) corr /= path1.size();

   return corr;
}

// reference is path1.
double phone_acc(const vector<uchar>& path1, const vector<uchar>& path2, bool norm){
   assert(path1.size() == path2.size());

   vector<uchar> path1_trim;
   vector<uchar> path2_trim;
   trim_path(path1, path1_trim);
   trim_path(path2, path2_trim);

   vector<int32> path1_trim_32;
   vector<int32> path2_trim_32;

   UcharToInt32(path1_trim, path1_trim_32);
   UcharToInt32(path2_trim, path2_trim_32);

   int32 dist = LevenshteinEditDistance(path1_trim_32, path2_trim_32);

   double corr = path1_trim.size() - dist;

   if(corr < 0) corr = 0;

   if(norm) corr /= path1_trim.size();

   return corr;
}

void UcharToInt32(const vector<uchar>& src_path, vector<int32>& des_path){
   des_path.resize(src_path.size());
   for(int i = 0; i < src_path.size(); ++i)
      des_path[i] = src_path[i];
}

void Int32ToUchar(const vector<int32>& src_path, vector<uchar>& des_path){
   des_path.resize(src_path.size());
   for(int i = 0; i < src_path.size(); ++i){
      des_path[i] = src_path[i];
      assert(src_path[i] == des_path[i]);
   }
}

int32 best(const vector<BaseFloat> &arr){
   assert(arr.size() >= 1);
   BaseFloat max = arr[0];
   int32 index = 0;
   for(int i = 1; i < arr.size(); ++i)
     if(max < arr[i]){
        max = arr[i];
        index = i;
     }
   return index;
}

void trim_path(const vector<uchar>& scr_path, vector<uchar>& des_path){
   des_path.clear();

   int32 prev = scr_path[0];
   des_path.push_back(scr_path[0]);

   for(int i = 1; i < scr_path.size(); ++i){
      if(prev != scr_path[i]){
         prev = scr_path[i];
         des_path.push_back(scr_path[i]);
      }
   }
}

void getPhone(const string &key, const string &timit, map<string, int> &phMap, vector<uchar> &phIdx){

   // get phone path
   size_t pos = key.find('_');
   string spk = key.substr(0, pos);
   string sen = key.substr(pos+1);

   char buf[BUFSIZE];
   sprintf(buf, "ls %s/*/*/%s/%s.phn", timit.c_str(), spk.c_str(), sen.c_str());
   string path = execute(buf); 

   //analyze phone
   ifstream fin(path.substr(0, path.find('\n')).c_str());
   int s, e;
   string phn;

   vector<int> ph_e;
   vector<int> ph_idx;

   while(fin >> s >> e >> phn){
      if(phMap.find(phn) == phMap.end() || phMap[phn] < 0){
         int size = ph_e.size();
         if(size == 0)
            continue;
         else
            ph_e[size-1] = (ph_e[size-1] + e)/2;
      }else{
         ph_e.push_back(e);
         ph_idx.push_back(phMap[phn]);
      }
   }
   ph_e[ph_e.size()-1] = e;

   double step = ph_e[ph_e.size()-1] /(double) phIdx.size();
   int j = 0;
   for(int i = 0, size = phIdx.size(); i < size; ++i){
      double now = step/2 + i*step;
      while(now >= ph_e[j]) j++;
      phIdx[i] = ph_idx[j];
   }
}

// use second value as key
void readPhMap(const string path, const string id_path, map<string, int> &phMap){
   map<string, int> inner;
   string line, tmp;
   int32 id;

   {
      ifstream fin(id_path.c_str());
      while(getline(fin, line)){
         stringstream ss(line);
         ss >> tmp >> id;
         inner[tmp] = id;
         assert(id == inner[tmp]);
      }

   }

   ifstream fin(path.c_str());

   while(getline(fin, line)){
      stringstream ss(line);
      string tmp2;
      ss >> tmp >> tmp2;

      if(tmp2.empty()
            || inner.find(tmp2) == inner.end()){

         phMap[tmp] = -1;
         continue;
      }

      phMap[tmp] = inner[tmp2];
   }
}


string execute(const string &cmd){
   FILE* pipe = popen(cmd.c_str(), "r");
   assert(pipe);

   string ret;
   char buf[BUFSIZE];
   while(!feof(pipe)){
      if( fgets(buf, BUFSIZE, pipe) != NULL)
         ret.append(buf);
   }
   pclose(pipe);
   return ret;
}

void print(const MatrixBase<BaseFloat> &mat, int row){
   BaseFloat sum = 0;
   BaseFloat sqr_sum = 0;
   int counter = 0;


   int start = 0, end = mat.NumRows();
   if(row >= 0){
      start = row; end = row+1;
   }

   for(int i = start; i < end; ++i){
      for(int j = 0; j < mat.NumCols(); ++j){
         cout << mat(i, j) << " ";
         sum += mat(i, j);
         sqr_sum += mat(i,j) * mat(i, j);
         counter++;
      }
      cout << endl;
   }
   cout << "Sum: " << sum << "\t SSUM: " << sqr_sum << "\t CNT: " << counter<< endl;
}

void print(const CuMatrixBase<BaseFloat> &cumat, int row){
   Matrix<BaseFloat> tmp(cumat.NumRows(), cumat.NumCols());
   tmp.CopyFromMat(cumat);
   print(tmp, row);
}

// N = # of packs_ptr, F = dimension of feats, S = max state.
void propPsi(int N, int F, int S, int maxL, PsiPack* packs_ptr){
   Timer tim;

   // TODO compute the dummy part only once.
   cuda_prop_psi(N, F, maxL, N, F, S, packs_ptr);

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

// N = # of packs_ptr, F = dimension of feats, S = max state.
void backPsi(int N, int F, int S, int maxL, PsiPack* packs_ptr){
   Timer tim;

   // TODO compute the dummy part only once.
   cuda_back_psi(N, maxL, F*S + F, N, F, S, packs_ptr);

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

