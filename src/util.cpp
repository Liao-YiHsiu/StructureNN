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

   vector<int> path1_trim_32;
   vector<int> path2_trim_32;

   UcharToInt32(path1_trim, path1_trim_32);
   UcharToInt32(path2_trim, path2_trim_32);

   int dist = LevenshteinEditDistance(path1_trim_32, path2_trim_32);

   double corr = path1_trim.size() - dist;

   if(corr < 0) corr = 0;

   if(norm) corr /= path1_trim.size();

   return corr;
}

double phone_frame_acc(const vector<uchar> &path1, const vector<uchar> &path2, bool dummy){
   return phone_acc(path1, path2, false) + frame_acc(path1, path2, true);
}

void UcharToInt32(const vector<uchar>& src_path, vector<int>& des_path){
   des_path.resize(src_path.size());
   for(int i = 0; i < src_path.size(); ++i)
      des_path[i] = src_path[i];
}

void Int32ToUchar(const vector<int>& src_path, vector<uchar>& des_path){
   des_path.resize(src_path.size());
   for(int i = 0; i < src_path.size(); ++i){
      des_path[i] = src_path[i];
      assert(src_path[i] == des_path[i]);
   }
}

int best(const vector<BaseFloat> &arr){
   assert(arr.size() >= 1);
   BaseFloat max = arr[0];
   int index = 0;
   for(int i = 1; i < arr.size(); ++i)
     if(max < arr[i]){
        max = arr[i];
        index = i;
     }
   return index;
}

void trim_path(const vector<uchar>& scr_path, vector<uchar>& des_path){
   des_path.clear();

   int prev = scr_path[0];
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
   int id;

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

string strAfter(const string &src, const string &key){
   size_t pos = src.find(key);
   assert(pos >= 0);
   return src.substr(pos + key.length());
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

void resizeBuff(CuMatrix<BaseFloat> *mat, int rows, int cols){
   if(mat->NumRows() < rows || mat->NumCols() != cols){
      mat->Resize(rows, cols, kUndefined);
   }
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

void propRPsi(RPsiPack* pack){
   Timer tim;

   cuda_prop_rpsi(pack->T, pack->L, pack);

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void backRPsi(RPsiPack *pack){
   Timer tim;

   cuda_back_rpsi(pack->T, pack->P, pack->L, pack);

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void dist_prop(const CuMatrixBase<BaseFloat> &mat, const int* seq_arr, int seq_stride,
      const int* id_arr, float** mat_arr, int* mat_arr_stride){
   Timer tim;

   int rows = mat.NumRows();

   cuda_dist_prop((rows-1)/BLOCKSIZE+1, BLOCKSIZE, getCuPointer(&mat),
         rows, mat.NumCols(), mat.Stride(), seq_arr, seq_stride, id_arr, mat_arr, mat_arr_stride);

   CU_SAFE_CALL(cudaGetLastError()); 
   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void comb_prop(float** mat_arr, int* mat_arr_stride, const int* seq_arr, int seq_stride,
      const int* id_arr, CuMatrixBase<BaseFloat> &mat){
   Timer tim;

   int rows = mat.NumRows()/seq_stride;

   cuda_comb_prop((rows-1)/BLOCKSIZE+1, BLOCKSIZE, getCuPointer(&mat),
         rows, mat.NumCols(), mat.Stride(), seq_arr, seq_stride, id_arr, mat_arr, mat_arr_stride);

   CU_SAFE_CALL(cudaGetLastError()); 
   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void dist_back(const CuMatrixBase<BaseFloat> &mat, const int* seq_arr, int seq_stride,
      const int* id_arr, float** mat_arr, int* mat_arr_stride){
   Timer tim;

   int rows = mat.NumRows()/seq_stride;

   cuda_dist_back((rows-1)/BLOCKSIZE+1, BLOCKSIZE, getCuPointer(&mat),
         rows, mat.NumCols(), mat.Stride(), seq_arr, seq_stride, id_arr, mat_arr, mat_arr_stride);

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void comb_back(float** mat_arr, int* mat_arr_stride, const int* seq_arr, int seq_stride,
      const int* id_arr, CuMatrixBase<BaseFloat> &mat){
   Timer tim;

   int rows = mat.NumRows();

   cuda_comb_back((rows-1)/BLOCKSIZE+1, BLOCKSIZE, getCuPointer(&mat),
         rows, mat.NumCols(), mat.Stride(), seq_arr, seq_stride, id_arr, mat_arr, mat_arr_stride);

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void embed_prop(const CuMatrixBase<BaseFloat> &in, const int* seq_arr, int seq_stride, 
      CuMatrixBase<BaseFloat> &out){
   Timer tim;

   int rows = out.NumRows();

   cuda_embed_prop((rows-1)/BLOCKSIZE+1, BLOCKSIZE,
         getCuPointer(&in), in.NumRows(), in.NumCols(), in.Stride(),
         seq_arr, seq_stride, 
         getCuPointer(&out), rows, out.Stride());

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void embed_back(const CuMatrixBase<BaseFloat> &out_diff, int seq_stride, 
      CuMatrixBase<BaseFloat> &in_diff){
   Timer tim;

   int threads = in_diff.NumRows() * in_diff.NumCols();

   cuda_embed_back((threads-1)/BLOCKSIZE+1, BLOCKSIZE,
         getCuPointer(&out_diff), out_diff.NumRows(), out_diff.Stride(), seq_stride,
         getCuPointer(&in_diff), in_diff.NumRows(), in_diff.NumCols(), in_diff.Stride());

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void blendsum_prop(const CuMatrixBase<BaseFloat> &in, const int* seq_arr, int seq_size,
      CuMatrixBase<BaseFloat> &out){
   Timer tim;
   
   int threads = out.NumRows() * out.NumCols();

   cuda_blendsum_prop((threads-1)/BLOCKSIZE+1, BLOCKSIZE,
         getCuPointer(&in), in.NumRows(), in.NumCols(), in.Stride(),
         seq_arr, seq_size, getCuPointer(&out), out.NumRows(), out.Stride());

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void blendsum_back(const CuMatrixBase<BaseFloat> &out_diff, const int *seq_arr, int seq_size,
      CuMatrixBase<BaseFloat> &in_diff){
   Timer tim;
   
   int threads = out_diff.NumRows() * out_diff.NumCols();

   cuda_blendsum_back((threads-1)/BLOCKSIZE+1, BLOCKSIZE,
         getCuPointer(&out_diff), out_diff.NumRows(), out_diff.NumCols(), out_diff.Stride(),
         seq_arr, seq_size, getCuPointer(&in_diff), in_diff.NumRows(), in_diff.Stride());

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void cuMemCopy(float* dst, int dst_pitch, const float* src, int src_pitch, int width, int height){
   Timer tim;

   int threads = width * height;

   cuda_mem_copy((threads-1)/BLOCKSIZE+1, BLOCKSIZE,
         dst, dst_pitch, src, src_pitch, width, height);

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void LockSleep(string filename, int ms){
   int fd = open(filename.c_str() , O_RDWR | O_CREAT, 0666);
   assert(fd > 0);

   int rc = flock(fd, LOCK_EX | LOCK_NB);
   if( rc == 0 ){
      usleep(ms * 1000);
   } 

   flock(fd, LOCK_UN);
   close(fd);
}
