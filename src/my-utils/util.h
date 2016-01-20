#ifndef _MY_UTIL_H_
#define _MY_UTIL_H_
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-matrix.h"
#include "util/edit-distance.h"
#include "my-utils/type.h"

#include <string>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>

#define BUFSIZE 4096
#define GPU_FILE "/tmp/gpu_lock"

using namespace kaldi;
using namespace std;

inline double sigmoid(double x){ return 1/(1+exp(-x)); }
inline double softplus(double x){ return x > 0 ? x + softplus(-x) : log(1+exp(x)); }
inline double log_add(double a, double b){ return a + softplus(b-a);}

double frame_acc(const vector<uchar>& path1, const vector<uchar>& path2, bool norm = true);
double phone_acc(const vector<uchar>& path1, const vector<uchar>& path2, bool norm = true);
double phone_frame_acc(const vector<uchar>& path1, const vector<uchar>& path2, bool norm = true);

int best(const vector<BaseFloat> &arr);

void trim_path(const vector<uchar>& scr_path, vector<uchar>& des_path);
void UcharToInt32(const vector<uchar>& src_path, vector<int>& des_path);
void Int32ToUchar(const vector<int>& src_path, vector<uchar>& des_path);

void getPhone(const string &key, const string &timit, map<string, int> &phMap, vector<uchar> &phIdx);
void readPhMap(const string path, const string id_path, map<string, int> &phMap);
string execute(const string &cmd);

string strAfter(const string &src, const string &key);

void print(const MatrixBase<BaseFloat> &mat, int row = -1);
void print(const CuMatrixBase<BaseFloat> &mat, int row = -1);

void resizeBuff(CuMatrix<BaseFloat> *mat, int rows, int cols);

void LockSleep(string filename, int ms = 2000);

template<typename T>
void VecToVecRef(vector<T>& src, vector<T*> &dest);

#endif
