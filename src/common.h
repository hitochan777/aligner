#ifndef _COMMON_H_
#define _COMMON_H_

#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <utility>
#include <vector>
#include <string> 
#include <array>
#include <algorithm>
#include <set>
// As of OS X 10.9, it looks like C++ TR1 headers are removed from the
// search paths. Instead, we can include C++11 headers.
#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#endif

#if defined(__APPLE__) && defined(MAC_OS_X_VERSION_10_9) && \
	MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9
#include <unordered_map>
#include <functional>
#else // Assuming older OS X, Linux or similar platforms
#include <tr1/unordered_map>
#include <tr1/functional>
namespace std {
	using tr1::unordered_map;
	using tr1::hash;
} // namespace std
#endif

enum Smoothing {KN,WB,KW,NO,VB};

using namespace std;

/*****************************typedef********************************/
typedef int WordID;
typedef vector<WordID> WordVector;
typedef unordered_map<WordID, double> Word2Double;
typedef vector<Word2Double> Word2Word2Double;
typedef unordered_map<WordVector,Word2Double/*,container_hash<WordVector>*/ > WordVector2Word2Double;
typedef vector<WordVector2Word2Double> VWV2WD;
/********************************************************************/

constexpr unsigned int str2int(const char* str, int h = 0){
	return !str[h] ? 5381 : (str2int(str, h+1)*33) ^ str[h];
}
#endif

