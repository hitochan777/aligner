#ifndef _TTABLES_H_
#define _TTABLES_H_

#include "src/common.h"
#include "src/LM.h"

struct Md {
	static double digamma(double x) {
		double result = 0, xx, xx2, xx4;
		for ( ; x < 7; ++x){
			result -= 1/x;
		}
		x -= 1.0/2.0;
		xx = 1.0/x;
		xx2 = xx*xx;
		xx4 = xx2*xx2;
		result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
		return result;
	}
};

class TTable {
	public:
		TTable() {}

		TTable(int _n);
		inline double prob(const WordVector& e, const WordID& f) const;
		inline double backoffProb(const WordVector& e, const WordID& f);
		inline void Increment(const WordVector& e, const int& f);
		inline void Increment(const WordVector& e, const int& f, double x);
		void NormalizeVB(const double alpha);
		void Normalize();
		void knEstimate();
		TTable& operator+=(const TTable& rhs);
		void ShowCounts(int index);
		void ShowCounts();
		void ShowTTable(int index);
		void ShowTTable();
		static WordVector makeWordVector(WordVector& trg,int index,int history, WordID kNULL);
		static enum Smoothing getSmoothMethod(std::string str);
		void ExportToFile(const char* filename, Dict& d);

	private:
		void _ShowCounts(int index);
		void _ShowTTable(int index);
	public:
		VWV2WD ttables;
		VWV2WD counts;
		LM lm;
		int n;//how many target words to consider in P(f|e_{i},e_{i+1},e_{i+n})
};

#endif
