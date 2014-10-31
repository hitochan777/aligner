#ifndef _TTABLES_H_
#define _TTABLES_H_

#include "common.h"
#include "LM.h"
#include "corpus.h"

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
	static double log_poisson(unsigned x, const double& lambda) {
		assert(lambda > 0.0);
		return log(lambda) * x - lgamma(x + 1) - lambda;
	}
};

class TTable {
	public:
		TTable() {}

		TTable(int _n);
		double prob(const WordVector& e, const WordID& f) const;
		double backoffProb(const WordVector& e, const WordID& f) const;
		void Increment(const WordVector& e, const int& f);
		void Increment(const WordVector& e, const int& f, double x);
		void NormalizeVB(const double alpha);
		void Normalize();
		void knEstimate();
		TTable& operator+=(const TTable& rhs);
		void ShowCounts(int index, Dict& d);
		void ShowCounts(Dict& d);
		void ShowTTable(int index, Dict& d);
		void ShowTTable(Dict& d);
		void copyFromKneserNeyLM(bool copyAll = false,bool copyAllProb = false);
		static WordVector makeWordVector(WordVector& trg,int index,int history, WordID kNULL);
		//void ExportToFile(const char* filename, Dict& d);

	private:
		void _ShowCounts(int index,Dict& d);
		void _ShowTTable(int index,Dict& d);
	public:
		VWV2WD ttables;
		VWV2WD counts;
		WV2D bow;//backoff weight
		LM lm;
		int n;//how many target words to consider in P(f|e_{i},e_{i+1},e_{i+n}). this corresponds to history + 1
		
};

#endif
