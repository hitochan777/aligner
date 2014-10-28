#include "ttables.h"

using namespace std;

TTable::TTable(int _n){
	this->n = _n;
	ttables.resize(_n+1);
	counts.resize(_n+1);
}

double TTable::prob(const WordVector& e, const WordID& f) const {
	WordVector2Word2Double::const_iterator it = ttables[n].find(e);
	if(it!=ttables[n].end()){// if target n-gram exists
		const Word2Double& w2d = it->second;
		const Word2Double::const_iterator it = w2d.find(f);
		if(it==w2d.end()){// if source word does not exist
			return 1e-9;//to avoid zero frequency problem
		}
		return it->second;
	}
	else{// if target n-gram does NOT exists 
		return 1e-9;//to avoid zero frequency problem
	}
}

double TTable::backoffProb(const WordVector& e, const WordID& f){
	for(int i = n;i>=0;--i){
		WordVector2Word2Double::const_iterator it = ttables[i].find(WordVector(e.begin(),e.begin()+i));
		if(it!=ttables[i].end()){// if target i-gram exists
			const Word2Double& w2d = it->second;
			const Word2Double::const_iterator it = w2d.find(f);
			if(it==w2d.end()){// if source word does not exist
				continue;
			}
			return it->second;
		}
	}
	return 1e-9;
}

void TTable::Increment(const WordVector& e, const int& f) {
	counts[n][e][f] += 1.0;
}

void TTable::Increment(const WordVector& e, const int& f, double x) {
	counts[n][e][f] += x;
}

void TTable::NormalizeVB(const double alpha) {
	ttables.swap(counts);
	for (WordVector2Word2Double::iterator it = ttables[n].begin(); it != ttables[n].end(); ++it) {
		double tot = 0;
		Word2Double& cpd = it->second;
		for (Word2Double::iterator it2 = cpd.begin(); it2 != cpd.end(); ++it2){
			tot += it2->second + alpha;
		}
		if (!tot){
			tot = 1;
		}
		for (Word2Double::iterator it2 = cpd.begin(); it2 != cpd.end(); ++it2){
			it2->second = exp(Md::digamma(it2->second + alpha) - Md::digamma(tot));
		}
	}
	counts.clear();
}

void TTable::Normalize() {
	ttables.swap(counts);
	for (WordVector2Word2Double::iterator it = ttables[n].begin(); it != ttables[n].end(); ++it) {
		double tot = 0;
		Word2Double& cpd = it->second;	
		for (Word2Double::iterator it2 = cpd.begin(); it2 != cpd.end(); ++it2){
			for(int i = 0;i <= n-1; ++i ){//calculate counts for 1~(n-1) grams from n-gram counts
				ttables[i][WordVector((it->first).begin(),(it->first).begin()+i)][it2->first]
					+= ttables[n][it->first][it2->first];
			}
			tot += it2->second;
		}
		if (!tot){
			tot = 1;
		}
		for (Word2Double::iterator it2 = cpd.begin(); it2 != cpd.end(); ++it2){
			it2->second /= tot;
		}
	}
	for(int i = 0;i <= n - 1; ++i){//normalize probabilities for 1~ (n-1) grams 
		for (WordVector2Word2Double::iterator it = ttables[i].begin(); it != ttables[i].end(); ++it) {
			double tot = 0;
			Word2Double& cpd = it->second;
			for (Word2Double::iterator it2 = cpd.begin(); it2 != cpd.end(); ++it2){
				tot += it2->second;
			}
			if (!tot){
				tot = 1;
			}
			for (Word2Double::iterator it2 = cpd.begin(); it2 != cpd.end(); ++it2){
				it2->second /= tot;
			}
		}
	}
	counts.clear();
	counts.resize(n+1);
}
void TTable::knEstimate(){

}

// adds counts from another TTable - probabilities remain unchanged
TTable& TTable::operator+=(const TTable& rhs) {
	if(rhs.n != n){
		std::cerr<<"Two tables have different n-gram number."<<std::endl;
	}
	for(int i = 0;i<std::min(rhs.n,n); ++i){
		for (WordVector2Word2Double::const_iterator it = rhs.counts[i].begin(); it != rhs.counts[i].end(); ++it) {
			const Word2Double& cpd = it->second;
			Word2Double& tgt = counts[i][it->first];
			for (auto p : cpd){
				tgt[p.first] += p.second;
			}
		}
	}
	return *this;
}

void TTable::ShowCounts(int index) {
	_ShowCounts(index);
}

void TTable::ShowCounts() {
	_ShowCounts(n);
}

void TTable::ShowTTable(int index) {
	_ShowTTable(index);
}

void TTable::ShowTTable(){
	_ShowTTable(n);
}

WordVector TTable::makeWordVector(WordVector& trg,int index,int history,WordID kNULL){//index start from 0
	int trglen = trg.size();
	if(history<0){
		throw std::invalid_argument("history must be non-negative integer.");
	}	
	if( index < 0 || index >= trglen ){
		throw std::invalid_argument("index is out of range in makeWordVector.");	
	}
	if( index - history < 0 ){
		WordVector wv;
		for(int i = 0; i < history - index; ++i){
			wv.push_back(kNULL);
		}
		wv.insert(wv.end(),&trg[0],&trg[index]+1);
		return wv;
	}
	return WordVector(&trg[index]-history,&trg[index]+1);
}	

enum Smoothing TTable::getSmoothMethod(string str){
	switch (str2int(str)){
		case str2int("normal"):
			return Normal;
			break;
		case str2int("vb"):
			return VB;
			break;
		case str2int("kn"):
			return KN;
			break;
		default:
			return Normal;
			break;
	}
	return Normal;
}


void TTable::_ShowCounts(int index) {
	for (WordVector2Word2Double::const_iterator it = counts[index-1].begin(); it != counts[index-1].end(); ++it) {
		const Word2Double& cpd = it->second;
		for (auto& p : cpd) {
			std::cerr << "c(" << TD::Convert(p.first) << '|' << TD::Convert(it->first) << ") = " << p.second << std::endl;
		}
	}
}

void TTable::_ShowTTable(int index){
	std::fprintf(stderr,"showing %d-gram prob table\n",index);
	std::fprintf(stderr,"size: %u\n",ttables[index-1].size());
	std::fprintf(stderr,"skipping cell with zero prob\n");
	for (WordVector2Word2Double::const_iterator it = ttables[index-1].begin(); it != ttables[index-1].end(); ++it) {
		const Word2Double& cpd = it->second;
		for (auto& p : cpd) {
			if(p.second==0) continue;//do not print prob with 0
			std::fprintf(stderr,"Pr(%s|%s) = %lf\n", TD::Convert(p.first).c_str(), TD::Convert(it->first).c_str(),p.second);
		}
	}

}

void TTable::ExportToFile(const char* filename, Dict& d) {
	std::ofstream file(filename);
	for (unsigned i = 0; i < ttable.size(); ++i) {
		const std::string& a = d.Convert(i);
		Word2Double& cpd = ttable[i];
		for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it) {
			const std::string& b = d.Convert(it->first);
			double c = log(it->second);
			file << a << '\t' << b << '\t' << c << std::endl;
		}
	}
	file.close();
}


