#include <getopt.h>
#include "common.h"
#include "ttables.h"
#include "corpus.h"
#include "da.h"
#include "LM.h"

using namespace std;

struct PairHash {
	size_t operator()(const pair<short,short>& x) const {
		return (unsigned short)x.first << 16 | (unsigned)x.second;
	}
};

Dict d;// this variable is going to be used throughout this program

string conditional_probability_filename = "";
int is_reverse = 0;
int ITERATIONS = 5;
int favor_diagonal = 0;
double prob_align_null = 0.08;
double diagonal_tension = 4.0;
int optimize_tension = 0;
int variational_bayes = 0;
double alpha = 0.01;
int no_null_word = 0;
int train_line = -1;
bool DO_TEST;
bool ONE_TEST_FILE;
bool ONE_INPUT_FILE;
int HISTORY = 0;
string testset,ftestset,etestset;
string fname,ffname,efname;
double likelihood = 0;
double denom = 0.0;
int lc = 0;
bool flag = false;
double c0 = 0;
double emp_feat = 0;
double toks = 0;
double base2_likelihood;
enum Smoothing smooth = NO;
unordered_map<pair<short, short>, unsigned, PairHash> size_counts;

void ParseLine(const string& line,vector<unsigned>* src,vector<unsigned>* trg){
	static const unsigned kDIV = d.Convert("|||");
	static vector<unsigned> tmp;
	src->clear();
	trg->clear();
	d.ConvertWhitespaceDelimitedLine(line, &tmp);
	unsigned i = 0;
	while(i < tmp.size() && tmp[i] != kDIV) {
		src->push_back(tmp[i]);
		++i;
	}
	if (i < tmp.size() && tmp[i] == kDIV) {
		++i;
		for (; i < tmp.size() ; ++i){
			trg->push_back(tmp[i]);
		}
	}
}

void ParseLineFromSeparateFiles(const string& fline,const string& eline,vector<unsigned>* src,vector<unsigned>* trg){
	static vector<unsigned> tmp;
	src->clear();
	trg->clear();
	d.ConvertWhitespaceDelimitedLine(fline, &tmp);
	for(unsigned i = 0;i < tmp.size();++i) {
		src->push_back(tmp[i]);
	}
	tmp.clear();
	d.ConvertWhitespaceDelimitedLine(eline, &tmp);
	for(unsigned i = 0;i < tmp.size();++i) {
		trg->push_back(tmp[i]);
	}
	return ;
}

void printProcess(FILE* fp){
	fprintf(fp,"  log_e likelihood: %lf\n",likelihood);
	fprintf(fp,"  log_2 likelihood: %lf\n",base2_likelihood);
	fprintf(fp,"     cross entropy: %lf\n",(-base2_likelihood / denom));
	fprintf(fp,"        perplexity: %lf\n",pow(2.0, -base2_likelihood / denom));
	fprintf(fp,"      posterior p0: %lf\n",c0 / toks);
	fprintf(fp," posterior al-feat: %lf\n",emp_feat);
	//cerr << "     model tension: " << mod_feat / toks << endl;
	fprintf(fp,"       size counts: %d\n",size_counts.size());
	return ;
}

struct option options[] = {
	{"input",required_argument, 0,'i'},
	{"finput",required_argument, 0,0},
	{"einput",required_argument,0,0},
	{"train_line",required_argument,0,'l'},
	{"testset",required_argument,0,0},
	{"ftestset",required_argument,0,0},
	{"etestset",required_argument,0,0},
	{"test_align_output_file",required_argument,0,0},
	{"train_align_output_file",required_argument,0,0},
	{"train_process_output_file",required_argument,0,0},
	{"test_output_file",required_argument,0,0},
	{"history",required_argument,0,'h'},
	{"smoothing",required_argument,0,0},
	{"reverse",no_argument,&is_reverse,1},
	{"iterations",required_argument, 0,'I'},
	{"favor_diagonal",no_argument,&favor_diagonal,0},
	{"p0",required_argument, 0,'p'},
	{"diagonal_tension",required_argument, 0,'T'},
	{"optimize_tension",no_argument,&optimize_tension,1},
	{"variational_bayes",no_argument,&variational_bayes,1},
	{"alpha",required_argument, 0,'a'},
	{"no_null_word",no_argument,&no_null_word,1},
	{"conditional_probabilities", required_argument, 0,'c'},
	{0,0,0,0}
};

bool InitCommandLine(int argc, char** argv){
	while (1) {
		int oi;
		int c = getopt_long(argc, argv, "i:rI:dp:T:ova:Nc:l:h:", options, &oi);
		if (c == -1){
			break;
		}
		switch(c) {
			case 0:{
				       string opt = options[oi].name;
				       switch(str2int(opt.c_str())){
					       case str2int("finput"):
						       ffname = optarg;
						       break;
					       case str2int("einput"):
						       efname = optarg;
						       break;
					       case str2int("testset"):
						       testset = optarg;
						       break;
					       case str2int("ftestset"):	
						       ftestset = optarg;
						       break;
					       case str2int("etestset"):
						       etestset = optarg;
						       break;
					       case str2int("test_align_output_file"):
						       break;
					       case str2int("train_align_output_file"):
						       break;
					       case str2int("train_process_output_file"):
						       break;
					       case str2int("test_output_file"):
						       break;
					       default:
						       //some error message here
						       break;
				       }
				       break;
			       }
			case 'i': 
			       fname = optarg;
			       break;
			case 'r':
			       is_reverse = 1;
			       break;
			case 'I':
			       ITERATIONS = atoi(optarg);
			       break;
			case 'd':
			       favor_diagonal = 1;
			       break;
			case 'p':
			       prob_align_null = atof(optarg);
			       break;
			case 'T':
			       diagonal_tension = atof(optarg);
			       break;
			case 'o':
			       optimize_tension = 1;
			       break;
			case 'v':
			       variational_bayes = 1;
			       break;
			case 'a':
			       alpha = atof(optarg);
			       break;
			case 'N':
			       no_null_word = 1;
			       break;
			case 'c':
			       conditional_probability_filename = optarg;
			       break;
			case 'l':
			      	train_line = atoi(optarg); 
			       break;
			default:
			       return false;
		}
	}
	return true;
}

int main(int argc, char** argv) {
	if (!InitCommandLine(argc, argv)) {
		cerr << "Usage: " << argv[0] << " -i file.fr-en\n"
			<< " Standard options ([USE] = strongly recommended):\n"
			<< "  -i: [REQ] Input parallel corpus\n"
			<< "  -v: [USE] Use Dirichlet prior on lexical translation distributions\n"
			<< "  -d: [USE] Favor alignment points close to the monotonic diagonoal\n"
			<< "  -o: [USE] Optimize how close to the diagonal alignment points should be\n"
			<< "  -r: Run alignment in reverse (condition on target and predict source)\n"
			<< "  -c: Output conditional probability table\n"
			<< " Advanced options:\n"
			<< "  -I: number of iterations in EM training (default = 5)\n"
			<< "  -p: p_null parameter (default = 0.08)\n"
			<< "  -N: No null word\n"
			<< "  -a: alpha parameter for optional Dirichlet prior (default = 0.01)\n"
			<< "  -T: starting lambda for diagonal distance parameter (default = 4)\n";
		return 1;
	}
	bool use_null = !no_null_word;
	if (variational_bayes && alpha <= 0.0) {
		cerr << "--alpha must be > 0\n";
		return 1;
	}
	double prob_align_not_null = 1.0 - prob_align_null;
	const unsigned kNULL = d.Convert("<eps>");
	TTable t2s;
	double tot_len_ratio = 0;
	double mean_srclen_multiplier = 0;
	vector<double> probs;
	FILE *tof = stderr;
	FILE *taof = stdout;	
	for (int iter = 0; iter < ITERATIONS; ++iter) {
		const bool final_iteration = (iter == (ITERATIONS - 1));
		ifstream in,fin,ein;
		if(ONE_INPUT_FILE){
			in.open(fname.c_str(), ifstream::in);
			if (!in) {
				cerr << "Can't read " << fname << endl;
				return 1;
			}
		}
		else{
			fin.open(ffname.c_str(), ifstream::in);
			ein.open(efname.c_str(), ifstream::in);
			if(!fin){
				cerr << "Can't read " << ffname << endl;
				return 1;
			}
			if(!ein){
				cerr << "Can't read " << efname << endl;
				return 1;
			}
		}
		cerr << "ITERATION " << (iter + 1) << (final_iteration ? " (FINAL)" : "") << endl;
		likelihood = 0;
		denom = 0.0;
		lc = 0;
		flag = false;
		string line,fline,eline;
		string ssrc, strg;
		vector<unsigned> src, trg;
		c0 = 0;
		emp_feat = 0;
		toks = 0;
		while(true) {
			if(/*train_line != -1 &&*/ lc >= train_line){
				break;
			}
			if(ONE_INPUT_FILE){
				getline(in,line);	
			}
			else{
				getline(fin,fline);
				getline(ein,eline);
			}
			if (ONE_INPUT_FILE && !in){
				break;
			}
			else if(!ONE_INPUT_FILE && (!fin || !ein)){
				break;
			}
			++lc;
			if (lc % 1000 == 0) { 
				fprintf(tof,".");
				fprintf(stderr,".");
				flag = true;
			}
			if (lc %50000 == 0) { 
				fprintf(tof," [%d]\n",lc);
				fprintf(stderr," [%d]\n",lc);
				flag = false;
			}
			src.clear();
			trg.clear();
			if(ONE_INPUT_FILE){
				ParseLine(line, &src, &trg);
			}
			else{
				ParseLineFromSeparateFiles(fline,eline,&src,&trg);
			}
			if (is_reverse){
				swap(src, trg);
			}
			if (src.size() == 0 || trg.size() == 0) {
				fprintf(stderr,"Error: %d\n%s\n",lc,line.c_str());
				return 1;
			}
			if (iter == 0){
				tot_len_ratio += static_cast<double>(src.size()) / static_cast<double>(trg.size());
			}
			denom += src.size();
			probs.resize(trg.size() + 1);
			if (iter == 0){
				++size_counts[make_pair<short,short>(src.size(), trg.size())];
			}
			bool first_al = true;  // used when printing alignments
			toks += src.size();
			for (unsigned j = 0; j < src.size(); ++j) {
				const unsigned& f_j = src[j];
				double sum = 0;
				double prob_a_i = 1.0 / (trg.size() + use_null);  // uniform (model 1)
				if (use_null) {
					WordVector wv;
					wv.push_back(kNULL);
					if (favor_diagonal){
						prob_a_i = prob_align_null;
					}
					probs[0] = t2s.prob(wv, f_j) * prob_a_i;
					sum += probs[0];
				}
				double az = 0;
				if (favor_diagonal){
					az = DiagonalAlignment::ComputeZ(j+1, src.size(), trg.size(), diagonal_tension) / prob_align_not_null;
				}
				for (unsigned i = 1; i <= trg.size(); ++i) {
					if (favor_diagonal){
						prob_a_i = DiagonalAlignment::UnnormalizedProb(j + 1, i, src.size(), trg.size(), diagonal_tension) / az;
					}
					probs[i] = t2s.prob(TTable::makeWordVector(trg,i-1,HISTORY,kNULL), f_j) * prob_a_i;
					sum += probs[i];
				}
				if (final_iteration) {
					double max_p = -1;
					int max_index = -1;
					if (use_null) {
						max_index = 0;
						max_p = probs[0];
					}
					for (unsigned i = 1; i <= trg.size(); ++i) {
						if (probs[i] > max_p) {
							max_index = i;
							max_p = probs[i];
						}
					}
					if (max_index > 0) {
						if (first_al){
							first_al = false;
						}
						else{
							cout << ' ';
						}
						if (is_reverse){
							cout << j << '-' << (max_index - 1);
						}
						else{
							cout << (max_index - 1) << '-' << j;
						}
					}
				} 
				else{
					if (use_null) {
						double count = probs[0] / sum;
						WordVector wv;
						wv.push_back(kNULL);
						c0 += count;
						t2s.Increment(wv, f_j, count);
					}
					for (unsigned i = 1; i <= trg.size(); ++i) {
						const double p = probs[i] / sum;
						if(smooth == KN){ 
							WordVector vec = TTable::makeWordVector(trg,i-1,HISTORY,kNULL);
							reverse(vec.begin(),vec.end());
							t2s.lm.addNgram(vec,f_j,p);
						}
						else{
							t2s.Increment(TTable::makeWordVector(trg, i-1, HISTORY, kNULL), f_j, p);
						}
						emp_feat += DiagonalAlignment::Feature(j, i, trg.size(), src.size()) * p;//what is this line doing?
					}
				}
				likelihood += log(sum);
			}
			if (final_iteration) cout << endl;
		}

		// log(e) = 1.0
		base2_likelihood = likelihood / log(2);

		if (flag) { 
			cerr << endl;
		}
		if (iter == 0) {
			mean_srclen_multiplier = tot_len_ratio / lc;
			cerr << "expected target length = source length * " << mean_srclen_multiplier << endl;
		}
		emp_feat /= toks;
		if(tof!=stderr){
			printProcess(tof);
		}
		printProcess(stderr);
		if (!final_iteration) {
			if (favor_diagonal && optimize_tension && iter > 0) {
				for (int ii = 0; ii < 8; ++ii) {
					double mod_feat = 0;
					unordered_map<pair<short,short>,unsigned,PairHash>::iterator it = size_counts.begin();
					for(; it != size_counts.end(); ++it) {
						const pair<short,short>& p = it->first;
						for (short j = 1; j <= p.first; ++j)
							mod_feat += it->second * DiagonalAlignment::ComputeDLogZ(j, p.first, p.second, diagonal_tension);
					}
					mod_feat /= toks;
					fprintf(tof,"  %d model al-feat: %lf (tension=%lf)\n",ii+1,mod_feat,diagonal_tension);
					diagonal_tension += (emp_feat - mod_feat) * 20.0;
					if (diagonal_tension <= 0.1){
						diagonal_tension = 0.1;
					}
					if (diagonal_tension > 14){
						diagonal_tension = 14;
					}
				}
				fprintf(tof,"     final tension: %lf\n",diagonal_tension);
			}
			switch(smooth){
				case VB:
					t2s.NormalizeVB(alpha);
					break;
				case KN:
					t2s.knEstimate();
				case NO:
				default:
					t2s.Normalize();
					break;
			}
			//prob_align_null *= 0.8; // XXX
			//prob_align_null += (c0 / toks) * 0.2;
			prob_align_not_null = 1.0 - prob_align_null;
		}
		if(ONE_INPUT_FILE){
			in.close();	
		}
		else{
			fin.close();
			ein.close();	
		}
	}
	/*if (!conditional_probability_filename.empty()) {
		cerr << "conditional probabilities: " << conditional_probability_filename << endl;
		t2s.ExportToFile(conditional_probability_filename.c_str(), d);
	}*/

	if(tof!=stderr){
		fclose(tof);
	}
	if(taof!=stdout){
		fclose(taof);
	}
	if (DO_TEST) {
		double tlp = 0;
		int lc = 0;
		string line,fline,eline;
		ifstream in,fin,ein;
		if(ONE_INPUT_FILE){
			in.open(fname.c_str(), ifstream::in);
			if (!in) {
				cerr << "Can't read " << fname << endl;
				return 1;
			}
		}
		else{
			fin.open(ffname.c_str(), ifstream::in);
			ein.open(efname.c_str(), ifstream::in);
			if(!fin){
				cerr << "Can't read " << ffname << endl;
				return 1;
			}
			if(!ein){
				cerr << "Can't read " << efname << endl;
				return 1;
			}
		}
		while (true) {
			++lc;
			vector<WordID> src, trg;
			if(ONE_TEST_FILE){
				getline(in,line);
			}
			else{
				getline(fin,fline);
				getline(ein,eline);
			}
			if (ONE_TEST_FILE && !in){
				break;
			}
			else if(!ONE_TEST_FILE && (!fin || !ein)){
				break;
			}

			if(ONE_TEST_FILE){
				ParseLine(line, &src, &trg);
			}
			else{
				ParseLineFromSeparateFiles(fline,eline,&src,&trg);
			}

			//cerr << TD::GetString(src) << " ||| " << TD::GetString(trg) << " |||";
			if (is_reverse){
				swap(src, trg);
			}
			bool first_al = true;  // used for write_alignments
			double log_prob = Md::log_poisson(src.size(), 0.05 + trg.size() * mean_srclen_multiplier);//why 0.05 is needed
			// compute likelihood
			for (unsigned j = 0; j < src.size(); ++j) {
				const WordID& f_j = src[j];
				double sum = 0;
				int a_j = 0;
				double max_pat = 0;
				double prob_a_i = 1.0 / (trg.size() + use_null);  // uniform (model 1)
				if (use_null) {
					WordVector wv;
					wv.push_back(kNULL);
					if (favor_diagonal){
						prob_a_i = prob_align_null;
					}
					max_pat = t2s.backoffProb(wv, f_j) * prob_a_i;
					sum += max_pat;
				}
				double az = 0;
				if (favor_diagonal){
					az = DiagonalAlignment::ComputeZ(j+1, trg.size(), src.size(), diagonal_tension) / prob_align_not_null;
				}
				for (unsigned i = 1; i <= trg.size(); ++i) {
					if (favor_diagonal){
						prob_a_i = DiagonalAlignment::UnnormalizedProb(j + 1, i, trg.size(), src.size(), diagonal_tension) / az;
					}
					double pat = t2s.prob(TTable::makeWordVector(trg,i-1,HISTORY,kNULL), f_j) * prob_a_i;
					if (pat > max_pat){
						max_pat = pat;
						a_j = i;
					}
					sum += pat;
				}
				log_prob += log(sum);
				//if (write_alignments) {
				if(true){
					if (a_j > 0) {
						if (first_al){
							first_al = false;
						}
						else{
							fprintf(taof," ");
						}
						if (is_reverse){
							fprintf(taof,"%d-%d",j , a_j - 1);
						}
						else{
							fprintf(taof, "%d-%d", a_j - 1, j);
						}
					}
				}
			}
			tlp += log_prob;
			fprintf(taof,"\n");
			//fprintf(taof," ||| %lf\n",log_prob);
			} // loop over test set sentences
			fprintf(tof,"TOTAL LOG PROB %lf\n",tlp);
			if(tof!=stderr){
				fclose(tof);
			}
			if(taof!=stdout){
				fclose(taof);
			}
		}

		return 0;
	}

