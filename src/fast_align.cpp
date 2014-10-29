#include <getopt.h>
#include "common.h"
#include "ttables.h"
#include "corpus.h"
#include "da.h"
#include "LM.h"

#include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
namespace po = boost::program_options;

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

bool verifyConf(po::variables_map* conf){
	return conf->count("help") ||
		(conf->count("input")!=0 && (conf->count("einput")!=0 || conf->count("finput")!=0)) ||
		(conf->count("input")==0 && conf->count("einput")==0 && conf->count("finput")==0) ||
		(conf->count("testset")!=0 && (conf->count("etestset")!=0 || conf->count("ftestset")!=0)) ||
		((conf->count("etestset")==0) ^ (conf->count("ftestset")==0));
}

bool InitCommandLine(int argc, char** argv, po::variables_map* conf) {
	po::options_description opts("Configuration options");
	opts.add_options()
		("input,i",po::value<string>(),"Parallel corpus input file")
		("finput",po::value<string>(),"source corpus input file")
		("einput",po::value<string>(),"target corpus input file")
		("reverse,r","Reverse estimation (swap source and target during training)")
		("iterations,I",po::value<unsigned>()->default_value(5),"Number of iterations of EM training")
		//("bidir,b", "Run bidirectional alignment")
		("favor_diagonal,d", "Use a static alignment distribution that assigns higher probabilities to alignments near the diagonal")
		("prob_align_null", po::value<double>()->default_value(0.08), "When --favor_diagonal is set, what's the probability of a null alignment?")
		("diagonal_tension,T", po::value<double>()->default_value(4.0), "How sharp or flat around the diagonal is the alignment distribution (<1 = flat >1 = sharp)")
		("train_line,l",po::value<int>()->default_value(-1),"how many lines to use when extracting sentences from training corpus.")
		("optimize_tension,o", "Optimize diagonal tension during EM")
		("variational_bayes,v","Infer VB estimate of parameters under a symmetric Dirichlet prior")
		("alpha,a", po::value<double>()->default_value(0.01), "Hyperparameter for optional Dirichlet prior")
		("no_null_word,N","Do not generate from a null token")
		("output_parameters,p", po::value<string>(), "Write model parameters to file")
		("beam_threshold,t",po::value<double>()->default_value(-4),"When writing parameters, log_10 of beam threshold for writing parameter (-10000 to include everything, 0 max parameter only)")
		("hide_training_alignments,H", "Hide training alignments (only useful if you want to use -x option and just compute testset statistics)")
		("testset,x", po::value<string>(), "After training completes, compute the log likelihood of this set of sentence pairs under the learned model")
		("ftestset", po::value<string>(), "Testset for source language. This must be paired with etestset and cannot be paired with testset.")
		("etestset", po::value<string>(), "Testset for target language. This must be paired with ftestset and cannot be paired with testset.")
		("test_align_output_file",po::value<string>(),"alignment result file for testdata")
		("train_align_output_file",po::value<string>(),"alignment result file for training data")
		("train_process_output_file",po::value<string>(),"ouput file for training process")
		("test_output_file",po::value<string>(),"output file for testdata except for alignment result")
		("no_add_viterbi,V","When writing model parameters, do not add Viterbi alignment points (may generate a grammar where some training sentence pairs are unreachable)")
		("force_align,f",po::value<string>(), "Load previously written parameters to 'force align' input. Set --diagonal_tension and --mean_srclen_multiplier as estimated during training.")
		("mean_srclen_multiplier,m",po::value<double>()->default_value(1), "When --force_align, use this source length multiplier")
		("history",po::value<int>()->default_value(0), "How many histories to use. When history is 0, translation probability is probability of ONE word to the other ONE word.");
	po::options_description clo("Command line options");
	clo.add_options()
		("config", po::value<string>(), "Configuration file")
		("help,h", "Print this help message and exit");
	po::options_description dconfig_options, dcmdline_options;
	dconfig_options.add(opts);
	dcmdline_options.add(opts).add(clo);

	po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
	if (conf->count("config")) {
		ifstream config((*conf)["config"].as<string>().c_str());
		po::store(po::parse_config_file(config, dconfig_options), *conf);
	}
	po::notify(*conf);
	if (verifyConf(conf)) {
		cerr << "Usage " << argv[0] << " [OPTIONS] -i corpus.fr-en\n";
		cerr << dcmdline_options << endl;
		return false;
	}
	return true;
}

int main(int argc, char** argv) {
	po::variables_map conf;

	if (!InitCommandLine(argc, argv, &conf)){
		return 1;
	}

	bool use_null = !no_null_word;
	if (variational_bayes && alpha <= 0.0) {
		cerr << "--alpha must be > 0\n";
		return 1;
	}
	double prob_align_not_null = 1.0 - prob_align_null;
	const unsigned kNULL = d.Convert("<eps>");
	TTable t2s(HISTORY + 1);
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

