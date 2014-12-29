#include "LM.h"
int _debug=0;

Node* LM::setNode(WordID* context, int clen){
	Node* pNode=&_root;
	for(; clen>0; clen--, context++){
		int wrd=*context;
		pNode=&(pNode->childs[wrd]);
	}
	return pNode;
}

// void Node::normalize(){
// 	double total=0;
// 	for(ZMap<WordID, double>::iterator iter=probs.begin();iter!=probs.end();iter++){
// 		total+=iter->second;
// 	}
// 	for(ZMap<WordID, double>::iterator iter=probs.begin();iter!=probs.end();iter++){
// 		iter->second=log10(iter->second/total);
// 	}
// 	for(ZMap<WordID, Node>::iterator iter=childs.begin();iter!=childs.end();iter++){
// 		iter->second.normalize();
// 	}
// }
//
// void Node::prune(double threshold){
// 	for(ZMap<WordID, double>::iterator iter=probs.begin();iter!=probs.end();){
// 		ZMap<WordID, double>::iterator curIter=iter;
// 		iter++;
// 		if(curIter->second<=threshold){
// 			WordID wrd=curIter->first;
// 			probs.erase(probs.find(wrd));
// 			types.erase(types.find(wrd));
// 			//cout<<"prune "<<curIter->second<<" "<<threshold<<endl;
// 		}
// 	}
// 	//if(probs.size()==0)childs.clear();
// }
//
// void Node::erase(WordID wrd){
// 	ZMap<WordID, double>::iterator iter=probs.find(wrd);
// 	if(iter!=probs.end())probs.erase(iter);
// 	ZMap<WordID, FracType>::iterator titer=types.find(wrd);
// 	if(titer!=types.end())types.erase(titer);
// }
//
// Node* LM::lookupNode(WordID* context, int clen){
// 	Node* pNode=&_root;
// 	for(;clen>0;clen--,context++){
// 		WordID wrd=*context;
// 		if(pNode->childs.find(wrd)!=pNode->childs.end())
// 			pNode=&(pNode->childs[wrd]);
// 		else return NULL;
// 	}
// 	return pNode;
// }
//
// double& LM::setProb(WordID wrd, WordID* context, int clen){
// 	Node* pNode=setNode(context,clen);
// 	if(pNode->probs.find(wrd)==pNode->probs.end())
// 		pNode->probs[wrd]=0;
// 	return pNode->probs[wrd];
// }

Node* LM::addFracCount(WordID wrd, WordID* context, int clen, double fcount, double ftype){
	Node* pNode=setNode(context,clen);
	if(pNode->probs.find(wrd)==pNode->probs.end()){
		pNode->probs[wrd]=0;
	}
	pNode->probs[wrd]+=fcount;
	pNode->types[wrd].update(ftype);
	return pNode;
}

// Node* LM::addFracCount(WordID wrd, WordID* context, int clen, double fcount, FracType& ftype){
// 	Node* pNode=setNode(context,clen);
// 	pNode->probs[wrd]=fcount;
// 	pNode->types[wrd]=ftype;
// 	return pNode;
// }
//
// void LM::wbEstimate(Node& node){
// 	double ntype=0;
// 	double ntoken=0;
// 	ZMap<WordID,double>::iterator iter;
// 	iter=node.probs.begin();
// 	for(;iter!=node.probs.end();iter++)
// 		ntoken+=iter->second;
//
// 	ZMap<WordID,FracType>::iterator tIter;
// 	tIter=node.types.begin();
// 	for(;tIter!=node.types.end();tIter++)
// 		ntype-=expm1(tIter->second[0]);
//
// 	//	ntype+=(1-iter->second);
// 	//cerr<<"ntype:"<<ntype<<endl;
// 	if(ntype==0){cerr<<"error, ntype is 0"; exit(1);}
// 	node.bow=log10(ntype)-log10(ntype+ntoken);
//
// 	iter=node.probs.begin();
// 	for(;iter!=node.probs.end();iter++){
// 		if(iter->second==0){cerr<<"error, nfrac is 0"; exit(1);}
// 		iter->second=log10(iter->second)-log10(ntype+ntoken);
// 	}
// 	node.types.clear();
//
// 	ZMap<WordID,Node>::iterator childIter=node.childs.begin();
// 	for(;childIter!=node.childs.end();childIter++)
// 		wbEstimate(childIter->second);
// }
//
// void LM::mlEstimate(Node& node){
// 	double ntoken=0;
// 	ZMap<WordID,double>::iterator iter;
// 	iter=node.probs.begin();
// 	for(;iter!=node.probs.end();iter++)
// 		ntoken+=iter->second;
// 	node.bow=-99;
//
// 	iter=node.probs.begin();
// 	for(;iter!=node.probs.end();iter++){
// 		if(iter->second==0){cerr<<"error, nfrac is 0"; exit(1);}
// 		iter->second=log10(iter->second)-log10(ntoken);
// 	}
// 	node.types.clear();
//
// 	for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();iter++){
// 		mlEstimate(iter->second);
// 	}
// }

void LM::computeBOW(bool interpolate){
	WordVector context; 
	for(int i=0;i<order();i++){
		computeBOWByLayer(_root,context,i+1, interpolate);
	}
}

// void LM::setSomeDetails(){
// 	_root.probs[_vocab.add(unk)]=_root.bow;
// 	_root.probs[_vocab.add(sent_begin)]=-99;
// 	calcNumOfGrams();
// }

// double& LM::setBackoff(WordID* context, int clen){
// 	Node* pNode=setNode(context,clen);
// 	return pNode->bow;
// }

double LM::probBO(WordID wrd, WordID* context, int clen){//i confirmed that this function is CORRECT.
	double prob=0;
	double bow=0;
	Node* pNode=&_root;
	for(;clen>=0;context++,clen--){
		if(pNode->probs.find(wrd)!=pNode->probs.end()){
			prob=pNode->probs[wrd];
			bow=0;
		}
		if(clen==0){
			break;
		}
		WordID cwrd=*context;
		if(pNode->childs.find(cwrd)==pNode->childs.end()){
			/*
			 * P_{BO}(w|u) =
			 * 		P*(w|u) (uw is in model)
			 *		\beta(u)P_{BO} (otherwise)
			 * here P* is discounted probability and \beta(u) is discount weight.
			 * \beta(u) is represented as follows.
			 * \beta(u) = \frac{1-\sum_{w:uw is in model}P*(w|u)}{1-\sum_{w:uw is in model}P*(w|u')}
			 * For the derivation of \beta(u), please refer to http://cxwangyi.wordpress.com/2010/07/28/backoff-in-n-gram-language-models/
			 * 
			 * So in this case, that is, if no context is found in the learned model, {w:uw is in model} is empty set. Therefore, \beta{u} = 1
			 * If we take log, then there is nothing to add to current bow, so we just break from the for loop.
			 * */
			break;
		}
		pNode=&(pNode->childs[cwrd]);
		bow+=pNode->bow;
	}
	double penalty=0; //in case clen > 0
	return prob+bow+penalty*clen;
}
//
// bool LM::checkEntry(WordID wrd, WordID* context, int clen){
// 	Node* pNode=&_root;
// 	for(;clen>=0;context++,clen--){
// 		if(pNode->probs.find(wrd)==pNode->probs.end()){
// 			return false;
// 		}
// 		if(clen==0){	
// 			break;
// 		}
// 		WordID cwrd=*context;
// 		if(pNode->childs.find(cwrd)==pNode->childs.end()){
// 			return false;
// 		}
// 		pNode=&(pNode->childs[cwrd]);
// 	}
// 	return true;
// }
//
// void LM::stat(WordID wrd, WordID* context, int clen, LMStat& lmstat){
// 	int real_clen=clen+1;
// 	Node* pNode=&_root;
// 	for(;clen>=0;context++,clen--){
// 		if(clen==0){
// 			break;
// 		}
// 		WordID cwrd=*context;
// 		if(pNode->childs.find(cwrd)==pNode->childs.end()){
// 			break;
// 		}
// 		pNode=&(pNode->childs[cwrd]);
// 	}
// 	real_clen-=clen;
// 	if(wrd==_vocab.add(unk)){
// 		real_clen=0;
// 	}
// 	lmstat.ngrams[real_clen]++;
// 	return;
// }
//
// void LM::read(string filename){
// 	ifstream is(filename.c_str());
// 	while(!is.eof()){
// 		string curline="";
// 		getline(is,curline);
// 		if(curline=="\\data\\"){
// 			//cout<<endl<<curline<<endl;
// 			readNumofGrams(is);
// 		}
// 		else if(curline.find("-grams:")!=string::npos){
// 			//cout<<curline<<endl;
// 			curline=curline.substr(curline.find("\\")+1);
// 			curline=curline.substr(0,curline.find("-"));
// 			int order=atoi(curline.c_str());
// 			readNgram(is,order);
// 		}
// 		//else if(curline=="\\end\\")	cout<<curline<<endl;
// 	}
// }
//
// int LM::readNumofGrams(ifstream& is){
// 	string curline="";
// 	while(!is.eof()){
// 		getline(is,curline);
// 		//cout<<curline<<endl;
// 		if(curline.find("ngram")!=0){
// 			break;
// 		}
// 		curline.erase(0,curline.find("=")+1);
// 		_numofGrams.push_back(atoi(curline.c_str()));
// 	}
// 	return (int)_numofGrams.size();
// }
//
// void LM::readNgram(ifstream& is, int order){
// 	string curline="";
// 	while(!is.eof()){
// 		getline(is,curline);
// 		if(curline=="")
// 		{
// 			//cout<<curline<<endl;
// 			return;
// 		}
// 		vector<string> wrds;
// 		stringToVector(curline,wrds);
// 		double probRead=atof(wrds[0].c_str());
// 		WordVector context;
// 		_vocab.lookup(&wrds[1],context,order,true);
// 		_vocab.reverse(context);
// 		double& prob=setProb(context[0],&context[0]+1,order-1);
// 		prob=probRead;
// 		//cout<<probRead;
// 		//for(int i=0;i<order;i++)cout<<" "<<wrds[i+1];
//
// 		if((int)wrds.size()==order+2){
// 			double bowRead=atof(wrds.back().c_str());
// 			//if(order>1)cerr<<"setting bow "<<bowRead<<endl;
// 			double& bow=setBackoff(&context[0],order);
// 			bow=bowRead;
// 			//cout<<" "<<bowRead;
// 		}
// 		//cout<<endl;
// 	}
// }

void LM::clearCountsByLayer(Node& node, int order, int exception){
	if(order>1){
		for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();iter++){
			if((int)iter->first!=exception){
				clearCountsByLayer(iter->second,order-1,exception);
			}
		}
	}
	else{
		node.probs.clear();
		node.types.clear();
	}
}

void LM::setKNcountsByLayer(Node& node, WordVector& context, int order){
	if(order>1){
		for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();iter++){
			context.push_back(iter->first);
			setKNcountsByLayer(iter->second,context,order-1);
			context.pop_back();
		}
	}
	else{
		int clen = (int)context.size() - 1;
		if(clen < 0){
			return;
		}
		//cerr<<"lostType size:"<<node.types.size()<<endl;
		for(ZMap<WordID,FracType>::iterator typeIter=node.types.begin();typeIter!=node.types.end();typeIter++){
			//On count distributions, for each n-gram type vu'w we, we generate an (n-1)-gram token u'w with probability p(c(vu'w>0)) = 1 - p(c(vu'w)=0)
			addFracCount(typeIter->first,&context[0],clen,1-typeIter->second[0],1-typeIter->second[0]);
		}
	}
}

// void LM::setKNcountsByLayerTam(Node& node, WordVector& context, int order, double threshold){
// 	if(order>1){
// 		for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();iter++){
// 			context.push_back(iter->first);
// 			setKNcountsByLayerTam(iter->second,context,order-1,threshold);
// 			context.pop_back();
// 		}
// 	}
// 	else{
// 		int clen=(int)context.size()-1;
// 		if(clen<0){
// 			return;
// 		}
// 		//cerr<<"lostType size:"<<node.types.size()<<endl;
// 		for(ZMap<WordID,double>::iterator probIter=node.probs.begin();probIter!=node.probs.end();probIter++){
// 			double count=probIter->second;
// 			if(count>threshold){
// 				count=1;
// 			}
// 			addFracCount(probIter->first,&context[0],clen,count,count);
// 		}
// 	}
// }
//
// void LM::setMissKNcountsByLayer(Node& node, WordVector& context, int order){
// 	if(order>1){
// 		for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();iter++){
// 			context.push_back(iter->first);
// 			setMissKNcountsByLayer(iter->second,context,order-1);
// 			context.pop_back();
// 		}
// 	}
// 	else{
// 		for(int clen=(int)context.size()-1;clen>=(int)context.size()-1;clen--){
// 			if(clen<0){
// 				return;
// 			}
// 			Node* pnode=lookupNode(&context[1],clen);
// 			if(pnode==NULL){
// 				setNode(&context[1],clen);
// 			}
// 			if(pnode->probs.find(context[0])==pnode->probs.end()){
// 				double totalProb=0;
// 				for(ZMap<WordID,double>::iterator probIter=node.probs.begin();probIter!=node.probs.end();probIter++){
// 					totalProb+=probIter->second;
// 				}
// 				if(totalProb!=0){
// 					addFracCount(context[0],&context[1],clen,totalProb,totalProb);
// 				}
// 			}
// 		}
// 		//cerr<<"lostType size:"<<node.types.size()<<endl;
// 	}
// }
//
// void LM::cleanBadEntriesByLayer(Node& node, WordVector& context, int order){
// 	if(order>1)
// 	{
// 		for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();)
// 		{
// 			ZMap<WordID,Node>::iterator curIter=iter;
// 			iter++;
// 			context.push_back(curIter->first);
// 			cleanBadEntriesByLayer(curIter->second,context,order-1);
// 			if(curIter->second.probs.size()==0)node.childs.erase(curIter);
// 			context.pop_back();
// 		}
// 	}
// 	else
// 	{
// 		int clen=(int)context.size()-1;
// 		if(context[0]!=_vocab.lookup(sent_begin,false))
// 			if(!checkEntry(context[0],&context[0]+1,clen)){node.probs.clear();node.types.clear();node.childs.clear();}
// 	}
// }

void LM::calculateDiscounts(vector<vector<vector<double> > >& discounts, bool logrized){
	for(int i=0;i<order();i++){
		vector<vector<double> > discount(d.max());
		discounts.push_back(discount);
		vector<vector<double> > coc(d.max());
		for(unsigned int j = 0;j < d.max();++j){
			coc[j].resize(4,0);
		}
		collectCountofCount(_root,coc,i+1,logrized,kNULL);
		calculateDiscount(coc,discounts[i],i);
		/*for(int j=0;j<(int)coc.size();j++){
			cerr<<"n"<<j+1<<": "<<coc[j]<<endl;
		}*/
		/*for(int j=0;j<(int)discounts[i].size()-1;j++){
			cerr<<"D"<<j+1<<": "<<discounts[i][j]<<endl;
		}*/
	}
}

void calculateDiscount(vector<vector<double> >& coc, vector<vector<double> >& discount,int order){
	for(unsigned int e_i = 0;e_i<d.max();++e_i){
		discount[e_i].clear();
		double n1=coc[e_i][0];
		double n2=coc[e_i][1];
		double n3=coc[e_i][2];
		double n4=coc[e_i][3];
		double Y=n1/(n1+2*n2);
		double D1 = 0.0, D2 = 0.0, D3 = 0.0;
		if(n1>1E-10){
			D1 = 1-2*Y*n2/n1;
		}
		if(n2>1E-10){
			D2 = 2-3*Y*n3/n2;
		}
		if(n3>1E-10){
			D3 = 3-4*Y*n4/n3;
		}
		discount[e_i].push_back(D1);
		discount[e_i].push_back(D2);
		discount[e_i].push_back(D3);
		discount[e_i].push_back(D3);
		cout<<d.Convert(e_i+1)<<"::n1="<<n1<<",n2="<<n2<<",n3="<<n3<<",n4="<<n4<<endl;
		for(size_t i=1;i<discount[e_i].size();i++){
			if(discount[e_i][i]<0){
				cerr<<"discount["<<e_i<<"] "<<i<<":"<<discount[e_i][i]<<endl;
				discount[e_i][i]=discount[e_i][i-1];
			}
		}
		if(order==0){
			break;
		}
	}
}

// int discountSlot(double count){
// 	for(int i=1;i<4;i++){
// 		if(count<=double(i)){
// 			return i-1;
// 		}
// 	}
// 	return 3;
// }
//
// void LM::calcNumOfGrams(Node& node, int order){
// 	while((int)_numofGrams.size()<=order){
// 		_numofGrams.push_back(0);
// 	}
// 	_numofGrams[order]+=(int)node.probs.size();
// 	for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();iter++){
// 		calcNumOfGrams(iter->second,order+1);
// 	}
// 	return ;
// }
//
// void LM::reComputeLowerCount(){
// 	for(int order=(int)(_numofGrams.size());order>1;order--){
// 		WordVector context;
// 		reComputeLowerCount(_root,context,order);
// 	}
// }
//
// void LM::reComputeLowerCount(Node& node, WordVector& context, int order){
// 	if(order>1){
// 		for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();iter++){
// 			context.push_back(iter->first);
// 			reComputeLowerCount(iter->second,context,order-1);
// 			context.pop_back();
// 		}
// 	}
// 	else{
// 		_vocab.reverse(context);
// 		for(ZMap<WordID,double>::iterator iter=node.probs.begin();iter!=node.probs.end();iter++){
// 			addFracCount(iter->first,&context[0]+1,(int)context.size()-1,iter->second,1-node.types[iter->first][0]);
// 		}
// 		_vocab.reverse(context);
// 	}
// }

void LM::computeBOWByLayer(Node& node, WordVector& context, int order, bool interpolate){
	if(order>1){
		for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();iter++){
			context.push_back(iter->first);
			computeBOWByLayer(iter->second,context,order-1,interpolate);
			context.pop_back();
		}
	}
	else{
		double numerator=1.0;
		double denumerator=1.0;
		//cerr<<"lower weight: "<<exp10(node.bow)<<endl;
		//double sumu=0;/
		for(ZMap<WordID,double>::iterator iter=node.probs.begin();iter!=node.probs.end();iter++){
			double lowLogProb=0;
			if(context.size()>0){
				lowLogProb=probBO(iter->first,&context[0],(int)context.size()-1);//getting log(p'(w|u'))
				denumerator-=exp10(lowLogProb);
			}
			else{
				lowLogProb=-log10((double)node.probs.size()+1);//uniform distribution
			}
			iter->second=exp10(iter->second);
			if(interpolate){
				iter->second+=exp10(node.bow+lowLogProb);
			}

			numerator-=iter->second;
			iter->second=log10(iter->second);
		}
		/*
		 * numerator = 1 - \sum_{w : uw is in model}p(w|u)  
		 * denumerator = 1 - \sum_{w : u'w is in model}p(w|u') 
		 * therefore,
		 * \beta{uw} = numerator/denumerator
		 */
		//cerr<<"numerator:"<<numerator<<", sumu:"<<sumu<<endl;
		if(context.size()>0){
			node.bow=log10(numerator)-log10(denumerator);
		}
		else{
			node.bow-=log10((double)node.probs.size()+1);
		}
	}
}

void LM::collectCountofCount(Node& node, vector<vector<double> >& coc, int order, bool logrized,WordID e_i){
	if(order>1){
		for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();iter++){
			collectCountofCount(iter->second,coc,order-1, logrized,iter->first);
		}
	}
	else{
		for(ZMap<WordID,FracType>::iterator typeIter=node.types.begin();typeIter!=node.types.end();typeIter++){
			FracType& ftype=typeIter->second;
			for(int j=1;j<5;j++){
				coc[e_i-1][j-1] += ftype[j];//coc[j] = E[n_{j+1}](0<=j<=3)
			}
		}
	}
	return ;
}

// void LM::resetVocab(){
// 	vector<int> ids(_vocab.size(),0);
// 	for(ZMap<WordID,double>::iterator iter=_root.probs.begin();iter!=_root.probs.end();iter++){
// 		ids[iter->first]=1;
// 	}
// 	ids[_vocab.lookup(sent_begin,false)]=1;
// 	ids[_vocab.lookup(sent_end,false)]=1;
// 	ids[_vocab.lookup(unk,false)]=1;
// 	for(size_t id=1;id<ids.size();id++){
// 		if(ids[id]==0)_vocab.remove((int)id);
// 	}
// }

void LM::knEstimate(bool interpolate){
	vector<vector<vector<double> > > discounts;//discounts[order][e_i][count];
	setKNcounts();
	calculateDiscounts(discounts);
	// for(unsigned int i = 0;i<discounts[1].size();++i){
	// 	for(unsigned int j = 0;j<discounts[1][i].size();++j){
	// 		cout<<discounts[1][i][j]<<" ";	
	// 	}	
	// 	cout<<endl;
	// }
	adEstimate(_root, discounts, 0, kNULL);
	computeBOW(interpolate);
}

// void LM::normalize(){
// 	_root.normalize();
// }
//
// void LM::prune(Node& node, double* thresholds, int order){
// 	double threshold=0;
// 	if(thresholds!=NULL)threshold=thresholds[order];
// 	node.prune(threshold);
// 	for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();iter++){
// 		prune(iter->second,thresholds,order+1);
// 	}
// }

void LM::adEstimate(Node& node, vector<vector<vector<double> > >& discounts, int order, WordID e_i){
	vector<double> discount = discounts[order][e_i-1];
	double subtract=0;
	double total=0;//total = \sum_{w} E[c(uw)] = E[c(u?)]
	if(node.probs.size()>0){
		for(ZMap<WordID,double>::iterator iter=node.probs.begin();iter!=node.probs.end();iter++){
			total+=iter->second;//iter->second = E[c(uw)]
			FracType& ft=node.types[iter->first];
			//cerr<<"start getting discountMass"<<endl;
			double dMass=discountMass(ft,discount);
			//cerr<<"end getting discountMass"<<endl;
			//p(c(uw)=1)*D_{1} + p(c(uw)=2)*D_{2} + p(c(uw)>=3)*D_{3+}
			// if(iter->second<dMass){
			// 	cerr<<"iter->second is smaller than dMass"<<endl;
			// 	cerr<<"ft[1]="<<ft[1]<<",ft[2]="<<ft[2]<<",ft[3]="<<ft[3]<<endl;
			// 	cerr<<"iter->second="<<iter->second<<",dMass="<<dMass<<endl;
			// 	cerr<<"iter->second-dMass="<<iter->second-dMass<<endl;
			// 	exit(1);
			// }
			iter->second-=dMass;
			if(iter->second<0){
				iter->second = 1E-10;
			}
			subtract+=dMass;
		}
		node.bow=log10(subtract)-log10(total);
	}

	for(ZMap<WordID,double>::iterator iter=node.probs.begin();iter!=node.probs.end();iter++){
		// double tmp = iter->second;
		iter->second=log10(iter->second)-log10(total);
		// if(std::isnan(iter->second)){
		// 	cerr<<"isnan encountered:iter->second="<<tmp<<",total="<<total<<endl;
		// 	exit(1);
		// }
	}

	for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();iter++){
		adEstimate(iter->second,discounts,order+1,iter->first);
	}
}

void LM::setKNcounts(){// we need to propagate counts backwards
	for(int i=order(); i >= 2; i--){
		WordVector context;
		clearCountsByLayer(_root,i-1,_vocab.add(sent_begin));
		setKNcountsByLayer(_root,context,i);
	}
	return ; 
}

// void LM::setKNcountsTam(double threshold){
// 	for(int i=2;i<=order();i++){
// 		int layer=i;
// 		WordVector context;
// 		clearCountsByLayer(_root,layer-1,_vocab.add(sent_begin));
// 		setKNcountsByLayerTam(_root,context,layer,threshold);
// 	}
// }
//
// void LM::print(ostream& os, Node& node, string sufix, WordVector& context, int order, bool countOnly){
// 	if(order>1){
// 		for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();iter++){
// 			string newsufix=_vocab.int2str(iter->first)+" "+sufix;
// 			context.push_back(iter->first);
// 			print(os,iter->second,newsufix,context,order-1,countOnly);
// 			context.pop_back();
// 		}
// 	}
// 	else{
// 		for(ZMap<WordID,double>::iterator iter=node.probs.begin();iter!=node.probs.end();iter++){
// 			string str=_vocab.int2str(iter->first);
// 			if(sufix!=""){
// 				str=sufix+str;
// 			}
// 			string probStr=doubleToString(iter->second);
//
// 			if(countOnly){
// 				//for(int i=0;i<(int)context.size();i++)os<<context[i]<<" ";
// 				//os<<iter->first<<" "<<iter->second<<endl;
// 				os<<str+"\t"+probStr<<"\ttype: "<<node.types[iter->first]<<endl;
// 				continue;
// 			}
// 			else{
// 				str=probStr+"\t"+str;
// 				os<<str;
// 				//lookup BOW
// 				WordVector bowContext;
// 				bowContext.push_back(iter->first);
// 				bowContext.insert(bowContext.end(),context.begin(),context.end());
//
// 				Node* tmpNode=lookupNode(&bowContext[0],(int)bowContext.size());
// 				if(tmpNode!=NULL){
// 					os<<"\t"<<tmpNode->bow<<endl;
// 				}
// 				else{
// 					os<<"\t"<<0<<endl;
// 				}
// 			}
// 		}
// 	}
// }
//
// void LM::write(string filename){
// 	ofstream os(filename.c_str());
// 	WordVector context;
//
// 	os<<"\n\\data\\"<<endl;
//
// 	for(int i=1;i<=(int)_numofGrams.size();i++){
// 		os<<"ngram "<<i<<"="<<_numofGrams[i-1]<<endl;
// 	}
// 	os<<endl;
//
// 	for(int i=1;i<=(int)_numofGrams.size();i++){
// 		os<<"\\"<<i<<"-grams:"<<endl;
// 		print(os,_root,"",context,i,false);
// 		os<<endl;
// 	}
// 	os<<"\\end\\"<<endl;
// 	os.close();
// }
//
// void LM::writeCount(string filename){
// 	ofstream os(filename.c_str());
// 	WordVector context;
//
// 	for(int i=1;i<=(int)_numofGrams.size();i++){
// 		print(os,_root,"",context,i,true);
// 	}
// 	os.close();
// }
//
// double LM::sentLogProb(string& sentence, int ord, LMStat* pStat){
// 	string curline="<s> "+sentence+" </s>";
// 	vector<string> wrds;
// 	stringToVector(curline,wrds);
// 	if(wrds.size()==0)return 0;
// 	WordVector context;
// 	_vocab.lookup(&wrds[0],context,(int)wrds.size(),false);
// 	if(pStat!=NULL)pStat->nwrds=(int)wrds.size();
// 	return sentLogProb(&context[0],(int)context.size(),ord,pStat);
// }
//
// double LM::sentLogProb(WordID* context, int slen, int order, LMStat* pStat){
// 	double prob=0;
// 	_vocab.reverse(context,slen);
// 	for(int i=0;i<slen-1;i++){
// 		int clen=slen-i-1>order-1?order-1:slen-i-1;
// 		double curProb=probBO(context[i],context+i+1,clen);
// 		if(_debug==2){
// 			cout<<curProb<<": "<<_vocab.int2str(context[i])<<" | "<<_vocab.int2str(context[i+1])<<endl;
// 		}
// 		prob+=curProb;
// 		if(pStat!=NULL){
// 			stat(context[i],context+i+1,clen,*pStat);
// 		}
// 	}
// 	_vocab.reverse(context,slen);
// 	return prob;
// }
//
// double LM::ppl(string filename, LMStat* pStat, string probFilename,bool average){
// 	ifstream is(filename.c_str());
// 	double prob=0;
// 	int nwrds=0;
// 	ofstream os;
// 	if(probFilename!=""){
// 		os.open(probFilename.c_str());
// 	}
// 	while(!is.eof()){
// 		string curline="";
// 		getline(is,curline);
// 		curline="<s> "+curline+" </s>";
// 		vector<string> wrds;
// 		stringToVector(curline,wrds);
// 		if(wrds.size()==2){
// 			continue;
// 		}
// 		WordVector context;
// 		_vocab.lookup(&wrds[0],context,(int)wrds.size(),false);
// 		double slp=sentLogProb(&context[0],(int)context.size(),order(),pStat);
// 		prob+=slp;
// 		nwrds+=(int)(context.size()-1);
// 		if(probFilename!=""){
// 			if(average){
// 				os<<slp/(double)(context.size()-1)<<" "<<context.size()-1<<endl;
// 			}
// 			else{
// 				os<<slp<<" "<<context.size()-1<<endl;
// 			}
// 		}
// 	}
// 	if(pStat!=NULL)	{
// 		pStat->nwrds=nwrds;
// 	}
// 	if(probFilename!=""){
// 		os.close();
// 	}
// 	return prob;
// }
//
// void LM::readCountFile(string filename){
// 	ifstream is(filename.c_str());
// 	int order=0;
// 	while(!is.eof()){
// 		string curline="";
// 		getline(is,curline);
// 		vector<string> wrds;
// 		stringToVector(curline,wrds);
// 		if((int)wrds.size()<2){
// 			continue;
// 		}
// 		WordVector context;
// 		double count=atof(wrds.back().c_str());
// 		for(int i=0;i<(int)wrds.size()-1;i++){
// 			context.push_back(_vocab.add(wrds[i]));
// 			//cerr<<wrds[i]<<" ";
// 		}
// 		//cerr<<count<<endl;
// 		_vocab.reverse(context);
//
//
// 		addFracCount(context[0],&context[0]+1,(int)context.size()-1,count,count);
// 		order=max(order,(int)context.size());
// 	}
// 	for(int i=0;i<order;i++){
// 		if((int)_numofGrams.size()<=i)
// 			_numofGrams.push_back(0);
// 	}
// 	is.close();
// }
//
//
// bool LM::addOneLine(string& curline, int order, double*& p_weight, double threshold, double floor){
// 	double fraccount=1;
// 	if(p_weight!=NULL){
// 		fraccount=*p_weight;
// 		p_weight++;
// 	}
// 	if(fraccount<=threshold){
// 		return false;
// 	}
// 	if(fraccount<floor){
// 		fraccount=floor;
// 	}
//
// 	curline="<s> "+curline+" </s>";
// 	vector<string> wrds;
// 	stringToVector(curline,wrds);
// 	WordVector context;
//
// 	for(int i=0;i<(int)wrds.size();i++){
// 		//int ind=_vocab.add(wrds[i]);
// 		context.push_back(_vocab.add(wrds[i]));
// 	}
// 	_vocab.reverse(context);
//
// 	for(int i=0;i<(int)context.size()-1;i++){
// 		int maxDepth=min((int)context.size()-i-1,order-1);
// 		for(int depth=0;depth<=maxDepth;depth++){
// 			addFracCount(context[i],&context[i]+1,depth,fraccount,fraccount);
// 		}
// 	}
// 	return true;
// }

void LM::setOrder(int order){
	for(int i=0;i<order;i++){
		if((int)_numofGrams.size()<=i){
			_numofGrams.push_back(0);
		}
	}
}

// void LM::estimate(Smoothing sm, bool interpolate, double* tamcutoff){
// 	if(sm==KN){
// 		knEstimate(interpolate,tamcutoff);
// 	}
// 	/*else if(sm==WB){
// 	  wbEstimate(interpolate,tamcutoff);
// 	  }*/
// 	else if(sm==NO){
// 		mlEstimate(_root);
// 	}
// 	setSomeDetails();
// }
//
// void LM::wbEstimate(bool interpolate, double* cutoff){
// 	//prune(cutoff);
// 	wbEstimate(_root);
// 	computeBOW(interpolate);
// }
//
// void LM::knEstimate(bool interpolate, double* tamcutoff){
// 	knEstimate(false,interpolate,D_AD,tamcutoff);
// }
//
// int LM::readPlainFile(string filename, int order, double* p_weight, double threshold, double floor, int nSentence){
// 	ifstream is(filename.c_str());
// 	if(!is.good()){
// 		return 0;
// 	}
// 	init(order);
//
// 	int sentCount=0;
// 	int usedCount=0;
// 	while(!is.eof()){
// 		string curline="";
// 		getline(is,curline);
// 		if(curline==""){
// 			continue;
// 		}
// 		sentCount++;
// 		if(sentCount>nSentence&&nSentence>0){
// 			break;
// 		}
// 		usedCount+=addOneLine(curline,order,p_weight,threshold,floor);
// 	}
// 	is.close();
// 	return usedCount;
// }
//
// //void readCountFile(string filename, unordered_map<string,FracType>& ngrams);
//
// void readPlainFile(string filename, int order, double*& p_weight, int nSentence,unordered_map<string,pair<FracType,double>>& ngrams){
// 	ifstream is(filename.c_str());
// 	int sentCount=0;
// 	while(!is.eof()){
// 		string curline="";
// 		getline(is,curline);
// 		if(curline==""){
// 			continue;
// 		}
// 		sentCount++;
// 		if(sentCount>nSentence&&nSentence>0){
// 			break;
// 		}
// 		double fcount=1;
// 		if(p_weight!=NULL){
// 			fcount=*p_weight++;
// 		}
// 		vector<string> wrds;
// 		curline="<s> "+curline+" </s>";
// 		split(wrds,curline,is_any_of(" \t"));
//
// 		string str="";
// 		for(int len=0;len<order-1;len++){
// 			str+=wrds[len];
// 			if(ngrams.find(str)==ngrams.end()){
// 				ngrams[str].second=0;
// 			}
// 			pair<FracType,double>& p=ngrams[str];
// 			p.first.update(fcount);
// 			p.second+=fcount;
// 			str+=" ";
// 		}
//
// 		for(int i=0;i<(int)wrds.size()-order+1;i++){
// 			string str="";
// 			int len=0;
// 			for(;len<order-1;len++){
// 				str+=wrds[i+len]+" ";
// 			}
// 			str+=wrds[i+len];
// 			if(ngrams.find(str)==ngrams.end()){
// 				ngrams[str].second=0;
// 			}
// 			pair<FracType,double>& p=ngrams[str];
// 			p.first.update(fcount);
// 			p.second+=fcount;
// 		}
// 	}
// 	if(_debug){
// 		for(unordered_map<string,pair<FracType,double>>::iterator it = ngrams.begin();it!=ngrams.end();++it){       
// 			cout<<it->first<<" ||| count:"<<it->second.second<<" type:"<<it->second.first<<endl;
// 		}
// 	}
// }
//
// void LM::addNgrams(unordered_map<string,pair<FracType,double>>& ngrams){
// 	int order=0;
// 	_vocab.add(unk);
// 	for(unordered_map<string,pair<FracType,double>>::iterator it = ngrams.begin();it!=ngrams.end();++it){       
// 		vector<string> wrds;
// 		split(wrds,it->first,is_any_of(" \t"));
// 		if((int)wrds.size()<2){
// 			continue;
// 		}
// 		WordVector context;
// 		for(int i=0;i<(int)wrds.size();i++){
// 			context.push_back(_vocab.add(wrds[i]));
// 		}
// 		_vocab.reverse(context);
//
// 		addFracCount(context[0],&context[0]+1,(int)context.size()-1,it->second.second,it->second.first);
// 		for(int depth=0;depth<(int)context.size()-1;depth++){
// 			addFracCount(context[0],&context[0]+1,depth,it->second.second,it->second.second);
// 		}
// 		order=max(order,(int)context.size());
// 	}
//
// 	for(int i=0;i<order;i++){
// 		if((int)_numofGrams.size()<=i){
// 			_numofGrams.push_back(0);
// 		}
// 	}
// }

void LM::addNgram(WordVector& context,WordID wrd, double count){//context should be reversed beforehand
	if(count>1E-10){
		addFracCount(wrd, &context[0], context.size(), count, count);
	}	
	return ;
}

void LM::copyDiscountedProb(VWV2WD& ttables,bool copyAllProb){
	int len = ttables.size();
	ttables.clear();
	ttables.resize(len);
	WordVector context;
	_copyDiscountedProb(_root,context,ttables,copyAllProb);
}

void LM::_copyDiscountedProb(Node& node,WordVector& context,VWV2WD& ttables,bool copyAllProb){
	int len = context.size();
	if(copyAllProb || node.childs.empty()){
		for(ZMap<WordID,double>::iterator it = node.probs.begin();it != node.probs.end();++it){
			WordVector wv(context);
			reverse(wv.begin(),wv.end());
			ttables[len][wv][it->first] = exp10(it->second);
		}
	}
	if(node.childs.empty()){
		//cerr<<"node.childs.empty()==true node.probs.size()=="<<node.probs.size()<<endl;
		return ;
	}
	for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();iter++){
		context.push_back(iter->first);
		_copyDiscountedProb(iter->second,context,ttables,copyAllProb);
		context.pop_back();
	}
	return ;
}

void LM::copyBOW(WV2D& bow){
	bow.clear();
	WordVector context;
	_copyBOW(_root, context, bow);
}

void LM::_copyBOW(Node& node,WordVector& context,WV2D& bow){
	WordVector wv(context);
	reverse(wv.begin(),wv.end());
	bow[wv] = exp10(node.bow);
	if(node.childs.empty()){
		return ;
	}
	for(ZMap<WordID,Node>::iterator iter=node.childs.begin();iter!=node.childs.end();iter++){
		context.push_back(iter->first);
		_copyBOW(iter->second,context,bow);
		context.pop_back();
	}
	return ;
}
