// Logistic regression implementation using stochastic gradient descent
// (c) Tim Nugent
// timnugent@gmail.com

#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sstream>
#include <algorithm>
#include <map>

using namespace std;

vector<string> &split(const string &s, char delim, vector<std::string> &elems) {
    stringstream ss(s);
    string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

void usage(const char* prog){

   cout << "Read training data then classify test data using logistic regression:\nUsage:\n" << prog << " [options] training_data test_data" << endl << endl;
   cout << "Options:" << endl;   
   cout << "-s <int>   Shuffle dataset after each iteration. default 1" << endl;    
   cout << "-i <int>   Maximum iterations. default 50000" << endl;   
   cout << "-e <float> Convergence rate. default 0.005" << endl;    
   cout << "-a <float> Learning rate. default 0.001" << endl; 
   cout << "-m <file>  Write weights to model file" << endl;    
   cout << "-v         Verbose." << endl << endl;      
}

double vecnorm(map<int,double>& w1, map<int,double>& w2) {

    double sum = 0;
    for(auto it = w1.begin(); it != w1.end(); it++){
        double minus = w1[it->first] - w2[it->first];
        double r = minus * minus;
        sum += r;
    }
    return sqrt(sum);

}

double sigmoid(double x){
    return 1.0/(1.0 + exp(-x));
}

double classify(map<int,double>& features, map<int,double>& weights, int verbose = 0){

    double logit = 0.0;
    for(auto it = features.begin(); it != features.end(); it++){
        if(it->first != 0){
            logit += it->second * weights[it->first];
        }
    }
    return sigmoid(logit);
}

int main(int argc, const char* argv[]){

    // Learning rate
    double alpha = 0.001;
    // Max iterations
    unsigned int maxit = 50000;
    // Shuffle data set
    int shuf = 1;
    // Convergence threshold
    double eps = 0.005;
    // Verbose
    int verbose = 0;
    // Model output file
    string model = "";

    if(argc < 3){
        usage(argv[0]);
        return(1);
    }else{
        cout << "# called with:       ";
        for(int i = 0; i < argc; i++){
            cout << argv[i] << " ";
            if(string(argv[i]) == "-a" && i < argc-1){
                alpha = atof(argv[i+1]);
            }
            if(string(argv[i]) == "-m" && i < argc-1){
                model = string(argv[i+1]);
            }
            if(string(argv[i]) == "-s" && i < argc-1){
                shuf = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-i" && i < argc-1){
                maxit = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-e" && i < argc-1){
                eps = atof(argv[i+1]);
            }
            if(string(argv[i]) == "-v"){
                verbose = 1;
            }
            if(string(argv[i]) == "-h"){
                usage(argv[0]);
                return(1);
            }
        }
        cout << endl;
    }

    cout << "# learning rate:     " << alpha << endl;
    cout << "# convergence rate:  " << eps << endl;
    cout << "# max. iterations:   " << maxit << endl;    
    cout << "# training data:     " << argv[argc-2] << endl;
    cout << "# test data:         " << argv[argc-1] << endl;
    if(model.length()) cout << "# model output:      " << model << endl;

    vector<map<int,double> > data;
    map<int,double> weights;

    ifstream fin(argv[argc-2]);
    fin.ignore();
    string line;
    while (getline(fin, line)){
        if(line.length()){
            if(line[0] != '#' && line[0] != ' '){
                vector<string> tokens = split(line,' ');
                map<int,double> example;
                if(atoi(tokens[0].c_str()) == 1){
                    example[0] = 1;
                }else{
                    example[0] = 0;
                }
                for(unsigned int i = 1; i < tokens.size(); i++){
                    vector<string> feat_val = split(tokens[i],':');
                    example[atoi(feat_val[0].c_str())] = atof(feat_val[1].c_str());
                    weights[atoi(feat_val[0].c_str())] = 0.0;
                }
                data.push_back(example);
                //if(verbose) cout << "read example " << data.size() << " - found " << example.size()-1 << " features." << endl; 
            }    
        }
    }
    fin.close();

    cout << "# training examples: " << data.size() << endl;
    cout << "# features:          " << weights.size() << endl;

    double norm = 1.0;
    unsigned int n = 0;
    unsigned int correct = 0;
    unsigned int total = 0;
    random_device rd;
    mt19937 g(rd());
    vector<int> indicies(data.size());
    iota(indicies.begin(),indicies.end(),0);

    cout << "# stochastic gradient descent:" << endl;
    while(norm > eps){
    //for(unsigned int n = 1; n <= maxit; n++){

        map<int,double> old_weights(weights);
        if(shuf) shuffle(indicies.begin(),indicies.end(),g);

        for (unsigned int i = 0; i < data.size(); i++){
            int label = data[indicies[i]][0];
            double predicted = classify(data[indicies[i]],weights);
            for(auto it = data[indicies[i]].begin(); it != data[indicies[i]].end(); it++){
                if(it->first != 0){
                    weights[it->first] += alpha * (label - predicted) * it->second;
                }
            }
        }
        norm = vecnorm(weights,old_weights);
        if(n && n % 100 == 0){       
            printf("# convergence: %1.4f iterations: %i\n",norm,n);     
            /*
            for(auto it = weights.begin(); it != weights.end(); it++){
                if(verbose) cout << it->first << ":" << it->second << endl;
            }
            */
        }
        n++;
        if(n > maxit){
            break;
        }               
    }

    if(model.length()){
        ofstream outfile;
        outfile.open(model.c_str());  
        for(auto it = weights.begin(); it != weights.end(); it++){
            outfile << it->first << " " << it->second << endl;
        }
        outfile.close();
        cout << "# written weights to file " << model << endl;
    }

    /*
    if(verbose) cout << "# final weights:" << endl;
    for(auto it = weights.begin(); it != weights.end(); it++){
        if(verbose) cout << it->first << ":" << it->second << endl;
    }
    if(verbose) cout << endl;  
    */

    cout << "# classifying:" << endl;
    fin.open(argv[argc-1]);
    while (getline(fin, line)){
        if(line.length()){
            if(line[0] != '#' && line[0] != ' '){
                vector<string> tokens = split(line,' ');
                map<int,double> example;
                int label = atoi(tokens[0].c_str());
                for(unsigned int i = 1; i < tokens.size(); i++){
                    vector<string> feat_val = split(tokens[i],':');
                    example[atoi(feat_val[0].c_str())] = atof(feat_val[1].c_str());
                }
                double predicted = classify(example,weights,0);
                if(verbose){
                    if(label > 0){
                        printf("label: +%i : prediction: %1.3f",label,predicted);
                    }else{
                        printf("label: %i : prediction: %1.3f",label,predicted);
                    }
                }
                if(((label == -1 || label == 0) && predicted < 0.5) || (label == 1 && predicted >= 0.5)){
                    if(verbose) cout << "\tcorrect" << endl;
                    correct++;
                }else{
                    if(verbose) cout << "\tincorrect" << endl;
                }
                total++;
            }    
        }
    }
    fin.close();

    printf ("# accuracy: %3.2f %% (%i/%i)\n", (100*(double)correct/total),correct,total);

    return(0);

}
