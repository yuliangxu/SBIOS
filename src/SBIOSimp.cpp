// final version of method2 for imputation
// with an option to update constrained eta
#include <RcppArmadillo.h>
#include "SBIOS_help_fun.h"
#include <time.h>
// [[Rcpp::depends(RcppArmadillo, BH, bigmemory)]]
using namespace Rcpp;
#include <progress.hpp>
#include <progress_bar.hpp>
#include <Rcpp/Benchmark/Timer.h>
#include <bigmemory/BigMatrix.h>
// [[Rcpp::depends(RcppProgress)]]






// [[Rcpp::export]]
arma::mat extract_mask(SEXP mask_fbm) {
  XPtr<BigMatrix> mask_p(mask_fbm);
  arma::mat mask_mat = arma::Mat<double>((double *)mask_p->matrix(), mask_p->nrow(), mask_p->ncol(), false);
  return mask_mat;
}

arma::mat extractBigMatColsRows(const arma::mat& aBigMat, arma::uvec col_idx,
                                arma::uvec row_idx ) {
  return aBigMat(row_idx,col_idx);
}


arma::vec High_to_low_vec(arma::vec& High_vec, int L, Rcpp::List& Phi_Q,
                          Rcpp::List& region_idx, Rcpp::List& L_idx){
  int num_region = region_idx.size();
  arma::colvec Low_vec(L,1);
  for(arma::uword r=0; r<num_region; r++){
    arma::uvec p_idx = region_idx[r];
    arma::uvec L_range = L_idx[r];
    arma::mat Q = Phi_Q[r];
    Low_vec(L_range) = Q.t()*High_vec(p_idx);
  }
  return Low_vec;
  
}

arma::mat High_to_low(const arma::mat& High_mat, int L, Rcpp::List& Phi_Q,
                      Rcpp::List& region_idx, Rcpp::List& L_idx){
  int num_region = region_idx.size();
  int n = High_mat.n_cols;
  arma::mat Low_mat = arma::zeros(L,n);
  for(arma::uword r=0; r<num_region; r++){
    arma::uvec p_idx = region_idx[r];
    arma::uvec L_range = L_idx[r];
    arma::mat Q = Phi_Q[r];
    Low_mat.rows(L_range) = Q.t()*High_mat.rows(p_idx);
  }
  return Low_mat;
  
}






void update_Y_star_at_r(arma::mat& Y_star_b, const arma::vec& Y_imp,
                        const arma::uvec& sub_idx,
                        const arma::mat& Q_t,
                        const List& imp_vec_b_r,
                        const List& imp_p_b_r){
  arma::uword n_indi = sub_idx.n_elem;
  for(arma::uword it =0; it<n_indi; it++){
    arma::uword i = sub_idx(it);
    arma::uvec imp_vec_i = imp_vec_b_r[i];
    arma::uvec imp_p_i = imp_p_b_r[i];
    // Rcout<<"imp_vec_i="<<imp_vec_i<<std::endl;
    // Rcout<<"imp_p_i ="<<imp_p_i<<std::endl;
    if(imp_vec_i.n_elem>0 && imp_p_i.n_elem>0){
      Y_star_b.col(it) += Q_t.cols(imp_p_i)*Y_imp(imp_vec_i);
    }
    
  }
}

void update_Y_star(arma::mat& Y_star, arma::vec& Y_imp,
                   const arma::uword batch_size,
                   const List& region_idx,
                   const List& L_idx,
                   const List& Phi_Q, const List& imp_vec_b,
                   const List& imp_p_b,
                   const int& direction){
  
  arma::uword num_region = Phi_Q.size();
  for(int r=0; r<num_region; r++){
    List imp_p_b_r = imp_p_b[r];
    List imp_vec_b_r = imp_vec_b[r];
    arma::uvec p_idx = region_idx[r];
    arma::uvec L_range = L_idx[r];
    arma::mat Q = Phi_Q[r];
    arma::mat Q_t = Q.t();
    for(int i=0;i<batch_size;i++){
      arma::uvec imp_vec_i = imp_vec_b_r[i];
      arma::uvec imp_p_i = imp_p_b_r[i];
      arma::uvec i_vec(1); i_vec(0)=i;
      if(imp_vec_i.n_elem>0 && imp_p_i.n_elem>0){
        if(direction == 0){
          Y_star(L_range,i_vec) += Q_t.cols(imp_p_i)*Y_imp(imp_vec_i);
        }
        if(direction == 1){
          Y_imp(imp_vec_i) = Q.rows(imp_p_i) *Y_star(L_range,i_vec);
        }
      }
      
      
      
    }
  }
  
}
// [[Rcpp::export]]
void update_Y_imp(const arma::mat& Y_star, arma::vec& Y_imp,
                  arma::vec beta, 
                  arma::rowvec X_b, double sigma_Y,
                  arma::mat gamma, arma::mat X_q,
                  const List& region_idx,
                  const List& L_idx,
                  const List& Phi_Q, const List& imp_vec_b,
                  const List& imp_p_b){
  arma::uword num_region = Phi_Q.size();
  for(int r=0; r<num_region; r++){
    List imp_p_b_r = imp_p_b[r];
    List imp_vec_b_r = imp_vec_b[r];
    arma::uvec p_idx = region_idx[r];
    arma::uvec L_range = L_idx[r];
    arma::mat Q = Phi_Q[r];
    // mat Q_t = Q.t();
    
    for(int i=0;i<X_b.n_elem;i++){
      arma::uvec imp_vec_i = imp_vec_b_r[i];
      arma::uvec imp_p_i = imp_p_b_r[i];
      arma::uvec p_i = p_idx(imp_p_i);
      arma::uvec i_vec(1); i_vec(0)=i;
      if(imp_vec_i.n_elem>0 && imp_p_i.n_elem>0){
        Y_imp(imp_vec_i) = Q.rows(imp_p_i) *Y_star(L_range,i_vec);
        // if(r==8 && i==0){Rcout<<"aaaaa"<<std::endl;}
        Y_imp(imp_vec_i) += beta(p_i) * X_b(i);
        // if(r==8 && i==0){Rcout<<"bbbbb"<<std::endl;}
        Y_imp(imp_vec_i) += gamma.rows(p_i) * X_q.col(i);
        // if(r==8 && i==0){Rcout<<"ccccc"<<std::endl;}
        Y_imp(imp_vec_i) += randn(size(Y_imp(imp_vec_i))) * sigma_Y;
        // if(r==8 && i==0){Rcout<<"ddddd"<<std::endl;}
      }
      
      
      
    }
  }
  
}

// [[Rcpp::export]]
void update_Y_mat(arma::mat& Y_mat, arma::vec& Y_imp,
                  arma::rowvec X_b, 
                  const List& region_idx,const List& imp_vec_b,
                  const List& imp_p_b){
  arma::uword num_region = region_idx.size();
  for(int r=0; r<num_region; r++){
    List imp_p_b_r = imp_p_b[r];
    List imp_vec_b_r = imp_vec_b[r];
    arma::uvec p_idx = region_idx[r];
    
    for(int i=0;i<X_b.n_elem;i++){
      arma::uvec imp_vec_i = imp_vec_b_r[i];
      arma::uvec imp_p_i = imp_p_b_r[i];
      arma::uvec p_i = p_idx(imp_p_i);
      arma::uvec i_vec(1); i_vec(0)=i;
      if(imp_vec_i.n_elem>0 && imp_p_i.n_elem>0){
        Y_mat(p_i,i_vec) = Y_imp(imp_vec_i);
      }
      
      
      
    }
  }
  
}
// Variables including Y:
// XY_term_allsample: Y * X (p by 1)
// XqYstar_term_allsample: Y_star * X_q (L by q)
// XY_eta_term: 
// 
// Need mask related variables:
// Y_imp: list of imputed Y by batch (total_batch)
// mask_imp: list of imputation idx by batch (total_batch)
// 
// Y_star: computed from the zero-imputed Y
// allSGLD version: adjust stepsize, use sgld to update all parameters




// [[Rcpp::export]]
List SBIOSimp(Rcpp::List& data_list, Rcpp::List& basis,  
              Rcpp::List& dimensions, Rcpp::List& imp_idx_list,
              arma::uvec total_imp, 
              Rcpp::List& init_params, Rcpp::List& region_idx, Rcpp::List& L_idx,
              Rcpp::List& batch_idx, 
              double lambda, double prior_p,
              int n_mcmc, int start_saving_imp,
              int start_delta, int subsample_size, double step,
              int begin_eta = 0,
              int seed = 2022,
              int thinning = 1, int burnin = 0,
              double a=1, double b=1,int interval_eta = 10, 
              bool all_sgld = false,
              double a_step = 0.001,
              double b_step = 10,
              double gamma_step = -0.55,
              bool testing = true, bool display_progress = true,
              bool update_individual_effect = false){
  // read all data as file backed matrices
  set_seed(seed);    
  
  Rcpp::List Y_list = data_list["Y_list"];
  Rcpp::List X_list = data_list["X_list"];
  Rcpp::List Y_star_list = data_list["Y_star_list"];
  List imp_list_vec = imp_idx_list["imp_list_vec"];
  List imp_list_p = imp_idx_list["imp_list_p"];
  int total_batch = Y_list.size();
  int n = dimensions["n"];
  int L = dimensions["L"];
  int p = dimensions["p"];
  int q = dimensions["q"];
  int num_region = region_idx.size();
  Rcpp::List Phi_Q = basis["Phi_Q"];
  Rcpp::List Phi_D = basis["Phi_D"];
  arma::colvec D_vec = basis["D_vec"];
  
  
  // read initial parameters 
  arma::colvec theta_beta = init_params["theta_beta"];
  arma::mat theta_eta = init_params["theta_eta"];
  arma::mat theta_eta_temp = theta_eta;
  arma::mat theta_gamma = init_params["theta_gamma"];
  arma::uvec delta = init_params["delta"];
  double sigma_Y = init_params["sigma_Y"], sigma_beta = init_params["sigma_beta"];
  double sigma_gamma = init_params["sigma_gamma"];
  double sigma_eta, sigma_eta2;
  double sigma_Y2 = sigma_Y*sigma_Y, sigma_beta2 = sigma_beta*sigma_beta;
  double sigma_gamma2 = sigma_gamma*sigma_gamma;
  if(update_individual_effect){
    sigma_eta = init_params["sigma_eta"];
    sigma_eta2 = sigma_eta*sigma_eta;
  }
  
  
  
  // precomputing for delta update
  // define pre-compute quantities
  arma::colvec XY_term_allsample = arma::zeros(p,1); // sum_i (Y_i(s))*X_i
  arma::mat XqYstar_term_allsample = arma::zeros(L,q);
  double X2_sum_allsample=0; arma::colvec X2_sum_allsample_q = arma::zeros(q,1);
  
  arma::mat XcXq_sumsq = arma::zeros(q-1,q); // sum_i X[-j]*X_j
  arma::colvec XXq_sumsq = arma::zeros(q,1); // sum_i X*X_j
  
  int saving_imp_counter = 0;
  
  // create a List to store imputed Y
  List Y_imp(total_batch); List Y_imp_mean(total_batch);
  
  // mat G; // (q+1) by n (X,C)^T
  for( int batch_counter=0; batch_counter<total_batch; batch_counter++ ){
    // get subsamples
    SEXP Y_fbm =  Y_list[batch_counter];
    XPtr<BigMatrix> Y_p(Y_fbm);
    arma::mat Y_mat = arma::Mat<double>((double *)Y_p->matrix(), Y_p->nrow(), Y_p->ncol(), false);
    const arma::mat X_batch = X_list[batch_counter];
    
    // if(batch_counter ==0 ){
    //   G = X_batch;
    // }else{
    //   G = join_horiz(G,X_batch);
    // }
    
    arma::rowvec X_b = X_batch.row(0);arma::mat X_q = X_batch.rows(1,q);
    X2_sum_allsample += sum(X_b%X_b);
    
    // Rcout<<"11111: batch ="<<batch_counter<<std::endl;
    SEXP Y_star_fbm =  Y_star_list[batch_counter];
    XPtr<BigMatrix> Y_star_p(Y_star_fbm);
    const arma::mat Y_star_b = arma::Mat<double>((double *)Y_star_p->matrix(), Y_star_p->nrow(), Y_star_p->ncol(), false);
    
    // initialize Y_imp
    arma::vec Y_imp_b = arma::zeros(total_imp(batch_counter),1);
    Y_imp[batch_counter] = Y_imp_b;
    Y_imp_mean[batch_counter] = Y_imp_b;
    
    for(int j=0; j<q; j++){
      X2_sum_allsample_q(j) += sum(X_q.row(j) %X_q.row(j));
      XqYstar_term_allsample.col(j) += Y_star_b * trans(X_q.row(j));
      XXq_sumsq(j) += accu(X_b %X_q.row(j));
      arma::uvec c_j = complement(j, j, q);
      XcXq_sumsq.col(j) += X_q.rows(c_j) * trans(X_q.row(j));
    }
    arma::uvec batch_range = batch_idx[batch_counter];
    arma::colvec XY_term = Y_mat * X_b.t();
    XY_term_allsample += XY_term;
  }
  
  
  // prepare for constrained eta update
  // if(testing){
  //   Rcout<<"size of G"<<G.n_rows<<","<<G.n_cols<<std::endl;
  // }
  // List H_mat = get_H_mat(G);
  
  // prepare pre-computation 
  arma::colvec XY_eta_term = arma::zeros(p,1);
  arma::mat XqYstar_theta_eta_term = arma::zeros(L,q);
  update_XY_eta_term(XY_eta_term,  XqYstar_theta_eta_term,
                     XY_term_allsample, XqYstar_term_allsample,theta_eta,
                     Phi_Q,region_idx,L_idx,batch_idx, X_list);
  // output results
  int total_mcmc = (n_mcmc-burnin)/thinning;
  arma::mat theta_beta_mcmc = arma::zeros(L,total_mcmc);
  if(testing){
    Rcout<<"total_mcmc = "<<total_mcmc<<std::endl;
  }
  arma::colvec sigma_Y2_mcmc = arma::zeros(total_mcmc,1);
  arma::colvec sigma_beta2_mcmc = arma::zeros(total_mcmc,1);
  arma::colvec sigma_eta2_mcmc = arma::zeros(total_mcmc,1);
  arma::colvec sigma_gamma2_mcmc = arma::zeros(total_mcmc,1);
  arma::cube theta_gamma_mcmc = arma::zeros(L,q,total_mcmc);
  arma::colvec sgld_step_mcmc = arma::zeros(n_mcmc,1);
  
  // initialize parameters
  arma::mat gamma = arma::zeros(p,q);
  arma::colvec beta = arma::zeros(p,1);
  
  // for testing purpose
  Rcpp::List cur_vars;
  Rcpp::List last_vars;
  
  
  // declare trace 
  // arma::colvec step_allregion = step * arma::ones(num_region,1);
  arma::colvec gradU_allregion = arma::zeros(L,1);
  arma::colvec logLL_mcmc = arma::zeros(total_mcmc,1);
  arma::umat delta_mcmc = arma::umat(p,total_mcmc);
  arma::mat theta_beta_cov_inv;
  arma::mat theta_beta_cov;
  arma::mat theta_eta_res(L,n);
  arma::colvec beta_star(L,1); // for computing X_i* D_delta * theta_beta
  double sigma_Y2_diff=0;
  // begin iterations
  Progress prog(n_mcmc, display_progress);
  Rcpp::Timer timer;  
  timer.step("start of iteration"); 
  int batch_counter = 0;
  arma::mat time_segment = arma::zeros(n_mcmc,5);
  
  for(arma::uword iter=0; iter<n_mcmc; iter++){
    prog.increment(); 
    
    
    
    clock_t t0;t0 = clock();
    
    // iterating through region
    for(arma::uword r=0; r<num_region; r++){
      arma::mat Q = Phi_Q[r];
      arma::mat Q_t = Q.t();
      arma::colvec D = Phi_D[r];
      int Lr = D.n_elem;
      arma::uvec p_idx = region_idx[r];
      arma::uvec L_range = L_idx[r];
      arma::uvec delta_r = delta(p_idx);
      arma::mat Q_delta = Q.rows(arma::find(delta_r>0));
      arma::mat D_delta = Q_delta.t() * Q_delta;
      arma::mat D_delta2 = D_delta * D_delta;
      
      
      
      // begin SGLD
      arma::mat X_batch = X_list[batch_counter];
      arma::rowvec X_b = X_batch.row(0);arma::mat X_q = X_batch.rows(1,q);
      arma::uword batch_size_b = X_b.n_elem;
      arma::uvec batch_range = batch_idx[batch_counter];
      arma::uvec sub_idx = arma::randperm( batch_size_b, subsample_size );//??must be integer
      arma::uvec batch_idx_sub = batch_range(sub_idx);
      
      // update Y_star with imputed values
      arma::vec Y_imp_b = Y_imp[batch_counter];
      SEXP Y_star_fbm =  Y_star_list[batch_counter];
      XPtr<BigMatrix> Y_star_p(Y_star_fbm);
      const arma::mat Y_star_sub_L_range_data = extractBigMatColsRows(
        arma::Mat<double>((double *)Y_star_p->matrix(), Y_star_p->nrow(), Y_star_p->ncol(), false),
        sub_idx, L_range
      );
      arma::mat Y_star_sub_L_range = Y_star_sub_L_range_data;
      List imp_p_b = imp_list_p[batch_counter];
      List imp_vec_b = imp_list_vec[batch_counter];
      List imp_p_b_r = imp_p_b[r];
      List imp_vec_b_r = imp_vec_b[r];
      update_Y_star_at_r(Y_star_sub_L_range, Y_imp_b, sub_idx,
                         Q_t, imp_vec_b_r, imp_p_b_r);
      double X2_sum_b = sum(X_b(sub_idx)%X_b(sub_idx));
      arma::colvec XY_star_L_range = ( Y_star_sub_L_range - theta_eta(L_range, batch_idx_sub) -
        theta_gamma.rows(L_range)*X_q.cols(sub_idx) )* X_b(sub_idx);
      
      // update theta_beta -- old version
      theta_beta_cov_inv = arma::diagmat(1/D/sigma_beta2) +
        X2_sum_b/sigma_Y2*D_delta2;
      theta_beta_cov = arma::inv_sympd(theta_beta_cov_inv);
      arma::colvec theta_beta_mean = theta_beta_cov * (D_delta * XY_star_L_range)/sigma_Y2;
      arma::colvec gradU = theta_beta_cov_inv * (theta_beta(L_range) - theta_beta_mean);
      gradU *= n/subsample_size;
      theta_beta(L_range) += -step/2*gradU + sqrt(step)*arma::randn(Lr,1);
      gradU_allregion(L_range) = gradU;
      beta(p_idx) = delta(p_idx) %( Q * theta_beta(L_range) );
      // beta(p_idx) = Q * theta_beta(L_range);
      beta_star(L_range) = D_delta * theta_beta(L_range);
      
      // // new version
      // arma::mat theta_beta_cov_post = arma::inv_sympd(X2_sum_b/sigma_Y2*D_delta2);
      // arma::colvec theta_beta_mean_post = theta_beta_cov_post * (D_delta * XY_star_L_range)/sigma_Y2;
      // arma::colvec gradU_prior = arma::diagmat(1/D/sigma_beta2)*theta_beta(L_range);
      // arma::colvec gradU_log = (X2_sum_b/sigma_Y2*D_delta2) *theta_beta(L_range) - (D_delta * XY_star_L_range)/sigma_Y2;
      // arma::colvec gradU = gradU_prior + (n/subsample_size)*gradU_log;
      // 
      // theta_beta(L_range) += -step/2*gradU + sqrt(step)*arma::randn(Lr,1);
      // gradU_allregion(L_range) = gradU;
      // beta(p_idx) = delta(p_idx) %( Q * theta_beta(L_range) );
      // beta_star(L_range) = D_delta * theta_beta(L_range);
      
      // if(testing){
      //   Rcout<<"region = "<<r<<std::endl;
      //   Rcout<<"max(theta_beta(L_range))="<<max(abs(theta_beta(L_range)))<<std::endl;
      // }
      
      
    }// end of one region
    // update batch counter
    batch_counter++ ;
    if(batch_counter == total_batch){batch_counter = 0;}
    
    t0 = clock() - t0;
    double sys_t0 = ((double)t0)/CLOCKS_PER_SEC;
    time_segment(iter,0) = sys_t0;
    
    // gradU_mcmc.col(iter) = gradU_allregion;
    // update theta_gamma
    t0 = clock();
    arma::mat mean_gamma = arma::mat(L,q);
    for(int j =0; j<q; j++){
      arma::colvec Sigma_gamma_j = 1/(1/D_vec/sigma_gamma2 + 1/sigma_Y2*X2_sum_allsample_q(j));
      arma::uvec c_j = complement(j, j, q);
      arma::colvec mean_gamma_j = XqYstar_theta_eta_term.col(j) -
        theta_gamma.cols(c_j) * XcXq_sumsq.col(j) - XXq_sumsq(j)*beta_star;
      mean_gamma_j %= Sigma_gamma_j/sigma_Y2;
      mean_gamma.col(j) = mean_gamma_j;
      
      if(all_sgld){
        arma::colvec theta_gamma_j_inc = -step/2/Sigma_gamma_j%(theta_gamma.col(j) - mean_gamma_j) + 
          arma::randn(L,1)*sqrt(step);
        theta_gamma.col(j) += theta_gamma_j_inc;
      }else{
        theta_gamma.col(j) = arma::randn(L,1)%sqrt(Sigma_gamma_j) +  mean_gamma_j;
      }
    }
    
    
    // theta_gamma_mcmc.slice(iter) = theta_gamma;
    gamma = Low_to_high(theta_gamma, p,Phi_Q, region_idx, L_idx);
    t0 = clock() - t0;
    sys_t0 = ((double)t0)/CLOCKS_PER_SEC;
    time_segment(iter,1) = sys_t0;
    
    // update delta
    t0 = clock();
    arma::colvec gamma_XXq_term = gamma*XXq_sumsq; // p x 1
    if(iter >= start_delta){
      arma::colvec delta_prior = arma::ones(p,1);
      arma::uvec sig_idx = arma::find(abs(beta)>lambda);
      arma::uvec nonsig_idx = arma::find(abs(beta)<=lambda);
      delta_prior(sig_idx) *= prior_p;
      delta_prior(nonsig_idx) *= 1-prior_p;
      
      arma::colvec logL_all = arma::zeros(p);
      for(arma::uword j=0;j<p;j++){
        double logL_j = -0.5/sigma_Y2*(beta(j)*beta(j)*X2_sum_allsample -
                                       2*(XY_eta_term(j) - gamma_XXq_term(j))*beta(j));
        // deal with p1=inf
        double p1 = exp(logL_j); p1*=delta_prior(j)/(1-delta_prior(j));
        double p0 = 1/(p1+1); p1=1-p0;
        if(arma::randu()<p1){
          delta(j) = 1;
        }else{
          delta(j) = 0;
        }
        logL_all(j) = logL_j;
      }
      
      // delta_mcmc.col(iter) = delta;
    }
    t0 = clock() - t0;
    sys_t0 = ((double)t0)/CLOCKS_PER_SEC;
    time_segment(iter,2) = sys_t0;
    
    // update theta_eta  and sigma_eta, sigma_Y
    // update XY_term_allsample, XqYstar_term_allsample
    t0 = clock();
    double logLL = 0;
    double b_Y=0; // for sigma_Y2 update
    // double b_eta = 0;// for updating b_eta
    XqYstar_term_allsample.zeros();
    XY_term_allsample.zeros();
    double max_Y_imp=0;
    if(iter%interval_eta==0){
      for(arma::uword b=0; b<total_batch;b++){
        arma::uvec batch_range = batch_idx[b];
        arma::mat X_batch = X_list[b];
        arma::rowvec X_b = X_batch.row(0);arma::mat X_q = X_batch.rows(1,q);
        
        arma::vec Y_imp_b = Y_imp[b];
        SEXP Y_star_fbm =  Y_star_list[b];
        XPtr<BigMatrix> Y_star_p(Y_star_fbm);
        const arma::mat Y_star_b_data = arma::Mat<double>((double *)Y_star_p->matrix(), Y_star_p->nrow(), Y_star_p->ncol(), false);
        arma::mat Y_star_b = Y_star_b_data;
        List imp_p_b = imp_list_p[b];
        List imp_vec_b = imp_list_vec[b];
        update_Y_star(Y_star_b,Y_imp_b,
                      X_b.n_elem,
                      region_idx,
                      L_idx,
                      Phi_Q,imp_vec_b,
                      imp_p_b,0);
        
        
        // update sigma
        
        
        arma::mat temp_mat_b = Y_star_b - beta_star*X_b - theta_gamma*X_q;
        if(update_individual_effect && iter > begin_eta){
          arma::mat theta_eta_b = theta_eta.cols(batch_range);
          theta_eta_res.cols(batch_range) = Y_star_b -  beta_star * X_b - theta_gamma*X_q;
          temp_mat_b += - theta_eta_b;
          double temp_mat_b_norm = norm(temp_mat_b,"fro");
          b_Y += temp_mat_b_norm*temp_mat_b_norm;
          // theta_eta_b.each_col() %= 1/sqrt(D_vec);
          // double eta_norm = norm(theta_eta_b,"fro");
          // b_eta += eta_norm*eta_norm;
          // update logLL
          logLL += norm(theta_eta_res.cols(batch_range) - theta_eta.cols(batch_range),"fro");
        }else{
          double temp_mat_b_norm = norm(temp_mat_b,"fro");
          b_Y += temp_mat_b_norm*temp_mat_b_norm;
          logLL += norm(temp_mat_b,"fro");
        }
        
        
        // update Y_star_b from the current model
        const arma::mat Y_star_b_pred = theta_eta.cols(batch_range) ;
        // gives very large values in the imputation
        update_Y_imp(Y_star_b_pred,Y_imp_b,
                     beta%delta, X_b, sqrt(sigma_Y2),
                     gamma, X_q,
                     region_idx,
                     L_idx,Phi_Q,
                     imp_vec_b,imp_p_b);
        Y_imp[b] = Y_imp_b;
        int max_Y_imp_b = max(abs(Y_imp_b));
        if(max_Y_imp_b>max_Y_imp){max_Y_imp = max_Y_imp_b;}
        if(iter > start_saving_imp){
          if(b==0){
            saving_imp_counter += 1;
          }
          
          arma::vec Y_imp_b_mean = Y_imp_mean[b];
          Y_imp_b_mean += Y_imp_b;
          Y_imp_mean[b] = Y_imp_b_mean;
          // if(b==0){
          //   Y_imp_b1.col(iter - start_saving_imp-1) = Y_imp_b;
          // }
          
        }
        SEXP Y_fbm =  Y_list[b];
        XPtr<BigMatrix> Y_p(Y_fbm);
        arma::mat Y_mat = arma::Mat<double>((double *)Y_p->matrix(), Y_p->nrow(), Y_p->ncol(), false);
        // uvec mask_b = mask_imp[b];
        update_Y_mat(Y_mat,Y_imp_b,X_b, 
                     region_idx,
                     imp_vec_b,imp_p_b);
        
        
        // update XqYstar_term_allsample,XY_term_allsample
        for(int j=0; j<q; j++){
          XqYstar_term_allsample.col(j) += Y_star_b * trans(X_q.row(j));
        }
        
        // pre-compute
        arma::colvec XY_term = Y_mat * X_b.t();
        XY_term_allsample += XY_term;
      }
      logLL*= (-0.5/sigma_Y2);
      logLL+= -0.5*L*n*log(sigma_Y2*2*3.14159);
      
      if(update_individual_effect && iter > begin_eta){
        arma::colvec eta_sigma_pos = 1/(1/sigma_Y2 + 1/sigma_eta2/D_vec);
        
        if(all_sgld){
          theta_eta_res.each_col() %= eta_sigma_pos/sigma_Y2;
          arma::mat grad_theta_eta = theta_eta - theta_eta_res;
          grad_theta_eta.each_col() %= 1/eta_sigma_pos;
          // double step_eta = step_allregion(r)/eta_step;
          arma::mat theta_eta_inc = -step/2*grad_theta_eta + randn(size(grad_theta_eta))*sqrt(step);
          theta_eta += theta_eta_inc;
          
        }else{
          theta_eta_res.each_col() %= eta_sigma_pos/sigma_Y2;
          theta_eta += theta_eta_res;
          // theta_eta = hyperplane_MVN_multiple(G,H_mat,eta_sigma_pos,theta_eta_res.t());
        }
        
        arma::mat theta_eta_b = theta_eta;
        theta_eta_b.each_col() %= 1/sqrt(D_vec);
        double b_eta = dot( theta_eta_b, theta_eta_b );
        sigma_eta2 = 1/arma::randg( arma::distr_param(a + n*L/2, 1/(b+b_eta/2) ) );
        
      }
      
      
      
      double sigma_Y2_old = sigma_Y2;
      sigma_Y2 = 1/arma::randg( arma::distr_param(a + n*L/2, 1/(b+b_Y/2)) );
      sigma_Y2_diff = sigma_Y2 - sigma_Y2_old;
      
      double b_beta = b+dot(theta_beta,theta_beta/D_vec)/2;
      // b_beta *= n/subsample_size;
      sigma_beta2 = 1/arma::randg( arma::distr_param(a + L/2, 1/b_beta) );
      // sigma_beta2 = 1/arma::randg( arma::distr_param(a + L/2,
      //                                                1/(b+dot(theta_beta,theta_beta/D_vec)/2)) );
      sigma_gamma2 = 1/arma::randg( arma::distr_param(a + L*q/2,
                                                      1/(b+  accu(theta_gamma%(theta_gamma.each_col()/D_vec))/2)) );
      // Rcout<<"iter = "<<iter<<"; sigma_Y2="<<sigma_Y2<<"; sigma_beta2="<<sigma_beta2<<std::endl;
      
      
      // // update XY_eta_term
      update_XY_eta_term(XY_eta_term,  XqYstar_theta_eta_term,
                         XY_term_allsample, XqYstar_term_allsample,theta_eta,
                         Phi_Q,region_idx,L_idx,batch_idx, X_list);
      
    }
    t0 = clock() - t0;
    sys_t0 = ((double)t0)/CLOCKS_PER_SEC;
    time_segment(iter,3) = sys_t0;
    
    
    // theta_beta_mcmc.col(iter) = theta_beta;
    if(iter > burnin){
      if((iter-burnin)%thinning == 0){
        int mcmc_iter = (iter-burnin)/thinning-1;
        sigma_eta2_mcmc(mcmc_iter) = sigma_eta2;
        theta_beta_mcmc.col(mcmc_iter) = theta_beta;
        theta_gamma_mcmc.slice(mcmc_iter) = theta_gamma;
        delta_mcmc.col(mcmc_iter) = delta;
        logLL_mcmc(mcmc_iter) = logLL;
        sigma_beta2_mcmc(mcmc_iter) = sigma_beta2;
        sigma_Y2_mcmc(mcmc_iter) = sigma_Y2;
        sigma_gamma2_mcmc(mcmc_iter) = sigma_gamma2;
      }
    }
    
    
    if( testing ){
      Rcout<<"iter = "<<iter<<std::endl;
      Rcout<<"max(theta_beta)="<<max(abs(theta_beta))<<std::endl;
      cur_vars = Rcpp::List::create(Named("theta_beta") = theta_beta,
                                    Named("theta_gamma") = theta_gamma,
                                    Named("delta") = delta,
                                    Named("sigma_Y2") = sigma_Y2,
                                    Named("Y_imp") = Y_imp);
    }
    
    
    // tune step size
    step = a_step*pow((b_step+iter),gamma_step);
    sgld_step_mcmc(iter) = step;
    
  }//end of one iteration
  
  // get Y_imp_mean
  for(arma::uword b=0; b<total_batch;b++){
    arma::vec Y_imp_b_mean = Y_imp_mean[b];
    Y_imp_b_mean /= saving_imp_counter;
    Y_imp_mean[b] = Y_imp_b_mean;
  }
  // return all results
  return Rcpp::List::create(Rcpp::Named("theta_beta_mcmc") = theta_beta_mcmc,
                            Rcpp::Named("theta_gamma_mcmc") = theta_gamma_mcmc,
                            Rcpp::Named("theta_eta") = theta_eta,
                            Rcpp::Named("sigma_beta2_mcmc") =  sigma_beta2_mcmc,
                            Rcpp::Named("sigma_gamma2_mcmc") =  sigma_gamma2_mcmc,
                            Rcpp::Named("sigma_Y2_mcmc") =  sigma_Y2_mcmc,
                            Rcpp::Named("sigma_eta2_mcmc") =  sigma_eta2_mcmc,
                            Rcpp::Named("logLL_mcmc") = logLL_mcmc,
                            Rcpp::Named("sgld_step_mcmc") =  sgld_step_mcmc,
                            Rcpp::Named("time_segment") = time_segment,
                            Rcpp::Named("Y_imp") = Y_imp,
                            Rcpp::Named("Y_imp_mean") = Y_imp_mean,
                            Rcpp::Named("delta_mcmc") =  delta_mcmc);
}


