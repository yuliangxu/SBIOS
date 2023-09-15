// Using the new derivations
#include <RcppArmadillo.h>
#include "SBIOS_help_fun.h"
#include <time.h>
// [[Rcpp::depends(RcppArmadillo, BH, bigmemory)]]
using namespace Rcpp;
#include <progress.hpp>
#include <progress_bar.hpp>
#include <Rcpp/Benchmark/Timer.h>
#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>
// [[Rcpp::depends(RcppProgress)]]









// [[Rcpp::export]]
List SBIOS0(Rcpp::List& data_list, Rcpp::List& basis,  
            Rcpp::List& theta_eta_path,  
                                Rcpp::List& dimensions,
                                Rcpp::List& init_params, Rcpp::List& region_idx, Rcpp::List& L_idx,
                                Rcpp::List& batch_idx, double lambda, double prior_p,
                                int n_mcmc, int start_delta, int subsample_size, double step,
                                int burnin=0, int thinning = 10,
                                double a=1, double b=1,int interval_eta = 10, 
                                int start_eta = 10, int start_saving_eta = 10,
                                int stop_tuning_stepsize = 1e5,
                                bool all_sgld = 0,
                                int sgld_freq = 50,
                                double a_step = 0.001,
                                double b_step = 10,
                                double gamma_step = -0.55,
                                bool update_individual_effect = 1,
                                bool testing = 0, bool display_progress = true){
  // read all data as file backed matrices
  Rcpp::List Y_list = data_list["Y_list"];
  Rcpp::List X_list = data_list["X_list"];
  Rcpp::List Y_star_list = data_list["Y_star_list"];
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
  arma::mat theta_gamma = init_params["theta_gamma"];
  arma::uvec delta = init_params["delta"];
  double sigma_Y = init_params["sigma_Y"], sigma_beta = init_params["sigma_beta"],
                                                                   sigma_gamma = init_params["sigma_gamma"];
  double sigma_eta = init_params["sigma_eta"];
  double sigma_Y2 = sigma_Y*sigma_Y, sigma_beta2 = sigma_beta*sigma_beta, sigma_eta2 = sigma_eta*sigma_eta;
  double sigma_gamma2 = sigma_gamma*sigma_gamma;
  
  arma::colvec XY_term_allsample = arma::zeros(p,1); // sum_i (Y_i(s))*X_i
  arma::mat XqYstar_term_allsample = arma::zeros(L,q);
  double X2_sum_allsample=0; arma::colvec X2_sum_allsample_q = arma::zeros(q,1);
  
  arma::mat XcXq_sumsq = arma::zeros(q-1,q); // sum_i X[-j]*X_j
  arma::colvec XXq_sumsq = arma::zeros(q,1); // sum_i X*X_j
  
  
  // mat G; // (q+1) by n (X,C)^T
  for( int batch_counter=0; batch_counter<total_batch; batch_counter++ ){
    // get subsamples
    SEXP Y_fbm =  Y_list[batch_counter];
    XPtr<BigMatrix> Y_p(Y_fbm);
    const arma::mat Y_mat = arma::Mat<double>((double *)Y_p->matrix(), Y_p->nrow(), Y_p->ncol(), false);
    const arma::mat X_batch = X_list[batch_counter];
    
    
    arma::rowvec X_b = X_batch.row(0);arma::mat X_q = X_batch.rows(1,q);
    X2_sum_allsample += sum(X_b%X_b);
    
    SEXP Y_star_fbm =  Y_star_list[batch_counter];
    XPtr<BigMatrix> Y_star_p(Y_star_fbm);
    const arma::mat Y_star_b = arma::Mat<double>((double *)Y_star_p->matrix(), Y_star_p->nrow(), Y_star_p->ncol(), false);
    for(int j=0; j<q; j++){
      X2_sum_allsample_q(j) += sum(X_q.row(j) %X_q.row(j));
      XqYstar_term_allsample.col(j) += Y_star_b * trans(X_q.row(j));
      XXq_sumsq(j) += accu(X_b %X_q.row(j));
      arma::uvec c_j = complement(j, j, q);
      XcXq_sumsq.col(j) += X_q.rows(c_j) * trans(X_q.row(j));
    }
    // pre-compute
    arma::uvec batch_range = batch_idx[batch_counter];
    arma::colvec XY_term = Y_mat * X_b.t();
    XY_term_allsample += XY_term;
  }
  
  if(testing){
    return List::create(Named("total_batch")  = total_batch,
                        Named("X2_sum_allsample")  = X2_sum_allsample,
                        Named("XcXq_sumsq")  = XcXq_sumsq,
                        Named("XXq_sumsq")  = XXq_sumsq);
  }
  
  arma::colvec XY_eta_term = arma::zeros(p,1);
  arma::mat XqYstar_theta_eta_term = arma::zeros(L,q);
  update_XY_eta_term_memsave(XY_eta_term,  XqYstar_theta_eta_term,
                     XY_term_allsample, XqYstar_term_allsample,theta_eta_path,
                     Phi_Q,region_idx,L_idx,batch_idx, X_list);
  // output results
  int total_mcmc = (n_mcmc-burnin)/thinning;
  arma::mat theta_beta_mcmc = arma::zeros(L,total_mcmc);
  // arma::mat gradU_mcmc = arma::zeros(L,total_mcmc);
  arma::colvec sigma_Y2_mcmc = arma::zeros(total_mcmc,1);
  arma::colvec sigma_beta2_mcmc = arma::zeros(total_mcmc,1);
  arma::colvec sigma_eta2_mcmc = arma::zeros(total_mcmc,1);
  arma::colvec sigma_gamma2_mcmc = arma::zeros(total_mcmc,1);
  arma::cube theta_gamma_mcmc = arma::zeros(L,q,total_mcmc);
  arma::colvec logLL_mcmc = arma::zeros(total_mcmc,1);
  arma::umat delta_mcmc = arma::umat(p,total_mcmc);
  arma::colvec sgld_step_mcmc = arma::zeros(n_mcmc,1);
  arma::colvec logLL_prior_mcmc = arma::zeros(total_mcmc,1);
  arma::colvec logLL_prior_theta_beta_mcmc = arma::zeros(total_mcmc,1);
  arma::mat delta_logP_mcmc = arma::mat(p,total_mcmc);
  arma::colvec logL_all = arma::zeros(p);
  
  // to be deleted 
  // arma::colvec beta = init_params["beta"];
  // arma::mat gamma = init_params["gamma"];
  arma::mat gamma = arma::zeros(p,q);
  arma::colvec beta = arma::zeros(p,1);
  
  // declare
  arma::colvec step_allregion = step * arma::ones(num_region,1);
  arma::colvec gradU_allregion = arma::zeros(L,1);
  arma::mat theta_beta_cov_inv;
  arma::mat theta_beta_cov;
  // arma::mat theta_eta_res(L,n);
  arma::colvec beta_star(L,1); // for computing X_i* D_delta * theta_beta
  
  // begin iterations
  Progress prog(n_mcmc, display_progress);
  Rcpp::Timer timer;  
  timer.step("start of iteration"); 
  int batch_counter = 0;
  arma::mat time_segment = arma::zeros(n_mcmc,5);
  double logLL=0;
  
  bool if_update_eta=0;
  double save_eta_count = 0;
  int sgld_step_counter = 0;
  
  for(arma::uword iter=0; iter<n_mcmc; iter++){
    prog.increment(); 
    double log_prior = 0;
    // if(iter==stop_burnin){
    //   timer.step("stop of burnin");        // record the starting point
    // }
    
    if(update_individual_effect && iter > start_eta ){
      if_update_eta = 1;
    }
    // Rcout<<"iter = "<<iter<<";aaaaa"<<std::endl;
    clock_t t0;t0 = clock();
    
    // iterating through region
    for(arma::uword r=0; r<num_region; r++){
      arma::mat Q = Phi_Q[r];
      arma::colvec D = Phi_D[r];
      int Lr = D.n_elem;
      int pr = Q.n_rows;
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
      
      for(int sgld_iter = 0; sgld_iter<sgld_freq; sgld_iter++){
        arma::uvec sub_idx = arma::randperm( batch_size_b, subsample_size );//??must be integer
        // arma::uvec batch_idx_sub = batch_range(sub_idx);
        // SEXP Y_fbm =  Y_list[batch_counter];
        // XPtr<BigMatrix> Y_p(Y_fbm);
        SEXP Y_star_fbm =  Y_star_list[batch_counter];
        XPtr<BigMatrix> Y_star_p(Y_star_fbm);
        const arma::mat Y_star_sub = extractBigMatCols(
          arma::Mat<double>((double *)Y_star_p->matrix(), Y_star_p->nrow(), Y_star_p->ncol(), false),
          sub_idx
        );
        
        double X2_sum_b = sum(X_b(sub_idx)%X_b(sub_idx));
        SEXP theta_eta_b_address =  theta_eta_path[b];
        XPtr<BigMatrix> theta_eta_b_pointer(theta_eta_b_address);
        const arma::mat theta_eta_b = arma::Mat<double>((double *)theta_eta_b_pointer->matrix(), theta_eta_b_pointer->nrow(), theta_eta_b_pointer->ncol(), false);
        // Rcout<<"when updating beta, theta_eta_b(0,0)="<<theta_eta_b(0,0)<<std::endl;
        
        arma::colvec XY_star = ( Y_star_sub - theta_eta_b.cols(sub_idx) - 
          theta_gamma*X_q.cols(sub_idx) )* X_b(sub_idx);
        
        // update beta old version
      //   theta_beta_cov_inv = arma::diagmat(1/D/sigma_beta2) + 
      //     X2_sum_b/sigma_Y2*D_delta2;
      //   if(!theta_beta_cov_inv.is_sympd()){
      //     Rcout<<"theta_beta_cov_inv is not sympd"<<std::endl;
      //     Rcout<<"iter = "<<iter<<"; r ="<<r<<"; batch = "<<batch_counter<<std::endl;
      //     Rcout<<"sigma_beta2 = "<<sigma_beta2<<"; sigma_Y2="<<sigma_Y2<<std::endl;
      //     return List::create(Named("sigma_beta2") = sigma_beta2,
      //                         Named("sigma_Y2") = sigma_Y2,
      //                         Named("X2_sum_b") = X2_sum_b,
      //                         Named("D_delta2") = D_delta2,
      //                         Named("delta") = delta);
      //   }
      //   theta_beta_cov = arma::inv_sympd(theta_beta_cov_inv);
      //   arma::colvec theta_beta_mean = theta_beta_cov * (D_delta * XY_star(L_range))/sigma_Y2;// change this
      //   arma::colvec gradU = theta_beta_cov_inv * (theta_beta(L_range) - theta_beta_mean);
      //   gradU *= n/subsample_size;
      //   theta_beta(L_range) += -step/2*gradU + sqrt(step)*arma::randn(Lr,1);
      //   gradU_allregion(L_range) = gradU;
      //   sgld_step_counter ++;
      //   
      //   beta(p_idx) = delta(p_idx) %( Q * theta_beta(L_range) );
      //   beta_star(L_range) = D_delta * theta_beta(L_range);
      
      // // uodate beta new
      // arma::mat theta_beta_cov_post = arma::inv_sympd(X2_sum_b/sigma_Y2*D_delta2);
      // arma::colvec theta_beta_mean_post = theta_beta_cov_post * (D_delta *XY_star(L_range))/sigma_Y2;
      arma::colvec gradU_prior = arma::diagmat(1/D/sigma_beta2)*theta_beta(L_range);
      arma::colvec gradU_log = (X2_sum_b/sigma_Y2*D_delta2) *theta_beta(L_range) - (D_delta * XY_star(L_range))/sigma_Y2;
      // arma::colvec gradU_logL = (X2_sum_b/sigma_Y2*D_delta2) *(theta_beta(L_range) - theta_beta_mean_post);
      // Rcout<<"iter = "<<iter <<"; region = "<<r<<std::endl;
      // Rcout<<"arma::max(arma::abs(gradU_logL - gradU_log))"<<arma::max(arma::abs(gradU_logL - gradU_log))<<std::endl;
      arma::colvec gradU = gradU_prior + (n/subsample_size)*gradU_log;
      
      theta_beta(L_range) += -step/2*gradU + sqrt(step)*arma::randn(Lr,1);
      gradU_allregion(L_range) = gradU;
      beta(p_idx) = delta(p_idx) %( Q * theta_beta(L_range) );
      // beta(p_idx) = ( Q * theta_beta(L_range) );
      beta_star(L_range) = D_delta * theta_beta(L_range);
      
      
      
      }
      
      
      
      
      
    }// end of one region
    // 
    // log_prior of beta
    double log_prior_theta_beta = -0.5*L*log(sigma_beta2) - 0.5*arma::accu(log(D_vec)) -
      0.5/sigma_beta2*arma::dot(theta_beta, theta_beta/D_vec);
    
    t0 = clock() - t0;
    double sys_t0 = ((double)t0)/CLOCKS_PER_SEC;
    time_segment(iter,0) = sys_t0;
    
    // gradU_mcmc.col(iter) = gradU_allregion;
    
    // update theta_gamma
    t0 = clock();
    arma::mat mean_gamma = arma::mat(L,q);
    arma::vec log_prior_theta_gamma = arma::zeros(q,1); 
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
      log_prior_theta_gamma(j) = -0.5*L*log(sigma_gamma2) - 0.5*arma::accu(log(D_vec)) -
        0.5/sigma_gamma2*arma::dot(theta_gamma.col(j), theta_gamma.col(j)/D_vec);
    }
    if(testing){
      return List::create(Named("mean_gamma")  = mean_gamma,
                          Named("beta_star")  = beta_star,
                          Named("XcXq_sumsq")  = XcXq_sumsq,
                          Named("XXq_sumsq")  = XXq_sumsq);
    }
    
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
      
      // arma::colvec logL_all = arma::zeros(p);
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
      
    }
    t0 = clock() - t0;
    sys_t0 = ((double)t0)/CLOCKS_PER_SEC;
    time_segment(iter,2) = sys_t0;
    t0 = clock();
    arma::vec b_Y_vec = arma::zeros(total_batch,1); // for sigma_Y2 update
    arma::vec b_eta_vec = arma::zeros(total_batch,1);
    
    // double b_eta = 0;// for updating b_eta
    if(iter%interval_eta==0){
      for(arma::uword batch_counter=0; batch_counter<total_batch;batch_counter++){
        arma::uvec batch_range = batch_idx[batch_counter];
        arma::mat X_batch = X_list[batch_counter];
        arma::rowvec X_b = X_batch.row(0);arma::mat X_q = X_batch.rows(1,q);
        
        SEXP Y_star_fbm =  Y_star_list[batch_counter];
        XPtr<BigMatrix> Y_star_p(Y_star_fbm);
        const arma::mat Y_star_b = arma::Mat<double>((double *)Y_star_p->matrix(), 
                                                     Y_star_p->nrow(), Y_star_p->ncol(), false);
        SEXP theta_eta_b_address =  theta_eta_path[batch_counter];
        XPtr<BigMatrix> theta_eta_b_pointer(theta_eta_b_address);
        arma::mat theta_eta_b = arma::Mat<double>((double *)theta_eta_b_pointer->matrix(), 
                                                  theta_eta_b_pointer->nrow(), theta_eta_b_pointer->ncol(), false);
        const arma::mat temp_mat_b = Y_star_b - beta_star*X_b - theta_gamma*X_q;
        if(if_update_eta){
          arma::colvec eta_sigma_pos = 1/(1/sigma_Y2 + 1/sigma_eta2/D_vec);
          arma::mat theta_eta_res = temp_mat_b - theta_eta_b;
          double temp_mat_b_norm = norm(theta_eta_res,"fro");
          b_Y_vec(batch_counter) = temp_mat_b_norm*temp_mat_b_norm;
          
          // update theta_eta using SGLD
          theta_eta_res.each_col() %= eta_sigma_pos/sigma_Y2;
          arma::mat grad_theta_eta = theta_eta_b - theta_eta_res;
          grad_theta_eta.each_col() %= 1/eta_sigma_pos;
          arma::mat theta_eta_inc = -step/2*grad_theta_eta + randn(size(grad_theta_eta))*sqrt(step);
          theta_eta_b += theta_eta_inc;
          
          
          b_eta_vec(batch_counter) = arma::accu(theta_eta_b % (theta_eta_b.each_col() % (1/sqrt(D_vec)))) ;
          
        }else{
          double temp_mat_b_norm = norm(temp_mat_b - theta_eta_b,"fro");
          b_Y_vec(batch_counter) = temp_mat_b_norm*temp_mat_b_norm;
        }
      }
      
      // log prior
      double b_eta = arma::accu(b_eta_vec);
      double log_prior_theta_eta = -0.5*n*L*log(sigma_eta2) - 0.5*n*arma::accu(log(D_vec)) -
        0.5/sigma_eta2*b_eta;
      
      log_prior = log_prior_theta_beta + log_prior_theta_eta + arma::accu(log_prior_theta_gamma);
      // logLL_prior_mcmc(iter) = log_prior;
      
      
      
      arma::uvec idxzero = arma::find(b_Y_vec == 0);
      if(idxzero.n_elem == 0){
        double b_Y = arma::accu(b_Y_vec);
        sigma_Y2 = 1/arma::randg( arma::distr_param(a + n*L/2, 1/(b+b_Y/2)) );
        logLL = b_Y;
        logLL*= (-0.5/sigma_Y2);
        logLL+= -0.5*L*n*log(sigma_Y2*2*3.14159);
      }
      
      idxzero = arma::find(b_eta_vec == 0);
      if(idxzero.n_elem == 0){
        double b_eta = arma::accu(b_eta_vec);
        sigma_eta2 = 1/arma::randg( arma::distr_param(a + n*L/2, 1/(b+b_eta/2)) );
      }
      
      update_XY_eta_term_memsave(XY_eta_term,  XqYstar_theta_eta_term,
                                 XY_term_allsample, XqYstar_term_allsample,theta_eta_path,
                                 Phi_Q,region_idx,L_idx,batch_idx, X_list);
    }
    
    
    double b_beta = b+dot(theta_beta,theta_beta/D_vec)/2;
    // b_beta *= n/subsample_size;
    sigma_beta2 = 1/arma::randg( arma::distr_param(a + L/2, 1/b_beta) );
    sigma_gamma2 = 1/arma::randg( arma::distr_param(a + L*q/2, 1/(b+accu(theta_gamma%(theta_gamma.each_col()/D_vec))/2)) );
    t0 = clock() - t0;
    sys_t0 = ((double)t0)/CLOCKS_PER_SEC;
    time_segment(iter,4) = sys_t0;
    // update batch counter
    batch_counter++ ;
    if(batch_counter == total_batch-1){
      batch_counter = 0;
    }
    
  
   Rcout<<"test....7, iter="<<iter<<std::endl;
    
    if(iter > burnin){
      if((iter-burnin)%thinning == 0){
        int mcmc_iter = (iter-burnin)/thinning-1;
        theta_beta_mcmc.col(mcmc_iter) = theta_beta;
        theta_gamma_mcmc.slice(mcmc_iter) = theta_gamma;
        delta_mcmc.col(mcmc_iter) = delta;
        logLL_mcmc(mcmc_iter) = logLL;
        logLL_prior_mcmc(mcmc_iter) = log_prior;
        sigma_eta2_mcmc(mcmc_iter) = sigma_eta2;
        sigma_beta2_mcmc(mcmc_iter) = sigma_beta2;
        sigma_Y2_mcmc(mcmc_iter) = sigma_Y2;
        sigma_gamma2_mcmc(mcmc_iter) = sigma_gamma2;
        logLL_prior_theta_beta_mcmc(mcmc_iter) = log_prior_theta_beta;
        delta_logP_mcmc.col(mcmc_iter) = logL_all;
      }
    }
    
    // tune step size
    if(iter < stop_tuning_stepsize){
      step = a_step*pow((b_step+sgld_step_counter),gamma_step);
      
    }
    Rcout<<"test....8, iter="<<iter<<std::endl;
    sgld_step_mcmc(iter) = step;
    
  }//end of one iteration
  
  
  
  // return all results
  return Rcpp::List::create(Rcpp::Named("theta_beta_mcmc") = theta_beta_mcmc,
                            Rcpp::Named("theta_gamma_mcmc") = theta_gamma_mcmc,
                            Rcpp::Named("sigma_beta2_mcmc") =  sigma_beta2_mcmc,
                            Rcpp::Named("sigma_eta2_mcmc") =  sigma_eta2_mcmc,
                            Rcpp::Named("sigma_gamma2_mcmc") =  sigma_gamma2_mcmc,
                            Rcpp::Named("sigma_Y2_mcmc") =  sigma_Y2_mcmc,
                            Rcpp::Named("logLL_mcmc") = logLL_mcmc,
                            Rcpp::Named("logLL_prior_mcmc") = logLL_prior_mcmc,
                            Rcpp::Named("logLL_prior_theta_beta_mcmc") = logLL_prior_theta_beta_mcmc,
                            Rcpp::Named("delta_logP_mcmc") = delta_logP_mcmc,
                            Rcpp::Named("time_segment") = time_segment,
                            Rcpp::Named("delta_mcmc") =  delta_mcmc);
}


