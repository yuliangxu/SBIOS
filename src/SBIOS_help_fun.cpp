// Using the new derivations
#include <RcppArmadillo.h>
#include "SBIOS_help_fun.h"
#include <bigmemory/BigMatrix.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
arma::mat big_address2mat(SEXP bigmat_address) {
  
  XPtr<BigMatrix> bigmat_pointer(bigmat_address);
  return arma::Mat<double>((double *)bigmat_pointer->matrix(), bigmat_pointer->nrow(), bigmat_pointer->ncol(), false);
}

// [[Rcpp::export]]
void set_seed(double seed) {
  Rcpp::Environment base_env("package:base");
  Rcpp::Function set_seed_r = base_env["set.seed"];
  set_seed_r(std::floor(std::fabs(seed)));
}

arma::mat extractBigMatCols(const arma::mat& aBigMat, arma::uvec idx ) {
  return aBigMat.cols(idx);
}
// [[Rcpp::export]]
arma::uvec complement(arma::uword start, arma::uword end, arma::uword n) {
  arma::uvec y1 = arma::linspace<arma::uvec>(0, start-1, start);
  arma::uvec y2 = arma::linspace<arma::uvec>(end+1, n-1, n-1-end);
  arma::uvec y = arma::join_cols(y1,y2);
  return y;
}
arma::mat Low_to_high(arma::mat& Low_mat, int p, Rcpp::List& Phi_Q,
                      Rcpp::List& region_idx, Rcpp::List& L_idx){
  int num_region = region_idx.size();
  int n = Low_mat.n_cols;
  arma::mat High_mat(p,n);
  for(arma::uword r=0; r<num_region; r++){
    arma::uvec p_idx = region_idx[r];
    arma::uvec L_range = L_idx[r];
    arma::mat Q = Phi_Q[r];
    High_mat.rows(p_idx) = Q*Low_mat.rows(L_range);
  }
  return High_mat;
}
arma::colvec Low_to_high_vec(const arma::colvec& Low_vec, int p,
                             const Rcpp::List& Phi_Q,
                             const Rcpp::List& region_idx, 
                             const Rcpp::List& L_idx){
  int num_region = region_idx.size();
  arma::colvec High_vec(p,1);
  for(arma::uword r=0; r<num_region; r++){
    arma::uvec p_idx = region_idx[r];
    arma::uvec L_range = L_idx[r];
    arma::mat Q = Phi_Q[r];
    High_vec(p_idx) = Q*Low_vec(L_range);
  }
  return High_vec;
}

void update_XY_eta_term(arma::colvec& XY_eta_term, arma::mat& XqYstar_theta_eta_term,
                        const arma::colvec& XY_term_allsample, const arma::mat& XqYstar_term_allsample,
                        const arma::mat& theta_eta, Rcpp::List& Phi_Q,Rcpp::List& region_idx, Rcpp::List& L_idx,
                        Rcpp::List& batch_idx, Rcpp::List& X_list){
  int total_batch = batch_idx.size();
  XY_eta_term = XY_term_allsample;
  int L = XqYstar_term_allsample.n_rows;
  int q = XqYstar_term_allsample.n_cols;
  XqYstar_theta_eta_term = arma::zeros(L,q);
  int p = XY_term_allsample.n_elem;
  for(int b = 0; b<total_batch; b++){
    arma::uvec batch_range = batch_idx[b];
    arma::mat theta_eta_b = theta_eta.cols(batch_range);
    arma::mat X_batch = X_list[b];
    int q = X_batch.n_rows-1;
    arma::rowvec X_b = X_batch.row(0);
    arma::mat X_q = X_batch.rows(1,q); //q by n_batch
    XqYstar_theta_eta_term += theta_eta_b * X_q.t();
    
    // theta_eta_b.each_row() %= X_b;
    // arma::colvec theta_eta_X_sum = sum(theta_eta_b,1);
    arma::colvec theta_eta_X_sum = theta_eta_b * X_b.t();
    arma::colvec eta_X_sum = Low_to_high_vec(theta_eta_X_sum, p,Phi_Q, region_idx, L_idx);
    XY_eta_term += -eta_X_sum;
  }
  XqYstar_theta_eta_term = XqYstar_term_allsample - XqYstar_theta_eta_term;
}

void update_XY_eta_term_memsave(arma::colvec& XY_eta_term, arma::mat& XqYstar_theta_eta_term,
                        const arma::colvec& XY_term_allsample, const arma::mat& XqYstar_term_allsample,
                        const Rcpp::List& theta_eta_path, Rcpp::List& Phi_Q,Rcpp::List& region_idx, Rcpp::List& L_idx,
                        Rcpp::List& batch_idx, Rcpp::List& X_list){
  int total_batch = batch_idx.size();
  XY_eta_term = XY_term_allsample;
  int L = XqYstar_term_allsample.n_rows;
  int q = XqYstar_term_allsample.n_cols;
  XqYstar_theta_eta_term = arma::zeros(L,q);
  int p = XY_term_allsample.n_elem;
  for(int b = 0; b<total_batch; b++){
    arma::uvec batch_range = batch_idx[b];
    SEXP theta_eta_b_address =  theta_eta_path[b];
    XPtr<BigMatrix> theta_eta_b_pointer(theta_eta_b_address);
    const arma::mat theta_eta_b = arma::Mat<double>((double *)theta_eta_b_pointer->matrix(), theta_eta_b_pointer->nrow(), theta_eta_b_pointer->ncol(), false);
    arma::mat X_batch = X_list[b];
    int q = X_batch.n_rows-1;
    arma::rowvec X_b = X_batch.row(0);
    arma::mat X_q = X_batch.rows(1,q); //q by n_batch
    XqYstar_theta_eta_term += theta_eta_b * X_q.t();
    
    // theta_eta_b.each_row() %= X_b;
    // arma::colvec theta_eta_X_sum = sum(theta_eta_b,1);
    arma::colvec theta_eta_X_sum = theta_eta_b * X_b.t();
    arma::colvec eta_X_sum = Low_to_high_vec(theta_eta_X_sum, p,Phi_Q, region_idx, L_idx);
    XY_eta_term += -eta_X_sum;
  }
  XqYstar_theta_eta_term = XqYstar_term_allsample - XqYstar_theta_eta_term;
}

