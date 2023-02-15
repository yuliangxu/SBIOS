#ifndef SBIOS_HELP_FUN_H
#define SBIOS_HELP_FUN_H

#include <RcppArmadillo.h>

void set_seed(double seed);
arma::mat extractBigMatCols(const arma::mat& aBigMat, arma::uvec idx );
arma::uvec complement(arma::uword start, arma::uword end, arma::uword n);
arma::mat Low_to_high(arma::mat& Low_mat, int p, Rcpp::List& Phi_Q,
                      Rcpp::List& region_idx, Rcpp::List& L_idx);
arma::colvec Low_to_high_vec(const arma::colvec& Low_vec, int p,
                             const Rcpp::List& Phi_Q,
                             const Rcpp::List& region_idx, 
                             const Rcpp::List& L_idx);
void update_XY_eta_term(arma::colvec& XY_eta_term, arma::mat& XqYstar_theta_eta_term,
                        const arma::colvec& XY_term_allsample, const arma::mat& XqYstar_term_allsample,
                        const arma::mat& theta_eta, Rcpp::List& Phi_Q,Rcpp::List& region_idx, Rcpp::List& L_idx,
                        Rcpp::List& batch_idx, Rcpp::List& X_list);
#endif