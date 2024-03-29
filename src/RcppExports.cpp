// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// SBIOS0
List SBIOS0(Rcpp::List& data_list, Rcpp::List& basis, Rcpp::List& theta_eta_path, Rcpp::List& dimensions, Rcpp::List& init_params, Rcpp::List& region_idx, Rcpp::List& L_idx, Rcpp::List& batch_idx, double lambda, double prior_p, int n_mcmc, int start_delta, int subsample_size, double step, int burnin, int thinning, double a, double b, int interval_eta, int start_eta, int start_saving_eta, int stop_tuning_stepsize, bool all_sgld, int sgld_freq, double a_step, double b_step, double gamma_step, bool update_individual_effect, bool testing, bool display_progress);
RcppExport SEXP _SBIOS_SBIOS0(SEXP data_listSEXP, SEXP basisSEXP, SEXP theta_eta_pathSEXP, SEXP dimensionsSEXP, SEXP init_paramsSEXP, SEXP region_idxSEXP, SEXP L_idxSEXP, SEXP batch_idxSEXP, SEXP lambdaSEXP, SEXP prior_pSEXP, SEXP n_mcmcSEXP, SEXP start_deltaSEXP, SEXP subsample_sizeSEXP, SEXP stepSEXP, SEXP burninSEXP, SEXP thinningSEXP, SEXP aSEXP, SEXP bSEXP, SEXP interval_etaSEXP, SEXP start_etaSEXP, SEXP start_saving_etaSEXP, SEXP stop_tuning_stepsizeSEXP, SEXP all_sgldSEXP, SEXP sgld_freqSEXP, SEXP a_stepSEXP, SEXP b_stepSEXP, SEXP gamma_stepSEXP, SEXP update_individual_effectSEXP, SEXP testingSEXP, SEXP display_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List& >::type data_list(data_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type basis(basisSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type theta_eta_path(theta_eta_pathSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type dimensions(dimensionsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type init_params(init_paramsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type region_idx(region_idxSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type L_idx(L_idxSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type batch_idx(batch_idxSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type prior_p(prior_pSEXP);
    Rcpp::traits::input_parameter< int >::type n_mcmc(n_mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type start_delta(start_deltaSEXP);
    Rcpp::traits::input_parameter< int >::type subsample_size(subsample_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type step(stepSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thinning(thinningSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< int >::type interval_eta(interval_etaSEXP);
    Rcpp::traits::input_parameter< int >::type start_eta(start_etaSEXP);
    Rcpp::traits::input_parameter< int >::type start_saving_eta(start_saving_etaSEXP);
    Rcpp::traits::input_parameter< int >::type stop_tuning_stepsize(stop_tuning_stepsizeSEXP);
    Rcpp::traits::input_parameter< bool >::type all_sgld(all_sgldSEXP);
    Rcpp::traits::input_parameter< int >::type sgld_freq(sgld_freqSEXP);
    Rcpp::traits::input_parameter< double >::type a_step(a_stepSEXP);
    Rcpp::traits::input_parameter< double >::type b_step(b_stepSEXP);
    Rcpp::traits::input_parameter< double >::type gamma_step(gamma_stepSEXP);
    Rcpp::traits::input_parameter< bool >::type update_individual_effect(update_individual_effectSEXP);
    Rcpp::traits::input_parameter< bool >::type testing(testingSEXP);
    Rcpp::traits::input_parameter< bool >::type display_progress(display_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(SBIOS0(data_list, basis, theta_eta_path, dimensions, init_params, region_idx, L_idx, batch_idx, lambda, prior_p, n_mcmc, start_delta, subsample_size, step, burnin, thinning, a, b, interval_eta, start_eta, start_saving_eta, stop_tuning_stepsize, all_sgld, sgld_freq, a_step, b_step, gamma_step, update_individual_effect, testing, display_progress));
    return rcpp_result_gen;
END_RCPP
}
// big_address2mat
arma::mat big_address2mat(SEXP bigmat_address);
RcppExport SEXP _SBIOS_big_address2mat(SEXP bigmat_addressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type bigmat_address(bigmat_addressSEXP);
    rcpp_result_gen = Rcpp::wrap(big_address2mat(bigmat_address));
    return rcpp_result_gen;
END_RCPP
}
// set_seed
void set_seed(double seed);
RcppExport SEXP _SBIOS_set_seed(SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type seed(seedSEXP);
    set_seed(seed);
    return R_NilValue;
END_RCPP
}
// complement
arma::uvec complement(arma::uword start, arma::uword end, arma::uword n);
RcppExport SEXP _SBIOS_complement(SEXP startSEXP, SEXP endSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::uword >::type start(startSEXP);
    Rcpp::traits::input_parameter< arma::uword >::type end(endSEXP);
    Rcpp::traits::input_parameter< arma::uword >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(complement(start, end, n));
    return rcpp_result_gen;
END_RCPP
}
// extract_mask
arma::mat extract_mask(SEXP mask_fbm);
RcppExport SEXP _SBIOS_extract_mask(SEXP mask_fbmSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type mask_fbm(mask_fbmSEXP);
    rcpp_result_gen = Rcpp::wrap(extract_mask(mask_fbm));
    return rcpp_result_gen;
END_RCPP
}
// update_Y_imp
void update_Y_imp(const arma::mat& Y_star, arma::vec& Y_imp, arma::vec beta, arma::rowvec X_b, double sigma_Y, arma::mat gamma, arma::mat X_q, const List& region_idx, const List& L_idx, const List& Phi_Q, const List& imp_vec_b, const List& imp_p_b);
RcppExport SEXP _SBIOS_update_Y_imp(SEXP Y_starSEXP, SEXP Y_impSEXP, SEXP betaSEXP, SEXP X_bSEXP, SEXP sigma_YSEXP, SEXP gammaSEXP, SEXP X_qSEXP, SEXP region_idxSEXP, SEXP L_idxSEXP, SEXP Phi_QSEXP, SEXP imp_vec_bSEXP, SEXP imp_p_bSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Y_star(Y_starSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type Y_imp(Y_impSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::rowvec >::type X_b(X_bSEXP);
    Rcpp::traits::input_parameter< double >::type sigma_Y(sigma_YSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X_q(X_qSEXP);
    Rcpp::traits::input_parameter< const List& >::type region_idx(region_idxSEXP);
    Rcpp::traits::input_parameter< const List& >::type L_idx(L_idxSEXP);
    Rcpp::traits::input_parameter< const List& >::type Phi_Q(Phi_QSEXP);
    Rcpp::traits::input_parameter< const List& >::type imp_vec_b(imp_vec_bSEXP);
    Rcpp::traits::input_parameter< const List& >::type imp_p_b(imp_p_bSEXP);
    update_Y_imp(Y_star, Y_imp, beta, X_b, sigma_Y, gamma, X_q, region_idx, L_idx, Phi_Q, imp_vec_b, imp_p_b);
    return R_NilValue;
END_RCPP
}
// update_Y_mat
void update_Y_mat(arma::mat& Y_mat, arma::vec& Y_imp, arma::rowvec X_b, const List& region_idx, const List& imp_vec_b, const List& imp_p_b);
RcppExport SEXP _SBIOS_update_Y_mat(SEXP Y_matSEXP, SEXP Y_impSEXP, SEXP X_bSEXP, SEXP region_idxSEXP, SEXP imp_vec_bSEXP, SEXP imp_p_bSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type Y_mat(Y_matSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type Y_imp(Y_impSEXP);
    Rcpp::traits::input_parameter< arma::rowvec >::type X_b(X_bSEXP);
    Rcpp::traits::input_parameter< const List& >::type region_idx(region_idxSEXP);
    Rcpp::traits::input_parameter< const List& >::type imp_vec_b(imp_vec_bSEXP);
    Rcpp::traits::input_parameter< const List& >::type imp_p_b(imp_p_bSEXP);
    update_Y_mat(Y_mat, Y_imp, X_b, region_idx, imp_vec_b, imp_p_b);
    return R_NilValue;
END_RCPP
}
// SBIOSimp
List SBIOSimp(Rcpp::List& data_list, Rcpp::List& basis, Rcpp::List& dimensions, Rcpp::List& imp_idx_list, arma::uvec total_imp, Rcpp::List& init_params, Rcpp::List& region_idx, Rcpp::List& L_idx, Rcpp::List& batch_idx, double lambda, double prior_p, int n_mcmc, int start_saving_imp, int start_delta, int subsample_size, double step, int begin_eta, int seed, int thinning, int burnin, double a, double b, int interval_eta, bool all_sgld, double a_step, double b_step, double gamma_step, bool testing, bool display_progress, bool update_individual_effect);
RcppExport SEXP _SBIOS_SBIOSimp(SEXP data_listSEXP, SEXP basisSEXP, SEXP dimensionsSEXP, SEXP imp_idx_listSEXP, SEXP total_impSEXP, SEXP init_paramsSEXP, SEXP region_idxSEXP, SEXP L_idxSEXP, SEXP batch_idxSEXP, SEXP lambdaSEXP, SEXP prior_pSEXP, SEXP n_mcmcSEXP, SEXP start_saving_impSEXP, SEXP start_deltaSEXP, SEXP subsample_sizeSEXP, SEXP stepSEXP, SEXP begin_etaSEXP, SEXP seedSEXP, SEXP thinningSEXP, SEXP burninSEXP, SEXP aSEXP, SEXP bSEXP, SEXP interval_etaSEXP, SEXP all_sgldSEXP, SEXP a_stepSEXP, SEXP b_stepSEXP, SEXP gamma_stepSEXP, SEXP testingSEXP, SEXP display_progressSEXP, SEXP update_individual_effectSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List& >::type data_list(data_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type basis(basisSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type dimensions(dimensionsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type imp_idx_list(imp_idx_listSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type total_imp(total_impSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type init_params(init_paramsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type region_idx(region_idxSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type L_idx(L_idxSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type batch_idx(batch_idxSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type prior_p(prior_pSEXP);
    Rcpp::traits::input_parameter< int >::type n_mcmc(n_mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type start_saving_imp(start_saving_impSEXP);
    Rcpp::traits::input_parameter< int >::type start_delta(start_deltaSEXP);
    Rcpp::traits::input_parameter< int >::type subsample_size(subsample_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type step(stepSEXP);
    Rcpp::traits::input_parameter< int >::type begin_eta(begin_etaSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< int >::type thinning(thinningSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< int >::type interval_eta(interval_etaSEXP);
    Rcpp::traits::input_parameter< bool >::type all_sgld(all_sgldSEXP);
    Rcpp::traits::input_parameter< double >::type a_step(a_stepSEXP);
    Rcpp::traits::input_parameter< double >::type b_step(b_stepSEXP);
    Rcpp::traits::input_parameter< double >::type gamma_step(gamma_stepSEXP);
    Rcpp::traits::input_parameter< bool >::type testing(testingSEXP);
    Rcpp::traits::input_parameter< bool >::type display_progress(display_progressSEXP);
    Rcpp::traits::input_parameter< bool >::type update_individual_effect(update_individual_effectSEXP);
    rcpp_result_gen = Rcpp::wrap(SBIOSimp(data_list, basis, dimensions, imp_idx_list, total_imp, init_params, region_idx, L_idx, batch_idx, lambda, prior_p, n_mcmc, start_saving_imp, start_delta, subsample_size, step, begin_eta, seed, thinning, burnin, a, b, interval_eta, all_sgld, a_step, b_step, gamma_step, testing, display_progress, update_individual_effect));
    return rcpp_result_gen;
END_RCPP
}
// method2_SGLD_multiGP_w_eta
List method2_SGLD_multiGP_w_eta(Rcpp::List& data_list, Rcpp::List& basis, Rcpp::List& dimensions, Rcpp::List& init_params, Rcpp::List& region_idx, Rcpp::List& L_idx, Rcpp::List& batch_idx, double lambda, double prior_p, int n_mcmc, int start_delta, int subsample_size, double step, int burnin, int thinning, double a, double b, int interval_eta, int start_eta, int start_saving_eta, bool all_sgld, double a_step, double b_step, double gamma_step, bool update_individual_effect, bool testing, bool display_progress);
RcppExport SEXP _SBIOS_method2_SGLD_multiGP_w_eta(SEXP data_listSEXP, SEXP basisSEXP, SEXP dimensionsSEXP, SEXP init_paramsSEXP, SEXP region_idxSEXP, SEXP L_idxSEXP, SEXP batch_idxSEXP, SEXP lambdaSEXP, SEXP prior_pSEXP, SEXP n_mcmcSEXP, SEXP start_deltaSEXP, SEXP subsample_sizeSEXP, SEXP stepSEXP, SEXP burninSEXP, SEXP thinningSEXP, SEXP aSEXP, SEXP bSEXP, SEXP interval_etaSEXP, SEXP start_etaSEXP, SEXP start_saving_etaSEXP, SEXP all_sgldSEXP, SEXP a_stepSEXP, SEXP b_stepSEXP, SEXP gamma_stepSEXP, SEXP update_individual_effectSEXP, SEXP testingSEXP, SEXP display_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List& >::type data_list(data_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type basis(basisSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type dimensions(dimensionsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type init_params(init_paramsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type region_idx(region_idxSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type L_idx(L_idxSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type batch_idx(batch_idxSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type prior_p(prior_pSEXP);
    Rcpp::traits::input_parameter< int >::type n_mcmc(n_mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type start_delta(start_deltaSEXP);
    Rcpp::traits::input_parameter< int >::type subsample_size(subsample_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type step(stepSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thinning(thinningSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< int >::type interval_eta(interval_etaSEXP);
    Rcpp::traits::input_parameter< int >::type start_eta(start_etaSEXP);
    Rcpp::traits::input_parameter< int >::type start_saving_eta(start_saving_etaSEXP);
    Rcpp::traits::input_parameter< bool >::type all_sgld(all_sgldSEXP);
    Rcpp::traits::input_parameter< double >::type a_step(a_stepSEXP);
    Rcpp::traits::input_parameter< double >::type b_step(b_stepSEXP);
    Rcpp::traits::input_parameter< double >::type gamma_step(gamma_stepSEXP);
    Rcpp::traits::input_parameter< bool >::type update_individual_effect(update_individual_effectSEXP);
    Rcpp::traits::input_parameter< bool >::type testing(testingSEXP);
    Rcpp::traits::input_parameter< bool >::type display_progress(display_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(method2_SGLD_multiGP_w_eta(data_list, basis, dimensions, init_params, region_idx, L_idx, batch_idx, lambda, prior_p, n_mcmc, start_delta, subsample_size, step, burnin, thinning, a, b, interval_eta, start_eta, start_saving_eta, all_sgld, a_step, b_step, gamma_step, update_individual_effect, testing, display_progress));
    return rcpp_result_gen;
END_RCPP
}
// method2_gs_no_mem
List method2_gs_no_mem(Rcpp::List& data_list, Rcpp::List& basis, Rcpp::List& dimensions, Rcpp::List& init_params, Rcpp::List& region_idx, Rcpp::List& L_idx, Rcpp::List& batch_idx, double lambda, double prior_p, int n_mcmc, int start_delta, int subsample_size, double step, int burnin, int thinning, double a, double b, int interval_eta, int start_eta, int start_saving_eta, bool update_individual_effect, int seed, bool testing, bool display_progress);
RcppExport SEXP _SBIOS_method2_gs_no_mem(SEXP data_listSEXP, SEXP basisSEXP, SEXP dimensionsSEXP, SEXP init_paramsSEXP, SEXP region_idxSEXP, SEXP L_idxSEXP, SEXP batch_idxSEXP, SEXP lambdaSEXP, SEXP prior_pSEXP, SEXP n_mcmcSEXP, SEXP start_deltaSEXP, SEXP subsample_sizeSEXP, SEXP stepSEXP, SEXP burninSEXP, SEXP thinningSEXP, SEXP aSEXP, SEXP bSEXP, SEXP interval_etaSEXP, SEXP start_etaSEXP, SEXP start_saving_etaSEXP, SEXP update_individual_effectSEXP, SEXP seedSEXP, SEXP testingSEXP, SEXP display_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List& >::type data_list(data_listSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type basis(basisSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type dimensions(dimensionsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type init_params(init_paramsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type region_idx(region_idxSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type L_idx(L_idxSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type batch_idx(batch_idxSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type prior_p(prior_pSEXP);
    Rcpp::traits::input_parameter< int >::type n_mcmc(n_mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type start_delta(start_deltaSEXP);
    Rcpp::traits::input_parameter< int >::type subsample_size(subsample_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type step(stepSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thinning(thinningSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< int >::type interval_eta(interval_etaSEXP);
    Rcpp::traits::input_parameter< int >::type start_eta(start_etaSEXP);
    Rcpp::traits::input_parameter< int >::type start_saving_eta(start_saving_etaSEXP);
    Rcpp::traits::input_parameter< bool >::type update_individual_effect(update_individual_effectSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< bool >::type testing(testingSEXP);
    Rcpp::traits::input_parameter< bool >::type display_progress(display_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(method2_gs_no_mem(data_list, basis, dimensions, init_params, region_idx, L_idx, batch_idx, lambda, prior_p, n_mcmc, start_delta, subsample_size, step, burnin, thinning, a, b, interval_eta, start_eta, start_saving_eta, update_individual_effect, seed, testing, display_progress));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_SBIOS_SBIOS0", (DL_FUNC) &_SBIOS_SBIOS0, 30},
    {"_SBIOS_big_address2mat", (DL_FUNC) &_SBIOS_big_address2mat, 1},
    {"_SBIOS_set_seed", (DL_FUNC) &_SBIOS_set_seed, 1},
    {"_SBIOS_complement", (DL_FUNC) &_SBIOS_complement, 3},
    {"_SBIOS_extract_mask", (DL_FUNC) &_SBIOS_extract_mask, 1},
    {"_SBIOS_update_Y_imp", (DL_FUNC) &_SBIOS_update_Y_imp, 12},
    {"_SBIOS_update_Y_mat", (DL_FUNC) &_SBIOS_update_Y_mat, 6},
    {"_SBIOS_SBIOSimp", (DL_FUNC) &_SBIOS_SBIOSimp, 30},
    {"_SBIOS_method2_SGLD_multiGP_w_eta", (DL_FUNC) &_SBIOS_method2_SGLD_multiGP_w_eta, 27},
    {"_SBIOS_method2_gs_no_mem", (DL_FUNC) &_SBIOS_method2_gs_no_mem, 24},
    {NULL, NULL, 0}
};

RcppExport void R_init_SBIOS(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
