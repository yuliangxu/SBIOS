#' Scalable Bayesian Image On Scalar regression
#'
#' SBIOS function is an illustrative example for working with small data sets.
#' For large scale imaging mediation analysis, please run SBIOSimp or SBIOS0 with detailed options and controls. 
#' See README.md for an example
#' @param img_list A list of path to all image Nifti data, length(img_list) is the number of subject
#' @param predictor A matrix of dimension (q+1) x n
#' @param grids A matrix of dimension p x d, d is the dimension of the image, could be 2D or 3D
#' @param out_path The directory to save the coefficient beta (beta_mean) and its marginal inclusion probability (marginal_IP) as nifti file
#' @param region_idx A list of length num_region. Users are allowed to pre-specify
#'  the region parcellation by putting indices for each region as each component in this list.
#'  The dafault value is list(region1 = 1:p) with only 1 region.
#' @param mask_list A list of path to all mask Nifti data. Default is set to NULL, and not use any imputation.
#' @param basis A list object to speficy 
#' \itemize{
#'      \item Phi_Q A list object with length num_region.
#'     Each component represents one basis function, a matrix of dimension p_r by L_r for the r-th region.
#'     \item Phi_D A list object with length num_region.
#'     Each component represents one set of eigenvalues for one region, a vector of length L_r for the r-th region.
#' }
#' @param subsample_size An integer, the subsample size used in the SGLD algorithm, must be smaller than the total number of subjects in each batch of data.
#' @param init_params A list of initial parameters
#' \itemize{
#'      \item theta_beta, A vector of length L, where L is the total number of basis functions. Default value is rep(1,L).
#'      \item theta_gamma, A q by L matrix, default at 1
#'      \item theta_eta, A L by n matrix, default at 0
#'      \item eta, A p by n matrix, default at 0
#'      \item delta, A p by 1 vector, default at 1
#'      \item sigma_Y, default at 1
#'      \item sigma_beta, default at 1
#'      \item sigma_eta, default at 1
#'      \item sigma_gamma, default at 1
#' }
#' @param controls A list of control options
#' \itemize{
#'      \item lambda, thresholding parameter, default at 0.5
#'      \item prior_p, the prior of delta follows bernoulli(prior_p), default at 0.5
#'      \item n_mcmc, total number of iterations, default at 5000
#'      \item start_delta, at which iteration to update delta, default at 0
#'      \item step, initial step size, set at 1e-3
#'      \item burnin, from which iteration to start saving MCMC samples, default at 0
#'      \item thinning, interval to save MCMC samples, default at 1,
#'      \item interval_eta, interval to update eta, dafault at 100,
#'      \item start_saving_imp, from which iteration to start save the imputed outcomes, default at 1000
#'      \item seed, set random seed, default at sample(1:1e5,1)
#'      \item step_controls, A list object of component a, b, gamma, adjust the step size using: step = a*(b+iter)^gamma
#' }
#' @param n_batch number of batches
#' @param common_mask A vector of indices to indicate which voxels in grids(rows) to be included as the common area to perform the analysys
#' @export
SBIOS = function(img_list, predictor, grids, out_path, region_idx = NULL, mask_list = NULL, basis = NULL,
                 subsample_size = 100, init_params = NULL,controls = NULL, n_batch = NULL,
                 common_mask = NULL
){
  
  if(is.null(common_mask)){
    common_mask = 1:dim(grids)[1]
  }
  common_mask_idx = which(common_mask!=0)
  grids = grids[common_mask_idx,]
  p = dim(grids)[1]
  n = length(img_list)
  if(dim(predictor)[2] != n){
    stop(paste("make sure the dimension of the predictor is (q+1) x n, q is the number of confounders. dim(predictor)=",dim(predictor)))
  }
  q = dim(predictor)[1]-1
  if(is.null(n_batch)){
    n_batch = ceiling(n/subsample_size);
  }
  
  
  
  # create region_idx
  if(is.null(region_idx)){
    region_idx = list(region1 = 1:p)
    num_region = length(region_idx)
    if(num_region != length(basis$Phi_D)){
      stop("If basis is specified, region_idx must also be specified!")
      stop("length(region_idx) must be the same as length(basis$Phi_D)!")
    }
  }
  num_region = length(region_idx)
  
  # create basis
  if(is.null(basis)){
    print("creating basis, this might take a while.....")
    l = round(p/num_region*0.1)
    basis = generate_matern_basis2(grids, region_idx, rep(l,length(region_idx)),scale = 2,nu = 1/5,
                                   show_progress=T)
    saveRDS(basis, file.path(out_path,"basis.rds"))
    print(paste("basis completed, saved in ",file.path(out_path,"basis.rds")))
    print("Please use this basis as input next time.")
  }
  basis$D_vec = unlist(basis$Phi_D)
  basis$L_all = unlist(lapply(basis$Phi_D,length))
  basis$p_length = unlist(lapply(basis$Phi_Q,function(x){dim(x)[1]}))
  L = sum(basis$L_all)
  
  # read Nifti files and split into small batches
  n_each_batch = floor(n/n_batch)
  if((n - n_each_batch*n_batch)<subsample_size){
    n_batch = n_batch - 1
    batch_number = c(rep(n_each_batch, (n_batch-1)), n - (n_batch-1)*n_each_batch)
  }
  if(sum(batch_number) != n || any(batch_number<subsample_size)){
    stop(paste("batch number computation is wrong: subsample_size=",subsample_size,
               "batch_number = ",batch_number,"; n=",n))
  }
  
  person_1 = readNifti(img_list[1])
  nifti_dimension = dim(person_1)
  
  if(!is.null(mask_list)){
    # ---------- SBIOSimp ------------- #
    print("run SBIOSimp")
    
    print("step1: preprocess data to file-backed matrices")
    X = vector("list",n_batch)
    Y = vector("list",n_batch)
    Y_star = vector("list",n_batch)
    mask_list_fbm = vector("list",n_batch)
    person_counter = 0
    for(b in 1:n_batch){
      print(paste("reading data batch ",b))
      Y_b = matrix(NA, nrow = p, ncol = batch_number[b])
      mask_b = matrix(NA, nrow = p, ncol = batch_number[b])
      
      for(i in 1:batch_number[b]){
        person_counter = person_counter+1
        person_i = readNifti(img_list[person_counter])
        mask_i = readNifti(mask_list[person_counter])
        Y_b[,i] = person_i[common_mask_idx]
        mask_b[,i] = mask_i[common_mask_idx]
      }
      Y_star[[b]] = bigmemory::as.big.matrix(High_to_low(Y_b,basis))@address
      Y[[b]] = bigmemory::as.big.matrix(Y_b)@address
      if(b == 1){
        batch_idx = 1:batch_number[b]
      }else{
        batch_idx = 1:batch_number[b] + batch_number[b-1]
      }
      X[[b]] = predictor[,batch_idx]
      mask_list_fbm[[b]] = bigmemory::as.big.matrix(mask_b)
    }
    
    data_list = list(Y_list = Y, X_list = X,
                     Y_star_list = Y_star)
    
    total_imp = unlist(lapply(mask_list_fbm, function(x){sum(bigmemory::as.matrix(x)==0)}))
    
    imp_idx_list = get_imp_list(mask_list_fbm,region_idx)
    
    if(is.null(controls)){
      controls = list(lambda = 0.5, prior_p = 0.5, n_mcmc = 5000,
                      start_delta = 0 , step = 1e-3,
                      burnin = 0, thinning = 10, interval_eta = 100,start_saving_imp = 1000,
                      seed = sample(1:1e5,1))
      
      controls$step_controls = list(a=0.0001,b=10,gamma = -0.35)
    }
      controls$subsample_size = subsample_size
    if(is.null(init_params)){
      init_params = list(theta_beta = rep(1,L),
                         theta_gamma = matrix(rep(1,q*L), nrow = L),
                         theta_eta = matrix(0,nrow=L,ncol=n),
                         eta = matrix(0,nrow=p,ncol=n),
                         delta = rep(1,p),
                         sigma_Y = 1, sigma_beta = 1, sigma_eta = 1, sigma_gamma=1)
    }
    
    
    # create indices for voxel locations and basis 
    dimensions = list(n=n,L=L, p=p, q=q)
    region_idx_cpp = lapply(region_idx,function(x){x-1})
    L_idx = vector("list",num_region)
    L_start = 1
    for(r in 1:num_region){
      L_end = L_start + basis$L_all[1] - 1
      L_idx[[r]] = L_start:L_end
      L_start = L_end + 1
    }
    L_idx_cpp = lapply(L_idx,function(x){x-1})
    batch_idx = vector("list",length(data_list$Y_list))
    b_start = 1
    for(b in 1:length(data_list$Y_list)){
      b_end = b_start + dim(bigmemory::as.matrix(mask_list_fbm[[b]]) )[2] - 1
      batch_idx[[b]] = b_start:b_end
      b_start = b_end + 1
    }
    batch_idx_cpp = lapply(batch_idx,function(x){x-1})
    
    print("step2: running SBIOSimp")
    sgld = SBIOSimp(data_list, basis, dimensions, imp_idx_list, total_imp,
                    init_params, region_idx_cpp, L_idx_cpp,
                    batch_idx_cpp, lambda = controls$lambda, prior_p = controls$prior_p,
                    n_mcmc = controls$n_mcmc,
                    start_saving_imp = controls$start_saving_imp,
                    start_delta=controls$start_delta,
                    subsample_size=controls$subsample_size, step = controls$step,
                    burnin = controls$burnin,
                    seed = controls$seed,
                    thinning = controls$thinning,
                    interval_eta = controls$interval_eta,
                    all_sgld = 1,
                    a_step = controls$step_controls$a,
                    b_step = controls$step_controls$b,
                    gamma_step = controls$step_controls$gamma,
                    update_individual_effect = 1,
                    a = 1, b = 1,
                    testing = 0, display_progress = 1)
    
  }else{
    # ---------- SBIOS0 ------------- #
    print("run SBIOS0")
    print("step1: preprocess data to file-backed matrices")
    X = vector("list",n_batch)
    Y = vector("list",n_batch)
    Y_star = vector("list",n_batch)
    person_counter = 0
    for(b in 1:n_batch){
      print(paste("reading data batch ",b))
      Y_b = matrix(NA, nrow = p, ncol = batch_number[b])
      
      for(i in 1:batch_number[b]){
        person_counter = person_counter+1
        person_i = readNifti(img_list[person_counter])
        Y_b[,i] = person_i[common_mask_idx]
      }
      Y_star[[b]] = bigmemory::as.big.matrix(High_to_low(Y_b,basis))@address
      Y[[b]] = bigmemory::as.big.matrix(Y_b)@address
      if(b == 1){
        batch_idx = 1:batch_number[b]
      }else{
        batch_idx = 1:batch_number[b] + batch_number[b-1]
      }
      X[[b]] = predictor[,batch_idx]
    }
    
    data_list = list(Y_list = Y, X_list = X,
                     Y_star_list = Y_star)
    
    
    
    if(is.null(controls)){
      controls = list(lambda = 0.5, prior_p = 0.5, n_mcmc = 5000,
                      start_delta = 0 , step = 1e-3,
                      burnin = 0, thinning = 10, interval_eta = 100,start_saving_imp = 1000,
                      seed = sample(1:1e5,1))
      controls$step_controls = list(a=0.0001,b=10,gamma = -0.35)
    }
      controls$subsample_size = subsample_size
    if(is.null(init_params)){
      init_params = list(theta_beta = rep(1,L),
                         theta_gamma = matrix(rep(1,q*L), nrow = L),
                         theta_eta = matrix(0,nrow=L,ncol=n),
                         eta = matrix(0,nrow=p,ncol=n),
                         delta = rep(1,p),
                         sigma_Y = 1, sigma_beta = 1, sigma_eta = 1, sigma_gamma=1)
    }
    
    
    # create indices for voxel locations and basis 
    dimensions = list(n=n,L=L, p=p, q=q)
    region_idx_cpp = lapply(region_idx,function(x){x-1})
    L_idx = vector("list",num_region)
    L_start = 1
    for(r in 1:num_region){
      L_end = L_start + basis$L_all[1] - 1
      L_idx[[r]] = L_start:L_end
      L_start = L_end + 1
    }
    L_idx_cpp = lapply(L_idx,function(x){x-1})
    batch_idx = vector("list",length(data_list$Y_list))
    b_start = 1
    for(b in 1:length(data_list$Y_list)){
      b_end = b_start + dim(data_list$X_list[[b]])[2] - 1
      batch_idx[[b]] = b_start:b_end
      b_start = b_end + 1
    }
    batch_idx_cpp = lapply(batch_idx,function(x){x-1})
    
    
    init_theta_eta = init_params$theta_eta
    init_params$theta_eta = NULL
    init_params$eta = NULL
    theta_eta_path = vector("list",n_batch)
    for(b in 1:n_batch){
      print(paste("create theta_eta path for batch ",b))
      batch_idx = batch_idx_cpp[[b]]+1
      theta_eta_path[[b]] = bigmemory::as.big.matrix(init_theta_eta[,batch_idx])@address
    }
    print("step2: running SBIOS0")
    sgld = SBIOS0(data_list, basis, theta_eta_path,
                  dimensions,
                  init_params, region_idx_cpp, L_idx_cpp,
                  batch_idx_cpp, lambda = controls$lambda, prior_p = controls$prior_p,
                  n_mcmc = controls$n_mcmc,  start_delta=controls$start_delta,
                  subsample_size=controls$subsample_size, step = controls$step,
                  burnin = controls$burnin,
                  thinning = controls$thinning,
                  interval_eta = controls$interval_eta,
                  a_step = controls$step_controls$a,
                  b_step = controls$step_controls$b,
                  gamma_step = controls$step_controls$gamma,
                  update_individual_effect = 1,
                  a = 1, b = 1,
                  testing = 0, display_progress = 1)
  }
  
  print("step3: summarize result and output nifti file")
  # summarize result and output nifti file
  burnin = as.integer(((0.8*controls$n_mcmc):(controls$n_mcmc-1))/controls$thinning)
  burnin = burnin[!duplicated(burnin)]
  beta_mcmc = Low_to_high( sgld$theta_beta_mcmc[,burnin],basis)
  
  beta_mean = apply(beta_mcmc,1,mean)
  marginal_IP = apply(sgld$delta_mcmc[,burnin],1,mean)
  
  # write into Nifti
  out = rep(NA,prod(nifti_dimension))
  out[common_mask_idx] = beta_mean
  empty_nifti <- array(data = out, dim = nifti_dimension)
  nifti_object <- nifti(empty_nifti, datatype=16)
  writeNIfTI(nifti_object, 
             file.path(out_path,paste("beta_mean",sep="")))
  
  out = rep(NA,prod(nifti_dimension))
  out[common_mask_idx] = marginal_IP
  empty_nifti <- array(data = out, dim = nifti_dimension)
  nifti_object <- nifti(empty_nifti, datatype=16)
  writeNIfTI(nifti_object, 
             file.path(out_path,paste("marginal_IP",sep="")))
  
  return(sgld)
}
