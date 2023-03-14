
gc(reset = TRUE)
devtools::load_all()
n_rep=Sys.getenv("SLURM_ARRAY_TASK_ID")
sim_name = "scalable_sim"
datsim_name = paste(sim_name,n_rep,sep="")
# outpath = file.path("/gpfs/accounts/jiankang_root/jiankang1/yuliangx/GSRA/sim_data") # for remote
outpath = file.path("./data") # for local

missing_percent = 0.5
sigma_Y = 1

common_percent = 0.9
n=2000; subsample_size = 200
side = 60
n_batch = 4;



contiguous_common = T
library(BayesGPfit)
library(ggplot2)
library(viridis)
library(profvis)

if(n_rep != ""){
  Rprof_name = paste(sim_name,"Rprof_sim_",n_rep,".out",sep="")
}else{
  Rprof_name = "Rprof_sim0.out"
}
Rprof_name_sn = paste(Rprof_name,"_sn",sep="")
Rprof_name_ln = paste(Rprof_name,"_ln",sep="")

# create a testing case ---------------------------------------------------


num_region = 9

p = side*side
region_idx = vector("list",num_region)
grids = GP.generate.grids(d=2L,num_grids=side)


idx_matr = matrix(1:(side*side),ncol = side)
side_per_region = side/sqrt(num_region)
for(r in 1:num_region){
  idx = rep(NA,(side_per_region)^2)
  colr = r - floor(r/sqrt(num_region))*sqrt(num_region);if(colr==0){
    colr = sqrt(num_region)
  }
  rowr = ceiling(r/sqrt(num_region));
  col_range = (max(colr-1,0):colr)*side_per_region;col_range[1] = col_range[1]+1;
  row_range = (max(rowr-1,0):rowr)*side_per_region;row_range[1] = row_range[1]+1;
  region_idx[[r]] = c(idx_matr[row_range[1]:row_range[2],col_range[1]:col_range[2]])
}



grids_df = as.data.frame(grids)


center = apply(grids,2,mean)
rad = apply(grids,1,function(x){sum((x-center)^2)})
inv_rad = 2-rad
inv_rad_ST = Soft_threshold(inv_rad,1.2)

beta = log(inv_rad_ST^2+1)

l = round(p/num_region*0.1)
GP = generate_matern_basis2(grids, region_idx, rep(l,length(region_idx)),scale = 2,nu = 1/5,
                            show_progress=T)


L = sum(GP$L_all)



# generate gamma from basis
q = 4
theta_gamma = matrix(rnorm(L*q), nrow = L)
gamma = Low_to_high(theta_gamma,GP)


data_params = generate_large_block_multiGP_data_FBM(n,beta,gamma,GP,region_idx,outpath,datsim_name,
                                                    sigma_Y=sigma_Y,
                                                    q = q,n_batch = n_batch)
true_coef = cbind(data_params$beta*data_params$delta,data_params$gamma)


data_nm =file.path(outpath,paste(datsim_name,"data_multiGP_batch_n",n,"_p",p,"_L",L,sep=""))

data_path_list = NULL
for(b in 1:n_batch){
  data_path_list = c(data_path_list,paste(data_nm,"_b",b,".rds",sep=""))
}


basis = GP; 
basis$D_vec = unlist(basis$Phi_D)

L = length(unlist(GP$Phi_D))
init_params0 = list(theta_beta = rep(1,L),
                    theta_gamma = matrix(rep(1,q*L), nrow = L),
                    theta_eta = matrix(0,nrow=L,ncol=n),
                    eta = matrix(0,nrow=p,ncol=n),
                    delta = rep(1,p),
                    sigma_Y = 1, sigma_beta = 1, sigma_eta = 1, sigma_gamma=1)
beta_GP = beta
lambda = 0.5
beta_GP[beta>0] = beta[beta>0] + lambda
beta_GP[beta<0] = beta[beta<0] - lambda
theta_beta_init = High_to_low(as.matrix(beta_GP),basis)
delta = rep(1,p)
delta[beta==0]=0
init_params_true = list(theta_beta = theta_beta_init,
                        theta_gamma = theta_gamma,
                        theta_eta = data_params$theta_eta,
                        eta = Low_to_high(data_params$theta_eta,basis),
                        delta = delta,
                        sigma_Y = data_params$sigma_Y, 
                        sigma_beta = 1, sigma_eta = 1, sigma_gamma=1)

# read in list of data (smalln)----------------------------------------------------
data_nm = file.path(outpath,paste(datsim_name,"data_multiGP_batch_n",n,"_p",p,"_L",L,sep=""))

data_path_list_sn = NULL
for(b in 1:n_batch){
  data_path_list_sn = c(data_path_list_sn,paste(data_nm,"_b",b,".rds",sep=""))
}

total_batch = length(data_path_list_sn)
batch_size_list = diff(ceiling(seq(1,n+1,length.out = length(data_path_list_sn)+1)))

print(paste("common_percent = ",common_percent))
print(paste("missing_percent = ",missing_percent))
print(paste("contiguous_common = ",contiguous_common))
mask_list_all_sn = get_random_mask(total_batch, batch_size_list,p,n,
                                   common_percent, missing_percent,
                                   contiguous_common,grids)
data_list = NULL

utils::Rprof(Rprof_name,memory.profiling = TRUE)
data_list = read_data_list_to_FBM_impute(data_path_list_sn,mask_list_all_sn$mask_list_fbm,basis)

data_list$mask_list = mask_list_all_sn$mask_list

# get individual imputation list for method_imp_idx first
total_imp = unlist(lapply(mask_list_all_sn$mask_list_fbm, function(x){sum(bigmemory::as.matrix(x)==0)}))

imp_idx_list = get_imp_list(mask_list_all_sn$mask_list_fbm,region_idx)
utils::Rprof(NULL)
p_mem_readin = summaryRprof(filename = Rprof_name,memory = "tseries")
unlink(Rprof_name)




dimensions_sn = list(n=n,L=L, p=p, q=q)
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
  b_end = b_start + dim(bigmemory::as.matrix(mask_list_all_sn$mask_list_fbm[[b]]) )[2] - 1
  batch_idx[[b]] = b_start:b_end
  b_start = b_end + 1
}
batch_idx_cpp_sn = lapply(batch_idx,function(x){x-1})


# set controls ------------------------------------------------------------

## remember to change subsample size accordingly
controls = list(lambda = 0.5, prior_p = 0.5, n_mcmc = 5000,
                start_delta = 0 , subsample_size = subsample_size, step = 1e-3,
                burnin = 0, thinning = 10, interval_eta = 100,start_saving_imp = 1000,
                seed = sample(1:1e5,1))

# sgld with zero imp (smalln) --------------------------------------------------------
init_params0 = list(theta_beta = rep(1,L),
                    theta_gamma = matrix(rep(1,q*L), nrow = L),
                    theta_eta = matrix(0,nrow=L,ncol=dimensions_sn$n),
                    eta = matrix(0,nrow=p,ncol=dimensions_sn$n),
                    delta = rep(1,p),
                    sigma_Y = 1, sigma_beta = 1, sigma_eta = 1, sigma_gamma=1)

utils::Rprof(Rprof_name,memory.profiling = TRUE)
data_list_sn = read_data_list_to_FBM_impute(data_path_list_sn,mask_list_all_sn$mask_list_fbm,basis)
utils::Rprof(NULL)
p_mem_gs_readin = summaryRprof(filename = Rprof_name,memory = "tseries")
unlink(Rprof_name)

# create FBM for theta_eta
theta_eta_path = vector("list",n_batch)
for(b in 1:n_batch){
  print(paste("create theta_eta path for batch ",b))
  batch_idx = batch_idx_cpp_sn[[b]]+1
  theta_eta_path[[b]] = bigmemory::as.big.matrix(init_params0$theta_eta[,batch_idx])@address
}

devtools::load_all()
t0 = Sys.time()
utils::Rprof(Rprof_name_sn, memory.profiling = TRUE)
sgld0_sn = SBIOS0(data_list_sn, basis, theta_eta_path,
                  dimensions_sn,
                  init_params0, region_idx_cpp, L_idx_cpp,
                  batch_idx_cpp_sn, lambda = controls$lambda, prior_p = controls$prior_p,
                  n_mcmc = controls$n_mcmc,  start_delta=controls$start_delta,
                  subsample_size=controls$subsample_size, step = controls$step,
                  burnin = controls$burnin,
                  thinning = controls$thinning,
                  interval_eta = controls$interval_eta,
                  a = 1, b = 1,
                  testing = 0, display_progress = 1)
t1 = Sys.time()
utils::Rprof(NULL)
p_mem_gs_running_new_sn = summaryRprof(filename = Rprof_name_sn,memory = "tseries");unlink(Rprof_name)
sgld0_sn$elapsed = difftime(t1,t0,"secs")
sgld0_sn$mem_tseries = p_mem_gs_running_new_sn
# theta_eta_1 = SBIOS::big_address2mat(theta_eta_path[[1]])
# bigmemory::as.matrix(theta_eta_list[[b]])[1,1]
# theta_eta_1[1,1]
par(mfrow=c(2,2))
plot(sgld0_sn$logLL_mcmc[sgld0_sn$logLL_mcmc!=0], main = paste("logL: seed =",controls$seed))
plot(sgld0_sn$sigma_Y2_mcmc, main = "sigma_Y2_mcmc")
plot(sgld0_sn$sigma_beta2_mcmc, main="sigma_beta2_mcmc")
plot(sgld0_sn$sigma_eta2_mcmc, main="sigma_eta2_mcmc")
par(mfrow=c(1,1))

# utils::Rprof(Rprof_name_sn, memory.profiling = TRUE)
# sgld0_sn_old = method2_SGLD_multiGP_w_eta(data_list_sn, basis,
#                                           dimensions_sn,
#                                           init_params0, region_idx_cpp, L_idx_cpp,
#                                           batch_idx_cpp_sn, lambda = controls$lambda, prior_p = controls$prior_p,
#                                           n_mcmc = controls$n_mcmc,  start_delta=controls$start_delta,
#                                           subsample_size=controls$subsample_size, step = controls$step,
#                                           burnin = controls$burnin,
#                                           thinning = controls$thinning,
#                                           interval_eta = controls$interval_eta,
#                                           all_sgld = 1,
#                                           a_step = 0.001,
#                                           b_step = 10,
#                                           gamma_step = -0.55,
#                                           a = 1, b = 1,
#                                           testing = 0, display_progress = 1)
# 
# utils::Rprof(NULL)
# p_mem_gs_running_old = summaryRprof(filename = Rprof_name_sn,memory = "tseries");unlink(Rprof_name)
# 
# old_mem = get_mem(p_mem_gs_running_old)
# new_mem = get_mem(p_mem_gs_running_new)
# 
# knitr::kable(cbind(new_mem = new_mem$max_mem,
#                    old_mem = old_mem$max_mem),digits=2)
# par(mfrow=c(2,2))
# plot(sgld0_sn_old$logLL_mcmc[sgld0_sn_old$logLL_mcmc!=0], main = paste("logL: seed =",controls$seed))
# plot(sgld0_sn_old$sigma_Y2_mcmc, main = "sigma_Y2_mcmc")
# plot(sgld0_sn_old$sigma_beta2_mcmc, main="sigma_beta2_mcmc")
# plot(sgld0_sn_old$sigma_eta2_mcmc, main="sigma_eta2_mcmc")
# par(mfrow=c(1,1))
# t1 = Sys.time()
# sgld0_sn$controls = controls
# sgld0_sn$elapsed = difftime(t1,t0,units = "secs")
# sgld0_sn$readin_mem = get_mem(p_mem_gs_readin)
# sgld0_sn$running_mem = get_mem(p_mem_gs_running)

# generate and read in list of data (largen)----------------------------------------------------
n=4000; n_batch = 8;
data_params = generate_large_block_multiGP_data_FBM(n,beta,gamma,GP,region_idx,outpath,datsim_name,
                                                    sigma_Y=sigma_Y,
                                                    q = q,n_batch = n_batch)
true_coef = cbind(data_params$beta*data_params$delta,data_params$gamma)


data_nm =file.path(outpath,paste(datsim_name,"data_multiGP_batch_n",n,"_p",p,"_L",L,sep=""))
init_params0 = list(theta_beta = rep(1,L),
                    theta_gamma = matrix(rep(1,q*L), nrow = L),
                    theta_eta = matrix(0,nrow=L,ncol=n),
                    eta = matrix(0,nrow=p,ncol=n),
                    delta = rep(1,p),
                    sigma_Y = 1, sigma_beta = 1, sigma_eta = 1, sigma_gamma=1)





data_nm = file.path(outpath,paste(datsim_name,"data_multiGP_batch_n",n,"_p",p,"_L",L,sep=""))

data_path_list_ln = NULL
for(b in 1:n_batch){
  data_path_list_ln = c(data_path_list_ln,paste(data_nm,"_b",b,".rds",sep=""))
}

total_batch = length(data_path_list_ln)
batch_size_list = diff(ceiling(seq(1,n+1,length.out = length(data_path_list_ln)+1)))

print(paste("common_percent = ",common_percent))
print(paste("missing_percent = ",missing_percent))
print(paste("contiguous_common = ",contiguous_common))
mask_list_all_ln = get_random_mask(total_batch, batch_size_list,p,n,
                                   common_percent, missing_percent,
                                   contiguous_common,grids)
data_list = NULL

utils::Rprof(Rprof_name,memory.profiling = TRUE)
data_list = read_data_list_to_FBM_impute(data_path_list_ln,mask_list_all_ln$mask_list_fbm,basis)

data_list$mask_list = mask_list_all_ln$mask_list

# get individual imputation list for method_imp_idx first
total_imp = unlist(lapply(mask_list_all_ln$mask_list_fbm, function(x){sum(bigmemory::as.matrix(x)==0)}))

imp_idx_list = get_imp_list(mask_list_all_ln$mask_list_fbm,region_idx)
utils::Rprof(NULL)
p_mem_readin = summaryRprof(filename = Rprof_name,memory = "tseries")
unlink(Rprof_name)




dimensions_ln = list(n=n,L=L, p=p, q=q)
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
  b_end = b_start + dim(bigmemory::as.matrix(mask_list_all_ln$mask_list_fbm[[b]]) )[2] - 1
  batch_idx[[b]] = b_start:b_end
  b_start = b_end + 1
}
batch_idx_cpp_ln = lapply(batch_idx,function(x){x-1})

# sgld with zero imp (largen) --------------------------------------------------------
init_params0 = list(theta_beta = rep(1,L),
                    theta_gamma = matrix(rep(1,q*L), nrow = L),
                    theta_eta = matrix(0,nrow=L,ncol=dimensions_ln$n),
                    eta = matrix(0,nrow=p,ncol=dimensions_ln$n),
                    delta = rep(1,p),
                    sigma_Y = 1, sigma_beta = 1, sigma_eta = 1, sigma_gamma=1)

utils::Rprof(Rprof_name,memory.profiling = TRUE)
data_list_ln = read_data_list_to_FBM_impute(data_path_list_ln,mask_list_all_ln$mask_list_fbm,basis)
utils::Rprof(NULL)
p_mem_gs_readin = summaryRprof(filename = Rprof_name,memory = "tseries")
unlink(Rprof_name)

# create FBM for theta_eta
theta_eta_path = vector("list",n_batch)
for(b in 1:n_batch){
  print(paste("create theta_eta path for batch ",b))
  batch_idx = batch_idx_cpp_ln[[b]]+1
  theta_eta_path[[b]] = bigmemory::as.big.matrix(init_params0$theta_eta[,batch_idx])@address
}

devtools::load_all()
t0 = Sys.time()
utils::Rprof(Rprof_name_ln, memory.profiling = TRUE)
sgld0_ln = SBIOS0(data_list_ln, basis, theta_eta_path,
                  dimensions_ln,
                  init_params0, region_idx_cpp, L_idx_cpp,
                  batch_idx_cpp_ln, lambda = controls$lambda, prior_p = controls$prior_p,
                  n_mcmc = controls$n_mcmc,  start_delta=controls$start_delta,
                  subsample_size=controls$subsample_size, step = controls$step,
                  burnin = controls$burnin,
                  thinning = controls$thinning,
                  interval_eta = controls$interval_eta,
                  a = 1, b = 1,
                  testing = 0, display_progress = 1)
t1 = Sys.time()
utils::Rprof(NULL)
p_mem_gs_running_new_ln = summaryRprof(filename = Rprof_name_ln,memory = "tseries");unlink(Rprof_name)
sgld0_ln$mem_tseries = p_mem_gs_running_new_ln
sgld0_ln$elapsed = difftime(t1,t0,"secs")
# theta_eta_1 = SBIOS::big_address2mat(theta_eta_path[[1]])
# bigmemory::as.matrix(theta_eta_list[[b]])[1,1]
# theta_eta_1[1,1]
par(mfrow=c(2,2))
plot(sgld0_ln$logLL_mcmc[sgld0_ln$logLL_mcmc!=0], main = paste("logL: seed =",controls$seed))
plot(sgld0_ln$sigma_Y2_mcmc, main = "sigma_Y2_mcmc")
plot(sgld0_ln$sigma_beta2_mcmc, main="sigma_beta2_mcmc")
plot(sgld0_ln$sigma_eta2_mcmc, main="sigma_eta2_mcmc")
par(mfrow=c(1,1))



# analyze_result ----------------------------------------------------------
zeroimp_sn = sum_stats(sgld0_sn,data_params$beta*data_params$delta,"SGLD-zeroimp",
                       cutoff = 0.9,
                       fdr_control = "ip_cutoff")
zeroimp_ln = sum_stats(sgld0_ln,data_params$beta*data_params$delta,"SGLD-zeroimp",
                       cutoff = 0.9,
                       fdr_control = "ip_cutoff")

df = as.data.frame(rbind(zeroimp_sn$result,zeroimp_ln$result))
rownames(df) = c("zeroimp_sn","zeroimp_ln")
knitr::kable(df)
# summary(data_params$snratio)

beta = data_params$beta*data_params$delta



ip_list = list(zeroimp_sn = zeroimp_sn$ip$InclusionProb,
               zeroimp_ln = zeroimp_ln$ip$InclusionProb)
ROC_all = plot_ROC(ip_list,beta,len=50)
# hist(data_params$snratio)

df <- data.frame(x=c(ROC_all$FPR), val=c(ROC_all$TPR), 
                 method=rep(names(ip_list), each=dim(ROC_all$TPR)[1]))

# compute AUC
n_method = levels(as.factor(df$method))
AUC = rep(NA,length(n_method)); names(AUC) = n_method
for(i in 1:length(n_method)){
  sens = ROC_all$TPR[,i]
  omspec = ROC_all$FPR[,i]
  height = (sens[-1]+sens[-length(sens)])/2
  width = abs(-diff(omspec)) # = diff(rev(omspec))
  AUC[i] = sum(height*width)
}
# knitr::kable( t(AUC),digits = 3)

# compute controled TPR when FPR = 0.1
TPRcon = rep(NA,length(n_method)); names(AUC) = n_method
for(i in 1:length(n_method)){
  sens = ROC_all$TPR[,i]
  omspec = ROC_all$FPR[,i]
  if(n_method[i]=="MUA"){
    x = c(omspec <= 0.1) - c(omspec >= 0.1)
  }else{
    x = c(omspec >= 0.1) - c(omspec <= 0.1)
  }
  
  idx = c(order(x)[1],order(x)[1]-1)
  slope = (sens[idx[2]] - sens[idx[1]])/(omspec[idx[2]] - omspec[idx[1]])
  TPRcon[i] = slope*(0.1 - omspec[idx[1]]) + sens[idx[1]]
}
# knitr::kable( t(TPRcon),digits = 3)

# compute memory
par(mfrow = c(1,2))
mem_sn = get_mem(sgld0_sn$mem_tseries,bool_plot=T,main="mem_sn")
mem_ln = get_mem(sgld0_ln$mem_tseries,bool_plot=T,main="mem_ln")
par(mfrow = c(1,1))
mem_all = rbind(mem_ln , mem_sn)
sum_stats = as.data.frame(t(rbind(AUC = AUC,
                                  TPRcon = TPRcon) ))

all_result = cbind(sum_stats,mem_all)

knitr::kable(t(all_result),digits = 3)
outname = paste("./sim_result/",sim_name,"_n",n,"_p",p,"_mis",missing_percent,"_sigY",sigma_Y,"_sim",n_rep,sep="")
saveRDS(all_result,paste(outname,".rds",sep=""))

# unlink data when finished
for(dname in data_path_list_ln){
  unlink(dname)
}
for(dname in data_path_list_sn){
  unlink(dname)
}
unlink(file.path(outpath,paste(sim_name,"data_multiGP_batch_n",n,"_p",p,"_L",L,"params.rds",sep="")))
unlink(file.path(outpath,paste(sim_name,"data_multiGP_batch_n",2*n,"_p",p,"_L",L,"params.rds",sep="")))