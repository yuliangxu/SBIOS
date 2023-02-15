common_percent = 0.9
missing_percent = 0.5
contiguous_common = T
library(BayesGPfit)
library(ggplot2)
library(viridis)
library("profmem")


# create a testing case ---------------------------------------------------


n=3000

# source(file.path("./R","summary_functions.R"))
# source(file.path("./R","generate_data_FBM2.R"))
num_region = 9
side = 120 # > 50*50
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

# beta = readRDS(file.path(g_dir,paste("./data/drawing/circle_",side,".rds")))

center = apply(grids,2,mean)
rad = apply(grids,1,function(x){sum((x-center)^2)})
inv_rad = 2-rad
inv_rad_ST = Soft_threshold(inv_rad,1.2)

# beta = log(inv_rad_ST^2+1) + rnorm(p,sd=0.1)
beta = log(inv_rad_ST^2+1)
plot_img(beta,grids_df,"true beta")
# beta_ST = log(inv_rad_ST^2+1)
# plot_img(beta_ST,grids_df,"true beta_ST")

# beta = 1*(inv_rad_ST>0.5)
# plot_img(beta,grids_df,"true beta")
# beta_ST = beta 

l = round(p/num_region*0.1)
GP = generate_matern_basis2(grids, region_idx, rep(l,length(region_idx)),scale = 2,nu = 1/5,
                            show_progress=T)


L = sum(GP$L_all)



# generate gamma from basis
q = 4
theta_gamma = matrix(rnorm(L*q), nrow = L)
gamma = Low_to_high(theta_gamma,GP)

basis_path = file.path(paste("./data/data_basis_n",n,"_p",p,"_L",L,".rds",sep=""))
saveRDS(GP, basis_path)
outpath = file.path("./data")
data_params = generate_large_block_multiGP_data_FBM(n,beta,gamma,GP,region_idx,outpath,
                                                    sigma_Y=1,
                                                    q = q,n_batch = 6)
plot_img(data_params$beta*data_params$delta,grids_df,"true beta")
hist(data_params$snratio)
true_coef = cbind(data_params$beta*data_params$delta,data_params$gamma)


data_nm =file.path(paste("./data/data_multiGP_batch_n",n,"_p",p,"_L",L,sep=""))

data_path_list = c(paste(data_nm,"_b1.rds",sep=""),
                   paste(data_nm,"_b2.rds",sep=""),
                   paste(data_nm,"_b3.rds",sep=""),
                   paste(data_nm,"_b4.rds",sep=""),
                   paste(data_nm,"_b5.rds",sep=""),
                   paste(data_nm,"_b6.rds",sep=""))

basis = readRDS(basis_path)
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

# read in list of data ----------------------------------------------------
total_batch = length(data_path_list)
batch_size_list = diff(ceiling(seq(1,n+1,length.out = length(data_path_list)+1)))



print(paste("common_percent = ",common_percent))
print(paste("missing_percent = ",missing_percent))
print(paste("contiguous_common = ",contiguous_common))
mask_list_all = get_random_mask(total_batch, batch_size_list,p,n,
                                common_percent, missing_percent,
                                contiguous_common,grids)
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
  b_end = b_start + dim(data_list$Y_fbm_list[[b]])[2] - 1
  batch_idx[[b]] = b_start:b_end
  b_start = b_end + 1
}
batch_idx_cpp = lapply(batch_idx,function(x){x-1})

options(profmem.threshold = 2000)
p_mem_readin <- profmem({ 
  data_list = read_data_list_to_FBM_impute(data_path_list,mask_list_all$mask_list_fbm)
  
  # data_list$Y_star_list = get_Y_star(data_list$Y_fbm_list,basis,basis$region_idx_block)
  data_list$mask_list = mask_list_all$mask_list
  
  # get individual imputation list for method_imp_idx first
  unlist(lapply(mask_list_all$mask_list_fbm, function(x){table(as.matrix(x))}))
  total_imp = unlist(lapply(mask_list_all$mask_list_fbm, function(x){sum(as.matrix(x)==0)}))
  
  imp_idx_list = get_imp_list(mask_list_all$mask_list_fbm,region_idx)
  
})
utils:::format.object_size(sum(p_mem_readin$bytes,na.rm = T), "auto")
utils:::format.object_size(max(p_mem_readin$bytes,na.rm = T), "auto")
# # run MUA -----------------------------------------------------------------


Y_allsample = NULL
X_allsample = NULL
mask_allsample = NULL
options(profmem.threshold = 2000)
p_mem_MUA_readin <- profmem({
b_counter = 0
for(dpath in data_path_list){
    print(paste("processing data set  ",dpath))
    b_counter = b_counter + 1
    data_b = readRDS(dpath)
    Y_allsample = cbind(Y_allsample,data_b$Y)
    X_allsample = cbind(X_allsample,data_b$X)
    mask_allsample = cbind(mask_allsample, extract_mask( mask_list_all$mask_list[[b_counter]] ) )
  }



  Y_allsample = Y_allsample * mask_allsample

})
p_mem_MUA_running <- profmem({
  p = dim(Y_allsample)[1]
  q = dim(X_allsample)[1]-1
  pb = txtProgressBar(min = 0, max = p, initial = 0)
  mua_coef = matrix(NA,ncol = q+1, nrow = p)
  pvalue_MUA = matrix(NA,ncol = q+1, nrow = p)
  X_allsample_t = t(X_allsample)
  # eta_init = matrix(0,n,p)
  t0 = Sys.time()
  for( j in 1:p){
    Y_j = Y_allsample[j,]
    X_j = X_allsample_t
    m = lm(Y_j ~ X_j  - 1)
    mua_coef[j,] = coef(m)
    # eta_init[mask_j==TRUE,j] = Y_j - X_j%*%coef(m)
    pvalue_MUA[j,] = summary(m)$coefficients[,4]
    setTxtProgressBar(pb,j)
  }
  t1 =  Sys.time()
})


MUA = NULL
MUA$coef = mua_coef
MUA$p_value = pvalue_MUA
MUA$time = difftime(t1,t0,units = "secs")
MUA$total_mem = list(readin = utils:::format.object_size(sum(p_mem_MUA_readin$bytes,na.rm = T), "auto"),
  running = utils:::format.object_size(sum(p_mem_MUA_running$bytes,na.rm = T), "auto") )
MUA$max_mem = list(readin = utils:::format.object_size(max(p_mem_MUA_readin$bytes,na.rm = T), "auto"),
                     running = utils:::format.object_size(max(p_mem_MUA_running$bytes,na.rm = T), "auto") )

MUA$result = mua_sum_stats(MUA,beta,ip_thresh = 0.01)
MUA$snratio = snratio(X_allsample,MUA$coef,Y_allsample)
knitr::kable(t(MUA$result))
hist(MUA$snratio)


# beta = beta_ST
# set controls ------------------------------------------------------------

controls = list(lambda = 0.5, prior_p = 0.5, n_mcmc = 5000,
                start_delta = 0 , subsample_size = 400, step = 1e-3,
                burnin = 0, thinning = 10, interval_eta = 100,start_saving_imp = 1000,
                seed = sample(1:1e5,1))

# GS with zero imp --------------------------------------------------------

# Rcpp::sourceCpp(file.path(g_dir,"src/method2_gs_noMem.cpp"))
p_mem_gs_readin <- profmem({
  data_list_gs = read_data_list_to_gs(data_path_list,mask_list_all$mask_list_fbm)
})

t0 = Sys.time()
p_mem_gs_running <- profmem({
  
  gs0 = method2_gs_no_mem(data_list_gs, basis,
                                     dimensions,
                                     init_params0, region_idx_cpp, L_idx_cpp,
                                     batch_idx_cpp, lambda = controls$lambda, prior_p = controls$prior_p,
                                     n_mcmc = controls$n_mcmc,  start_delta=controls$start_delta,
                                     subsample_size=controls$subsample_size, step = controls$step,
                                     burnin = controls$burnin,
                                     thinning = controls$thinning,
                                     interval_eta = controls$interval_eta,
                                     a = 1, b = 1,
                                     testing = 0, display_progress = 1)
})
t1 = Sys.time()
gs0$controls = controls
gs0$elapsed = difftime(t1,t0,units = "secs")
gs0$total_mem = list(readin = utils:::format.object_size(sum(p_mem_gs_readin$bytes,na.rm = T), "auto"),
               running = utils:::format.object_size(sum(p_mem_gs_running$bytes,na.rm = T), "auto") )
gs0$max_mem = list(readin = utils:::format.object_size(max(p_mem_gs_readin$bytes,na.rm = T), "auto"),
                     running = utils:::format.object_size(max(p_mem_gs_running$bytes,na.rm = T), "auto") )


par(mfrow = c(2,4))
plot(gs0$logLL_mcmc[gs0$logLL_mcmc!=0], main = paste("logL: seed =",controls$seed))
plot(gs0$sigma_Y2_mcmc, main = "sigma_Y2_mcmc")
plot(gs0$sigma_beta2_mcmc, main="sigma_beta2_mcmc")
plot(gs0$sigma_eta2_mcmc, main="sigma_eta2_mcmc")

j=1;
plot(gs0$theta_beta_mcmc[j,], main = "theta_beta_mcmc[1,]");
abline(v=controls$start_sigma,col="red")
# abline(h=true_theta_beta[j],col="blue")
burnin = as.integer(((0.8*controls$n_mcmc):(controls$n_mcmc-1))/controls$thinning)
burnin = burnin[!duplicated(burnin)]
beta_mcmc = Low_to_high( gs0$theta_beta_mcmc[,burnin],GP)
beta = apply(beta_mcmc * gs0$delta_mcmc[,burnin],1,mean)
plot(beta, data_params$beta,main="beta_est vs beta_true",
     xlab = "beta_est",ylab = "beta_true");abline(0,1,col="red")
hist(data_params$snratio)
ip = apply(gs0$delta_mcmc[,burnin],1,function(x){mean(x!=0)})
hist(ip)
par(mfrow = c(1,1))




# run zero imp ------------------------------------------------------------

# Rcpp::sourceCpp(file.path(g_dir,"src/method2_final_zero_imp_allSGLD.cpp"))
data_list_gs = read_data_list_to_gs

t0 = Sys.time()
options(profmem.threshold = 2000)
p_mem_sgld0 <- profmem({ 
  sgld0 = method2_SGLD_multiGP_w_eta(data_list, basis,
                                     dimensions,
                                     init_params0, region_idx_cpp, L_idx_cpp,
                                     batch_idx_cpp, lambda = controls$lambda, prior_p = controls$prior_p,
                                     n_mcmc = controls$n_mcmc,  start_delta=controls$start_delta,
                                     subsample_size=controls$subsample_size, step = controls$step,
                                     burnin = controls$burnin,
                                     thinning = controls$thinning,
                                     interval_eta = controls$interval_eta,
                                     all_sgld = 1,
                                     a_step = 0.001,
                                     b_step = 10,
                                     gamma_step = -0.55,
                                     a = 1, b = 1,
                                     testing = 0, display_progress = 1)
})
t1 = Sys.time()
sgld0$controls = controls
sgld0$elapsed = difftime(t1,t0,units = "secs")
sgld0$total_mem = list(readin = utils:::format.object_size(sum(p_mem_readin$bytes,na.rm = T), "auto"),
                 running = utils:::format.object_size(sum(p_mem_sgld0$bytes,na.rm = T), "auto") )
sgld0$max_mem = list(readin = utils:::format.object_size(max(p_mem_readin$bytes,na.rm = T), "auto"),
                       running = utils:::format.object_size(max(p_mem_sgld0$bytes,na.rm = T), "auto") )


par(mfrow = c(2,4))
plot(sgld0$logLL_mcmc[sgld0$logLL_mcmc!=0], main = paste("logL: seed =",controls$seed))
plot(sgld0$sigma_Y2_mcmc, main = "sigma_Y2_mcmc")
plot(sgld0$sigma_beta2_mcmc, main="sigma_beta2_mcmc")
plot(sgld0$sigma_eta2_mcmc, main="sigma_eta2_mcmc")

j=1;
plot(sgld0$theta_beta_mcmc[j,], main = "theta_beta_mcmc[1,]"); 
abline(v=controls$start_sigma,col="red")
# abline(h=true_theta_beta[j],col="blue")
burnin = as.integer(((0.8*controls$n_mcmc):(controls$n_mcmc-1))/controls$thinning)
burnin = burnin[!duplicated(burnin)]
beta_mcmc = Low_to_high( sgld0$theta_beta_mcmc[,burnin],GP)
beta = apply(beta_mcmc * sgld0$delta_mcmc[,burnin],1,mean)
plot(beta, data_params$beta,main="beta_est vs beta_true",
     xlab = "beta_est",ylab = "beta_true");abline(0,1,col="red")
hist(data_params$snratio)
ip = apply(sgld0$delta_mcmc[,burnin],1,function(x){mean(x!=0)})
hist(ip)
par(mfrow = c(1,1))

zeroimp = sum_stats(sgld0,data_params$beta*data_params$delta,"SGLD-zeroimp",
                    cutoff = 0.9,
                    fdr_control = "ip_cutoff")
knitr::kable(t(zeroimp$result))


# run idx imp ------------------------------------------------------------

# Rcpp::sourceCpp(file.path(g_dir,"src/method2_final_idx_imp_allSGLD.cpp"))

t0 = Sys.time()
options(profmem.threshold = 2000)
p_mem_sgld <- profmem({ 
  sgld = method2_SGLD_multiGP_impute_idx_fixeta(data_list, basis,
                                               dimensions, imp_idx_list, total_imp,
                                               init_params0, region_idx_cpp, L_idx_cpp,
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
                                               a_step = 0.01,
                                               b_step = 10,
                                               gamma_step = -0.55,
                                               update_individual_effect = 1,
                                               a = 1, b = 1,
                                               testing = 0, display_progress = 1)
})

t1 = Sys.time()
t1-t0
sgld$elapsed = difftime(t1,t0,units = "secs")
sgld$controls = controls
sgld$total_mem = list(readin = utils:::format.object_size(sum(p_mem_readin$bytes,na.rm = T), "auto"),
                running = utils:::format.object_size(sum(p_mem_sgld$bytes,na.rm = T), "auto") )
sgld$max_mem = list(readin = utils:::format.object_size(max(p_mem_readin$bytes,na.rm = T), "auto"),
                      running = utils:::format.object_size(max(p_mem_sgld$bytes,na.rm = T), "auto") )


par(mfrow = c(2,4))
plot(sgld$logLL_mcmc[sgld$logLL_mcmc!=0], main = paste("logL: seed =",controls$seed))
plot(sgld$sigma_Y2_mcmc, main = "sigma_Y2_mcmc")
plot(sgld$sigma_beta2_mcmc, main="sigma_beta2_mcmc")
plot(sgld$sigma_eta2_mcmc, main="sigma_eta2_mcmc")

j=1;
plot(sgld$theta_beta_mcmc[j,], main = "theta_beta_mcmc[1,]"); 
burnin = as.integer(((0.8*controls$n_mcmc):(controls$n_mcmc-1))/controls$thinning)
burnin = burnin[!duplicated(burnin)]
beta_mcmc = Low_to_high( sgld$theta_beta_mcmc[,burnin],GP)
# beta_mcmc = basis$Phi_Q[[region_r]] %*% sgld$theta_beta_mcmc[,burnin]
beta = apply(beta_mcmc * sgld$delta_mcmc[,burnin],1,mean)
plot(beta, data_params$beta,main="beta_est vs beta_true",
     xlab = "beta_est",ylab = "beta_true");abline(0,1,col="red")
hist(data_params$snratio)
ip = apply(sgld0$delta_mcmc[,burnin],1,function(x){mean(x!=0)})
hist(ip)
par(mfrow = c(1,1))

# analyze_result ----------------------------------------------------------
zeroimp = sum_stats(sgld0,data_params$beta*data_params$delta,"SGLD-zeroimp",
                    cutoff = 0.9,
                    fdr_control = "ip_cutoff")
gsimp = sum_stats(sgld,data_params$beta*data_params$delta,"SGLD-gsimp",
                    cutoff =  0.9,
                    fdr_control = "ip_cutoff")
gszero = sum_stats(gs0,data_params$beta*data_params$delta,"SGLD-zeroimp",
                   cutoff = 0.9,
                   fdr_control = "ip_cutoff")

df = as.data.frame(rbind(gszero$result,zeroimp$result,gsimp$result))
df$mem_readin = c(gszero$mem$readin,zeroimp$mem$readin, gsimp$mem$readin)
df$mem_running = c(gszero$mem$running,zeroimp$mem$running, gsimp$mem$running)
rownames(df) = c("gszero","zero-imp","gs-imp")
knitr::kable(df)
summary(data_params$snratio)

beta = data_params$beta*data_params$delta

plot_ROC = function(ip_list, beta, len=20){
  thresh_grid = seq(0,1,length.out = len)
  FPR_all = NULL; TPR_all = NULL
  
  for(i in 1:length(ip_list)){
    ip = ip_list[[i]]
    TPR = rep(NA,len)
    FPR = rep(NA,len)
    
    if(names(ip_list)[i] == "MUA"){
      p_adj = p.adjust(ip,"BH")
      p_adj_thresh = quantile(p_adj,probs = thresh_grid)
      
      for(l in 1:len){
        res = as.matrix(table(p_adj <= p_adj_thresh[l], beta!=0))
        TPR[l] = res["TRUE","TRUE"]/sum(res[,"TRUE"]) # TP/P
        FPR[l] = res["TRUE","FALSE"]/sum(res[,"TRUE"]) # FP/P
      }
      TPR = c(TPR,1)
      FPR = c(FPR,1)
    }else{
      # ip_thresh_grid = quantile(ip,probs = thresh_grid)
      ip_thresh_grid = thresh_grid
      for(l in 1:len){
        thresh = ip_thresh_grid[l]
        res = as.matrix(table(ip>=thresh, beta!=0))
        if(! "TRUE" %in% rownames(res)){
          res = as.matrix(rbind(res,c(0,0)))
          rownames(res) = c("FALSE","TRUE")
        }
        TPR[l] = res["TRUE","TRUE"]/sum(res[,"TRUE"]) # TP/P
        FPR[l] = res["TRUE","FALSE"]/sum(res[,"TRUE"]) # FP/P
        
      }
      TPR = c(1,TPR)
      FPR = c(1,FPR)
    }
    
    TPR_all = cbind(TPR_all, TPR)
    FPR_all = cbind(FPR_all, FPR)
    
  }
  
  return(list(FPR = FPR_all,TPR= TPR_all))
  
}





ip_list = list(MUA = MUA$p_value[,1],
               gs = gszero$ip$InclusionProb,
               sgld0 = zeroimp$ip$InclusionProb,
               sgld_imp = gsimp$ip$InclusionProb)
ROC_all = plot_ROC(ip_list,beta,len=50)
hist(data_params$snratio)

df <- data.frame(x=c(ROC_all$FPR), val=c(ROC_all$TPR), 
                 method=rep(names(ip_list), each=dim(ROC_all$TPR)[1]))
ggplot(data = df, aes(x=x, y=val)) + geom_line(aes(colour=method)) + 
  ggtitle("ROC") + xlab("FPR") + ylab("TPR") 

# compute AUC
AUC = rep(NA,4); names(AUC) = c("MUA","gs","sgld0","sgld_imp")
for(i in 1:4){
  sens = ROC_all$TPR[,i]
  omspec = ROC_all$FPR[,i]
  height = (sens[-1]+sens[-length(sens)])/2
  width = abs(-diff(omspec)) # = diff(rev(omspec))
  AUC[i] = sum(height*width)
}
knitr::kable( t(AUC),digits = 3)

# compute controled TPR when FPR = 0.1
TPRcon = rep(NA,4); names(TPRcon) = c("MUA","gs","sgld0","sgld_imp")
for(i in 1:4){
  sens = ROC_all$TPR[,i]
  omspec = ROC_all$FPR[,i]
  if(i==1){
    x = c(omspec <= 0.1) - c(omspec >= 0.1)
  }else{
    x = c(omspec >= 0.1) - c(omspec <= 0.1)
  }
  
  idx = c(order(x)[1],order(x)[1]-1)
  slope = (sens[idx[2]] - sens[idx[1]])/(omspec[idx[2]] - omspec[idx[1]])
  TPRcon[i] = slope*(0.1 - omspec[idx[1]]) + sens[idx[1]]
}
knitr::kable( t(TPRcon),digits = 3)


mem_readin = utils:::format.object_size(sum(p_mem_readin$bytes,na.rm = T), "auto")
c(MUA$mem,gszero$mem,zeroimp$mem, gsimp$mem,mem_readin)


sum_stats = as.data.frame(t(rbind(AUC = AUC,
                                  TPRcon = TPRcon,
                  time = c(MUA$time,gszero$result["time"],zeroimp$result["time"], gsimp$result["time"]))
))
sum_stats$total_mem_readin = c(MUA$total_mem$readin,gszero$total_mem$readin,zeroimp$total_mem$readin, gsimp$total_mem$readin)
sum_stats$total_mem_running = c(MUA$total_mem$running,gszero$total_mem$running,zeroimp$total_mem$running, gsimp$total_mem$running)
sum_stats$max_mem_readin = c(MUA$max_mem$readin,gszero$max_mem$readin,zeroimp$max_mem$readin, gsimp$max_mem$readin)
sum_stats$max_mem_running = c(MUA$max_mem$running,gszero$max_mem$running,zeroimp$max_mem$running, gsimp$max_mem$running)

knitr::kable(sum_stats,digits = 3)
print(paste("readin memory for FBM:",mem_readin))

saveRDS(sum_stats,"./sim_result/sum_stats.rds")