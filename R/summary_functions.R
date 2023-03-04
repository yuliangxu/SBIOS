plot_check = function(sgld3){
  par(mfrow=c(1,3))
  plot(sgld3$logLL_mcmc[sgld3$logLL_mcmc!=0],ylab = "logLL_mcmc",type="l")
  plot(sgld3$sigma_Y2_mcmc[sgld3$sigma_Y2_mcmc!=0],ylab = "sigma_Y2_mcmc",type="l")
  plot(sgld3$theta_beta_mcmc[1,],type="l")
  par(mfrow=c(1,1))
}
sum_stats = function(sgld,beta,main,fdr_target=0.1,size_thresh = 0.01,
                     ip_quantile = 0.3, cutoff = 0.5,
                     fdr_control = 1){
  burnin = (0.8*dim(sgld$theta_beta_mcmc)[2]):dim(sgld$theta_beta_mcmc)[2]
  beta_mcmc = Low_to_high(as.matrix(sgld$theta_beta_mcmc[,burnin]), basis)
  beta_mcmc = beta_mcmc * sgld$delta_mcmc[,burnin]
  beta_mean = apply(beta_mcmc,1,mean)
  ip = NULL
  # FDR control 1: Jeff Morris - cutoff on beta
  if(fdr_control =="Morris" ){
    p = length(beta_mean)
    iprob = apply(beta_mcmc,1,function(x){mean(abs(x)<size_thresh)})
    # hist(iprob_sort)
    iprob_sort = sort(iprob)
    l = order(which(cumsum(iprob_sort)/(1:p) < fdr_target),
              decreasing = T)[1]
    if(is.na(l)){l=1}
    prob_thresh = iprob_sort[l]
    ggplot(grids_df) + geom_point(aes(x1, x2, color = iprob),shape=15,size=10) + scale_color_viridis()
    
    ggplot(grids_df) + geom_point(aes(x1, x2, color = 1*(iprob<prob_thresh)),shape=15,size=10) + scale_color_viridis()
    
    # ip_thresh = quantile(iprob,fdr_target)
    ip = NULL
    ip$InclusionProb = iprob
    ip$size_thresh = size_thresh
    ip$prob_thresh = prob_thresh
    ip$mapping = 1*(iprob<=prob_thresh)
  }
  
  
  # FDR control 2: cutoff on ip
  if( fdr_control == "ip_cutoff"){
    # ggplot(grids_df) + geom_point(aes(x1, x2, color = ip),shape=15,size=10) + scale_color_viridis()
    # ip = InclusionMap(beta_mcmc, beta, fdr_target = fdr_target)
    # if(!ip$tuning){
    #   print("cannot achieve target FDR, choose 0.5 as cutoff.")
    #   ip$mapping = 1*(apply(beta_mcmc,1,function(x){mean(x!=0)})>0.5)
    # }
    
    ip$InclusionProb = apply(beta_mcmc,1,function(x){mean(x!=0)})
    ip$mapping = 1*(ip$InclusionProb>=cutoff)
  }
  
  # FDR control 3: directly thresholding on quantile of ip
  if( fdr_control == "quantile_cutoff"){
    ip = NULL
    ip$InclusionProb = apply(beta_mcmc,1,function(x){mean(x!=0)})
    ip$prob_thresh = quantile(ip$InclusionProb, ip_quantile)
    ip$mapping = 1*(ip$InclusionProb > ip$prob_thresh)
    print(paste("ip$prob_thresh = ",ip$prob_thresh))
  }
  
  # ggplot(grids_df) + geom_point(aes(x1, x2, color = ip$mapping),shape=15,size=10) + scale_color_viridis()
  result = rep(NA,6)
  names(result) = c("FDR","Power","Accuracy","MSE_nonnull","MSE_null","time")
  result[1] = FDR(ip$mapping, beta)
  result[2] = Power(ip$mapping, beta)
  result[3] = Precision(ip$mapping, beta)
  beta_mapped = rep(0,p)
  beta_mapped[ip$mapping==1] = beta_mean[ip$mapping==1]
  # ggplot(grids_df) + geom_point(aes(x1, x2, color = beta_mapped),shape=15,size=10) + scale_color_viridis()
  mse = function(x,y){mean((x-y)^2)}
  result[4] = mse(beta_mapped[abs(beta)>0],beta[abs(beta)>0])
  result[5] = mse(beta_mapped[beta==0],beta[beta==0])
  result[6] = sgld$elapsed
  
  
  # get R2
  theta_gamma_mean = apply(sgld$theta_gamma_mcmc[, ,burnin],c(1,2),mean)
  gamma_mean = Low_to_high(theta_gamma_mean, basis)
  
  output = NULL
  output$result = result
  output$beta_mean = beta_mean
  output$ip = ip
  output$gamma_mean = gamma_mean
  if(!is.null(sgld$mem)){
    output$mem = sgld$mem
  }
  return(output)
}

mua_sum_stats = function(MUA,beta,ip_thresh = 0.01){
  p_adj = p.adjust(MUA$p_value[,1],"BH")
  MUA$mapping = 1*(p_adj < ip_thresh)
  beta_mean = MUA$coef[,1]
  result = rep(NA,6)
  names(result) = c("FDR","Power","Accuracy","MSE_nonnull","MSE_null","time")
  result[1] = FDR(MUA$mapping, beta)
  result[2] = Power(MUA$mapping, beta)
  result[3] = Precision(MUA$mapping, beta)
  beta_mapped = rep(0,p)
  beta_mapped[MUA$mapping==1] = beta_mean[MUA$mapping==1]
  # ggplot(grids_df) + geom_point(aes(x1, x2, color = beta_mapped),shape=15,size=10) + scale_color_viridis()
  mse = function(x,y){mean((x-y)^2)}
  result[4] = mse(beta_mapped[abs(beta)>0],beta[abs(beta)>0])
  result[5] = mse(beta_mapped[beta==0],beta[beta==0])
  result[6] = MUA$time
  return(result)
}
snratio = function(X_mat,coef,Y){
  Y_hat = coef%*%X_mat
  apply(Y_hat,1,var)/apply(Y_hat-apply(Y,1,mean),1,function(X){sum(X^2)})
}

get_R2 = function(Y_allsample, X_allsample, coef_mat){
  
  p = dim(Y_allsample)[1]
  # requires large memory
  # Y_pred = coef_mat %*% X_allsample
  # SSR = apply(Y_allsample - Y_pred,1,function(x){sum(x^2)})
  # SST = apply(Y_allsample,1,function(x){sum((x-mean(x))^2)})
  
  # or memory-saving
  SSR = SST = rep(NA,p)
  for(j in 1:p){
    res_j = coef_mat[j,] %*% X_allsample
    SSR[j] = sum((res_j - Y_allsample[j,])^2)
    SST[j] = sum((Y_allsample[j,] - mean(Y_allsample[j,]))^2)
  }
  return(1-SSR/SST)
  
}
get_imp_MSE = function(sgld3,mask_list_all,data_list,plot_b1=T){
  total_batch = length(mask_list_all$mask_list_fbm )
  mse = rep(NA,total_batch)
  mse0 = rep(NA,total_batch)
  for(b in 1:total_batch){
    Y = data_list$Y_true_fbm[[b]][]
    Y_imp_true = get_Y_imp_from_true(basis,L_idx_cpp,region_idx_cpp,imp_idx_list,Y,b=b)
    Y_imp_b = sgld3$Y_imp[[b]]
    # Y_imp_b = sgld3$Y_imp_mean[[b]]
    mse[b] = mean((Y_imp_true - Y_imp_b)^2)
    mse0[b] = mean((Y_imp_true)^2)
  }
  if(plot_b1){
    b=1
    Y = data_list$Y_true_fbm[[b]][]
    # Y_true = Y[as.matrix(mask_list_all$mask_list_fbm[[b]])==0] 
    Y_imp_true = get_Y_imp_from_true(basis,L_idx_cpp,region_idx_cpp,imp_idx_list,Y,b=b)
    plot(sgld3$Y_imp[[b]],Y_imp_true,asp=1)
    abline(0,1,col="red")
  }
  
  return(list(mse_imp  = mse, mse0 = mse0))
}
get_mse=function(x,y){
  return(mean((x-y)^2))
}

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




