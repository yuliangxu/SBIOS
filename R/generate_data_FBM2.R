library(MASS) # generate multivariate normal sample
library(BayesGPfit) # compute grids
library(Matrix) # compute matrix rank
library(bigmemory)
library(bigstatsr)
library(mvtnorm)
library(RSpectra)
# ====Parameter Setup====
get_mem = function(p_mem,sum_method = "tseries", bool_plot=F,main="mem"){
  # when using Rprof with summaryRprof(filename = "Rprof.out",memory = "both")
  if(sum_method == "both"){
    return(list(total_mem = sum(p_mem$by.self$mem.total), 
                max_mem = max(p_mem$by.self$mem.total)) )
  }
  if(sum_method == "tseries"){
    all_mem = apply(p_mem[,1:3],1,sum)*1e-6
    if(bool_plot){
      rname = as.numeric(rownames(p_mem))
      plot(rname, all_mem,main = main)
    }
    return( as.data.frame( cbind(total_mem = sum(all_mem), 
                max_mem = max(all_mem)) ) )
  }
  
  
}

get_imp_list_RDA = function(imp_list, region_idx){
  total_batch = length(imp_list)
  num_region = length(region_idx)
  total_imp = rep(NA,total_batch)
  imp_list_p = vector("list", total_batch)
  imp_list_imp = vector("list", total_batch)
  imp_counter = 0
  for(b in 1:total_batch){
    print(paste("processing mask for data batch ",b))
    mask_b = extract_mask(imp_list[[b]])
    total_imp[b] = sum(mask_b==0)
    n_b = dim(mask_b)[2]
    imp_b = vector("list",num_region)
    imp_pidx_b = vector("list",num_region)
    # get imp idx for each i
    for(r in 1:num_region){
      imp_b_r = vector("list",n_b)
      imp_p_r = vector("list",n_b)
      p_idx = region_idx[[r]]
      for(i in 1:n_b){
        mask_b_i_r = mask_b[p_idx,i]
        imp_p_r[[i]] = which(mask_b_i_r==0)-1
        # imp_p_r[[i]] = which(mask_b[,i]==0)-1
        num_imp = sum(mask_b_i_r==0)
        if(num_imp == 0){
          imp_b_r[[i]] = imp_p_r[[i]]
        }else{
          imp_b_r[[i]] = (1:num_imp)+imp_counter-1
        }
        imp_counter = imp_counter+num_imp
      }
      imp_b[[r]] = imp_b_r
      imp_pidx_b[[r]] = imp_p_r
    }
    imp_counter = 0
    imp_list_imp[[b]] = imp_b
    imp_list_p[[b]] = imp_pidx_b
  }
  # # reformat to array
  # imp_array_imp = array(NA,dim=c(total_batch,max_n_b,max_r))
  # imp_array_len = array(NA,dim=c(total_batch,max_n_b,max_r))
  return(list(imp_list_vec = imp_list_imp,
              imp_list_p = imp_list_p,
              total_imp = total_imp))
}


generate_nifti_method2 = function(total_mean, oneBrain, region_idx, mask, obs_mask){
  total_nifti = rep(NA,prod(dim(mask)))
  region_vec = as.numeric(names(table(mask)))[-1]
  ct = 0
  for(i in region_vec){
    ct = ct+1
    # print(paste("i=",i))
    idx = which(c(mask)%in%i)
    obs_i = obs_mask[idx]
    idx_obs = idx[obs_i>0]
    if(length(region_idx[[ct]])>0){
      total_nifti[idx_obs] = total_mean[region_idx[[ct]]]
    }
    
  }
  total_nifti_output = oneBrain 
  oro.nifti::img_data(total_nifti_output ) = array(total_nifti, dim = dim(oneBrain))
  
  return(total_nifti_output)
}

Soft_threshold = function(x,lambda){
  return( (x-sign(x)*lambda)*(abs(x)>lambda))
}
plot_img = function(img, grids_df,title="img",col_bar = NULL){
  ggplot(grids_df, aes(x=x1,y=x2)) +
    geom_tile(aes(fill = img)) +
    scale_fill_viridis_c(limits = col_bar, oob = scales::squish)+
    
    ggtitle(title)+
    theme(plot.title = element_text(size=30),legend.text=element_text(size=20))
}

get_size = function(x){
  return(format(object.size(x), units = "auto"))
}
get_imp_list = function(imp_list, region_idx){
  total_batch = length(imp_list)
  num_region = length(region_idx)
  imp_list_p = vector("list", total_batch)
  imp_list_imp = vector("list", total_batch)
  imp_counter = 0
  for(b in 1:total_batch){
    mask_b = bigmemory::as.matrix(imp_list[[b]])
    n_b = dim(mask_b)[2]
    imp_b = vector("list",num_region)
    imp_pidx_b = vector("list",num_region)
    # get imp idx for each i
    for(r in 1:num_region){
      imp_b_r = vector("list",n_b)
      imp_p_r = vector("list",n_b)
      p_idx = region_idx[[r]]
      for(i in 1:n_b){
        mask_b_i_r = mask_b[p_idx,i]
        imp_p_r[[i]] = which(mask_b_i_r==0)-1
        # imp_p_r[[i]] = which(mask_b[,i]==0)-1
        num_imp = sum(mask_b_i_r==0)
        if(num_imp == 0){
          imp_b_r[[i]] = imp_p_r[[i]]
        }else{
          imp_b_r[[i]] = (1:num_imp)+imp_counter-1
        }
        imp_counter = imp_counter+num_imp
      }
      imp_b[[r]] = imp_b_r
      imp_pidx_b[[r]] = imp_p_r
    }
    imp_counter = 0
    imp_list_imp[[b]] = imp_b
    imp_list_p[[b]] = imp_pidx_b
  }
  # # reformat to array
  # imp_array_imp = array(NA,dim=c(total_batch,max_n_b,max_r))
  # imp_array_len = array(NA,dim=c(total_batch,max_n_b,max_r))
  return(list(imp_list_vec = imp_list_imp,
              imp_list_p = imp_list_p))
}

get_Y_imp_from_true = function(basis,L_idx_cpp,region_idx_cpp,imp_idx_list,Y_pred,b=1){
  Y_imp_p = NULL
  for(r in 1:length(basis$Phi_Q)){
    Q = basis$Phi_Q[[r]]
    L_range = L_idx_cpp[[r]]
    p_idx = region_idx_cpp[[r]]
    for( i in 1:100){
      imp_vec_i =  imp_idx_list$imp_list_vec[[b]][[r]][[i]]
      imp_p_i = imp_idx_list$imp_list_p[[b]][[r]][[i]]
      p_idx_i = p_idx[imp_p_i+1]
      # print(p_idx_i)
      # Y_imp_b0[imp_vec_i+1] = Q[p_idx_i+1,] %*% theta_eta_b[L_range+1,i]
      Y_imp_p = c(Y_imp_p,Y_pred[p_idx_i+1,i] )
    }
  }
  return(Y_imp_p)
}

get_random_mask = function(total_batch,batch_size_list, p,n,
                           common_percent = 0.9, missing_percent = 0.1,
                           contiguous_common = F, grids = NULL){
  if(contiguous_common){
    if(is.null(grids)){
      print("Error: grids need to be specified for contiguous_common=T.")
    }else{
      cluster_centers = apply(grids,2,mean)
      dist = apply(grids, 1, function(x){sum((x-cluster_centers)^2)})
      common_area = order(dist)[1:(common_percent*p)]
    }
    
  }else{
    common_area = sort(sample(1:p,common_percent*p))
  }
  
  a = rep(0,p); a[common_area]=1;
  missing_area = which(a==0);
  n_missing = length(missing_area);
  mask_list = vector("list",total_batch)
  mask_list_fbm = vector("list",total_batch)
  for(b in 1:total_batch){
    n_b = batch_size_list[b]
    mask = matrix(NA,p,n_b)
    mask[missing_area,] = 1*(matrix(runif(n_missing*n_b),n_missing,n_b)<missing_percent)
    mask[common_area,] = 1;
    mask_list[[b]] = as.big.matrix(mask)@address
    mask_list_fbm[[b]] = as.big.matrix(mask)
  }
  return(list(mask_list = mask_list, mask_list_fbm = mask_list_fbm))
}

GP.simulate.curve.fast.new = function(x,poly_degree,a,b,
                                      center=NULL,scale=NULL,max_range=6){
  
  x = cbind(x)
  d = ncol(x)
  
  if(is.null(center)){
    center = apply(x,2,mean)
  }
  c_grids = t(x) - center
  if(is.null(scale)){
    max_grids =pmax(apply(c_grids,1,max),-apply(c_grids,1,min))
    scale=as.numeric(max_grids/max_range)
  }
  
  work_x = GP.std.grids(x,center=center,scale=scale,max_range=max_range)
  Xmat = GP.eigen.funcs.fast(grids=work_x,
                             poly_degree =poly_degree,
                             a =a ,b=b)
  lambda = GP.eigen.value(poly_degree=poly_degree,a=a,b=b,d=d)
  return(list(eigen.func = Xmat, eigen.value = lambda))
}




generate_basis_sq_FBM = function(grids,a=0.01,b=10,poly_degree=10){
  GP = GP.simulate.curve.fast.new(grids,poly_degree=poly_degree,a=a,b=b)
  qr = qr(GP$eigen.func)
  Q_fbm = qr.Q(qr)
  D = GP.eigen.value(poly_degree=poly_degree,a=a,b=b)
  GP = NULL
  GP$Q = as_FBM(Q_fbm)
  GP$Q_mat = Q_fbm
  GP$Q_fbm = as.big.matrix(Q_fbm,type="double")
  GP$D = D
  return(GP)
}

generate_block_sq_basis = function(grids, region_idx_list,a = 0.01, b=10, poly_degree=20,
                                   show_progress=FALSE){
  num_block = length(region_idx_list)
  Phi_D = vector("list",num_block)
  Phi_Q = vector("list",num_block)
  Lt = NULL; pt = NULL
  for(i in 1:num_block){
    if(show_progress){
      print(paste("Computing basis for block ",i))
    }
    GP = GP.simulate.curve.fast.new(x=grids[region_idx_list[[i]],], a=a ,b=b,poly_degree=poly_degree) # try to tune b, increase for better FDR
    K_esq = GP$eigen.func
    K_QR = qr(K_esq)
    Phi_Q[[i]] = qr.Q(K_QR)
    Phi_D[[i]] = GP$eigen.value
    Lt = c(Lt, length(Phi_D[[i]]))
    pt = c(pt, dim(Phi_Q[[i]])[1])
  }
  return(list(Phi_D = Phi_D,
              region_idx_block = region_idx_list,
              Phi_Q = Phi_Q,L_all = Lt,p_length=pt))
}

# based on beta and delta
generate_data_FBM = function(n,true_beta,basis,sigma_alpha=1e-3){
  data_FBM = NULL
  N = n
  d = length(true_beta)
  GP = basis
  L = dim(GP$Q)[2]
  X_lim = 3
  data_FBM$sigma_Y = 0.5
  data_FBM$X = runif(n = N, min = -X_lim, max = X_lim)
  data_FBM$theta_eta.true = matrix(rnorm(N * L)*rep(sqrt(GP$D),N), ncol = N)
  data_FBM$eta = GP$Q_mat %*% data_FBM$theta_eta.true
  GP$Q_T = big_transpose(GP$Q)
  data_FBM$theta_beta.true = big_prodVec(GP$Q_T,beta)
  data_FBM$Q = GP$Q_fbm; data_FBM$Q_T = GP$Q_T
  data_FBM$eps_star = matrix(rnorm(N*L,sd=data_FBM$sigma_Y),nrow=L)
  data_FBM$beta = big_prodVec(GP$Q,data_FBM$theta_beta.true)
  data_FBM$delta = as.numeric(1*I(true_beta!=0))
  # temp_FBM = big_prodMat(GP$Q_T,diag(data_FBM$delta)) %*% data_FBM$beta
  temp_FBM = big_prodVec(GP$Q_T,data_FBM$beta*data_FBM$delta)
  data_FBM$sigma_eta = 1
  
  data_FBM$Y = as.matrix(data_FBM$beta*data_FBM$delta) %*% t(as.matrix(data_FBM$X)) + 
    data_FBM$eta + matrix(rnorm(N*d,sd=data_FBM$sigma_Y),nrow=d)
  
  
  data_FBM$Y_star = t(GP$Q_mat) %*% data_FBM$Y
  # data_FBM$Y_star = temp_FBM%*%t(as.matrix(data_FBM$X)) + data_FBM$theta_eta.true + data_FBM$eps_star
  
  # data_FBM$logLL = (-0.5/data_FBM$sigma_Y^2)*(norm(data_FBM$Y_star - temp_FBM%*%t(as.matrix(data_FBM$X)) - data_FBM$theta_eta.true,"f"))^2
  data_FBM$logLL_star = (-0.5/data_FBM$sigma_Y^2)*
    (norm(data_FBM$Y_star - temp_FBM%*%t(as.matrix(data_FBM$X))
          - data_FBM$theta_eta.true,"f"))^2 +
    (-n*L)/2*log(2*pi*data_FBM$sigma_Y^2)
  
  data_FBM$logLL = (-0.5/data_FBM$sigma_Y^2)*
    norm(data_FBM$Y- as.matrix(data_FBM$beta*data_FBM$delta) %*% t(as.matrix(data_FBM$X)) -
           data_FBM$eta ,"f")^2 +
    (-n*L)/2*log(2*pi*data_FBM$sigma_Y^2)
  
  data_FBM$snratio$beta = rep(NA, L)
  beta_term = temp_FBM%*%t(as.matrix(data_FBM$X))
  res_term = data_FBM$eps_star
  for(i in 1:L){
    data_FBM$snratio$beta[i] = sd(beta_term[i,]) / sd( res_term[i,] )
  }
  
  return(data_FBM)
}

generate_large_block_data_FBM = function(n,beta_true, basis, region_idx, 
                                         outpath,n_batch = 6){
  X = runif(n)
  L = length(unlist(basis$Phi_D))
  num_region = length(basis$Phi_D)
  p = length(beta_true)
  sigma_Y = 0.1
  sigma_eta = 0.1
  sigma_beta = 0.1
  theta_eta = matrix(rnorm(n*L)*sigma_eta,nrow=L)*sqrt(unlist(basis$Phi_D))
  eta = matrix(NA,nrow=p,ncol=n)
  L_idx = cbind(c(1,1+cumsum(basis$L_all)[-(num_region)]) , 
                c(cumsum(basis$L_all)))
  theta_beta = rep(0,L)
  beta_transformed = rep(0,p)
  for(r in 1:num_region){
    print(paste("generating eta for region ",r))
    L_idx_r = L_idx[r,1]:L_idx[r,2]
    p_idx = region_idx[[r]]
    eta[p_idx, ] = basis$Phi_Q[[r]] %*% theta_eta[L_idx_r,]
    theta_beta[L_idx_r] = t(basis$Phi_Q[[r]]) %*% beta_true[p_idx]
    beta_transformed[p_idx] = basis$Phi_Q[[r]] %*% theta_beta[L_idx_r]
  }
  delta = 1*(beta_true>0)
  eps =  matrix(rnorm(n*p),nrow=p)*sigma_Y
  print("Creating Y")
  Y = as.matrix(beta_transformed*delta)%*%t(as.matrix(X)) + eta + eps
  
  
  logLL = (-0.5/sigma_Y^2)*sum(eps^2) + (-n*L)/2*log(2*pi*sigma_Y^2)
  print("Spliting data into batches")
  b_all = ceiling(seq(1,n+1,length.out = n_batch+1))
  for(b in 2:(n_batch+1)){
    b_idx = b_all[b-1]:(b_all[b]-1)
    data_b = list(Y=Y[,b_idx],X=X[b_idx])
    saveRDS(data_b,file.path(outpath,paste("data_batch_n",n,"_p",p,"_L",L,"_b",b-1,".rds",sep="")))
  }
  
  snratio = apply(as.matrix(beta_transformed*delta)%*%t(as.matrix(X)),1,var)/apply(Y,1,var)
  
  data_params = list(beta = beta_transformed,delta=delta,
                     snratio =snratio ,
                     # Y=Y,Y_star=Y_star,X=X,
                     # eta=eta,
                     theta_eta = theta_eta, 
                     logLL = logLL,
                     sigma_Y = sigma_Y, sigma_beta = sigma_beta, sigma_eta = sigma_eta)
  saveRDS(data_params,file.path(outpath,paste("data_batch_n",n,"_p",p,"_L",L,"params.rds",sep="")))
  return(data_params)
}

generate_large_block_multiGP_data_FBM = function(n,beta_true, gamma_true, basis,
                                                 region_idx, outpath,sim_name=NULL,
                                                 sigma_Y = 1,
                                                 sigma_eta = 0.1,
                                                 X_scale = 1,
                                                 q=2, # number of GP
                                                 n_batch = 6){
  X = matrix(runif(n*(1+q)), ncol=n)
  X[1,] = X[1,]
  X[2:(q+1),] = X[2:(q+1),]
  
  X = X*X_scale
  
  L = length(unlist(basis$Phi_D))
  num_region = length(basis$Phi_D)
  p = length(beta_true)
  # sigma_Y = 1
  # sigma_eta = 0.1
  sigma_beta = 0.1
  theta_eta = matrix(rnorm(n*L)*sigma_eta,nrow=L)*sqrt(unlist(basis$Phi_D))
  eta = matrix(NA,nrow=p,ncol=n)
  L_idx = cbind(c(1,1+cumsum(basis$L_all)[-(num_region)]) , c(cumsum(basis$L_all)))
  theta_beta = rep(0,L)
  beta_transformed = rep(0,p)
  for(r in 1:num_region){
    print(paste("generating eta for region ",r))
    L_idx_r = L_idx[r,1]:L_idx[r,2]
    p_idx = region_idx[[r]]
    eta[p_idx, ] = basis$Phi_Q[[r]] %*% theta_eta[L_idx_r,]
    theta_beta[L_idx_r] = t(basis$Phi_Q[[r]]) %*% beta_true[p_idx]
    beta_transformed[p_idx] = basis$Phi_Q[[r]] %*% theta_beta[L_idx_r]
  }
  delta = 1*(beta_true>0)
  eps =  matrix(rnorm(n*p),nrow=p)*sigma_Y
  print("Creating Y")
  Y = as.matrix(beta_transformed*delta)%*%X[1,] + gamma_true %*% X[2:(q+1),] + eta + eps
  
  
  logLL = (-0.5/sigma_Y^2)*sum(eps^2) + (-n*L)/2*log(2*pi*sigma_Y^2)
  print("Spliting data into batches")
  b_all = ceiling(seq(1,n+1,length.out = n_batch+1))
  for(b in 2:(n_batch+1)){
    b_idx = b_all[b-1]:(b_all[b]-1)
    data_b = list(Y=Y[,b_idx],X=X[,b_idx])
    saveRDS(data_b,file.path(outpath,paste(sim_name,"data_multiGP_batch_n",n,"_p",p,"_L",L,"_b",b-1,".rds",sep="")))
  }
  
  snratio = apply(as.matrix(beta_transformed*delta)%*%X[1,],1,var)/apply(Y,1,var)
  
  data_params = list(beta = beta_transformed,
                     delta=delta,
                     gamma = gamma_true,
                     snratio =snratio ,
                     theta_eta = theta_eta, 
                     # eta = eta,
                     logLL = logLL,
                     sigma_Y = sigma_Y, sigma_beta = sigma_beta, sigma_eta = sigma_eta)
  saveRDS(data_params,file.path(outpath,paste(sim_name,"data_multiGP_batch_n",n,"_p",p,"_L",L,"params.rds",sep="")))
  return(data_params)
}

read_data_list_to_FBM = function(data_path_list,basis){
  n_batch = length(data_path_list)
  X = vector("list",n_batch)
  Y = X
  Y_star = Y
  for(b in 1:n_batch){
    print(paste("reading data batch ",b))
    data_b = readRDS(data_path_list[b])
    Y[[b]] = as.big.matrix(data_b$Y)@address
    Y_star[[b]] = as.big.matrix(High_to_low(data_b$Y,basis))@address
    # Y_fbm[[b]] = as_FBM(data_b$Y)
    X[[b]] = data_b$X
  }
  return(list(Y_list = Y, X_list = X, Y_star_list = Y_star))
}

read_data_list_to_FBM_impute = function(data_path_list,mask_list_fbm,basis){
  n_batch = length(data_path_list)
  X = vector("list",n_batch)
  Y = X
  Y_star = Y
  # Y_fbm = Y
  # Y_true_fbm = Y_fbm
  for(b in 1:n_batch){
    print(paste("reading data batch ",b))
    mask_b = bigmemory::as.matrix(mask_list_fbm[[b]])
    data_b = readRDS(data_path_list[b])
    # Y_true_fbm[[b]] = as_FBM(data_b$Y)
    Y_b = data_b$Y * mask_b
    Y_star[[b]] = as.big.matrix(High_to_low(Y_b,basis))@address
    Y[[b]] = as.big.matrix(Y_b)@address
    # Y_fbm[[b]] = as_FBM(Y_b)
    X[[b]] = data_b$X
  }
  return(list(Y_list = Y, X_list = X,
              Y_star_list = Y_star
              # Y_fbm_list = Y_fbm,
              # Y_true_fbm = Y_true_fbm
  ))
}

read_data_list_to_gs = function(data_path_list,mask_list_fbm){
  n_batch = length(data_path_list)
  Y_star = Y = X = NULL
  for(b in 1:n_batch){
    print(paste("to gs: reading data batch ",b))
    
    data_b = readRDS(data_path_list[b])
    mask_b = bigmemory::as.matrix(mask_list_fbm[[b]])
    Y_b = data_b$Y * mask_b
    Y = cbind(Y,data_b$Y)
    X = cbind(X, data_b$X)
  }
  Y_star=High_to_low(Y,basis)
  return(list(Y = Y, X = X,
              Y_star = Y_star))
}


High_to_low = function(X, basis, display = 0){
  n_region = length(basis$Phi_Q)
  n = dim(X)[2]
  L = sum(basis$L_all)
  X_star = matrix(NA, nrow=L,ncol=n)
  L_start = 1
  for(r in 1:n_region){
    if(display){
      print(paste("region = ",r))  
    }
    
    Q_t = big_transpose(as_FBM(basis$Phi_Q[[r]]))
    L_end = L_start + dim(Q_t)[1] - 1
    row_idx = as.integer(basis$region_idx_block[[r]])
    if(n==1){
      X_star[L_start:L_end,] = big_prodVec(Q_t, X[row_idx,] ) 
    }else{
      X_star[L_start:L_end,] = big_prodMat(Q_t, X[row_idx,] ) 
    }
    L_start = L_end + 1
  }
  return(X_star)
}

Low_to_high = function(X_star,basis,display = 0){
  n_region = length(basis$Phi_Q)
  n = dim(X_star)[2]
  L = sum(basis$L_all)
  p = sum(basis$p_length)
  X = matrix(NA, nrow=p,ncol=n)
  L_start = 1
  for(r in 1:n_region){
    if(display){
      
      print(paste("region = ",r))
    }
    Q = as_FBM(basis$Phi_Q[[r]])
    L_end = L_start + dim(Q)[2] - 1
    row_idx = as.integer(basis$region_idx_block[[r]])
    if(dim(X_star)[2]==1){
      X[row_idx,] = big_prodVec(Q, X_star[L_start:L_end,] ) 
    }else{
      X[row_idx,] = big_prodMat(Q, X_star[L_start:L_end,] )   
    }
    L_start = L_end + 1
  }
  return(X)
}

stats_by_region = function(Y,basis,func,display = 0){
  n_region = length(basis$Phi_Q)
  n = dim(Y)[2]
  L = sum(basis$L_all)
  p = sum(basis$p_length)
  region_stats = matrix(NA,nrow = n_region, ncol = n)
  for(r in 1:n_region){
    if(display){
      
      print(paste("region = ",r))
    }
    p_idx = as.integer(basis$region_idx_block[[r]])
    region_stats[r,] = apply(Y[p_idx,],2,func) 
  }
  return(region_stats)
}


get_Y_star = function(Y_list, basis){
  n_batch  = length(Y_list)
  Y_star_list = vector("list",n_batch)
  for(b in 1:n_batch){
    print(paste("batch = ",b))
    Y_star_b = Y_list[[b]]
    Y_star_list[[b]] = as.big.matrix(High_to_low(Y_star_b,basis))@address
  }
  return(Y_star_list)
}

# multiple X
generate_data_FBM_multiX = function(n,beta_list, basis){
  n_x = lenghth(beta_list)
  data_FBM = NULL
  N = n
  d = length(true_beta)
  GP = basis
  L = dim(GP$Q)[2]
  X_lim = 3
  data_FBM$sigma_Y = 0.5
  data_FBM$X = matrix(runif(n = N*n_x, min = -X_lim, max = X_lim),ncol=n,nrow=n_x) # n_x by n
  data_FBM$theta_eta.true = matrix(rnorm(N * L)*rep(sqrt(GP$D),N), ncol = N)
  data_FBM$eta = GP$Q_mat %*% data_FBM$theta_eta.true
  GP$Q_T = big_transpose(GP$Q)
  
  data_FBM$theta_beta.true = matrix(NA, nrow = L, ncol = n_x)
  for(x_i in 1:n_x){
    data_FBM$theta_beta.true[,x_i] = big_prodVec(GP$Q_T,beta_list[[x_i]])
  }
  
  data_FBM$Q = GP$Q_fbm; data_FBM$Q_T = GP$Q_T
  data_FBM$eps_star = matrix(rnorm(N*L,sd=data_FBM$sigma_Y),nrow=L)
  data_FBM$beta = matrix(NA, nrow = p, ncol = n_x) # p by n_x
  for(x_i in 1:n_x){
    data_FBM$beta[,x_i] = big_prodVec(GP$Q,data_FBM$theta_beta.true[,x_i])
  }
  
  
  data_FBM$delta = matrix(NA, nrow = p, ncol = n_x) # p by n_x
  for(x_i in 1:n_x){
    data_FBM$delta[,x_i] = as.numeric(1*I(beta_list[[x_i]]!=0))
  }
  # continue from here
  
  
  data_FBM$sigma_eta = 1
  
  data_FBM$Y = as.matrix(data_FBM$beta*data_FBM$delta) %*% data_FBM$X + 
    data_FBM$eta + matrix(rnorm(N*d,sd=data_FBM$sigma_Y),nrow=d) # p by n
  
  
  data_FBM$Y_star = t(GP$Q_mat) %*% data_FBM$Y # L by n
  
  temp_FBM = big_prodMat(GP$Q_T,data_FBM$beta*data_FBM$delta )
  data_FBM$logLL_star = (-0.5/data_FBM$sigma_Y^2)*
    (norm(data_FBM$Y_star - temp_FBM%*%data_FBM$X
          - data_FBM$theta_eta.true,"f"))^2 +
    (-n*L)/2*log(2*pi*data_FBM$sigma_Y^2)
  
  data_FBM$logLL = (-0.5/data_FBM$sigma_Y^2)*
    norm(data_FBM$Y- as.matrix(data_FBM$beta*data_FBM$delta) %*% data_FBM$X -
           data_FBM$eta ,"f")^2 +
    (-n*L)/2*log(2*pi*data_FBM$sigma_Y^2)
  
  data_FBM$snratio$beta = rep(NA, L)
  # data_FBM$theta_eta = theta_eta
  beta_term = temp_FBM %*% data_FBM$X
  res_term = data_FBM$eps_star
  for(i in 1:L){
    data_FBM$snratio$beta[i] = sd(beta_term[i,]) / sd( res_term[i,] )
  }
  
  return(data_FBM)
}

matern_kernel = function(x,y,nu,l=1){
  d = sqrt(sum((x-y)^2))/l
  y = 2^(1-nu)/gamma(nu)*(sqrt(2*nu)*d)^nu*besselK(sqrt(2*nu)*d,nu)
  return(y)
}
generate_matern_basis2 = function(grids, region_idx_list, L_vec,scale = 2,nu = 1/5,
                                  show_progress = FALSE){
  if(nu=="vec"){
    nu_vec = region_idx_list["nu_vec"]
  }
  num_block = length(region_idx_list)
  Phi_D = vector("list",num_block)
  Phi_Q = vector("list",num_block)
  Lt = NULL; pt = NULL
  for(i in 1:num_block){
    if(show_progress){
      print(paste("Computing basis for block ",i))
    }
    p_i = length(region_idx_list[[i]])
    kernel_mat = matrix(NA,nrow = p_i, ncol=p_i)
    for(l in 1:p_i){
      if(nu=="vec"){
        kernel_mat[l,] = apply(grids[region_idx_list[[i]],],1,matern_kernel,y=grids[region_idx_list[[i]],][l,],nu = nu_vec[i],l=scale)
      }else{
        kernel_mat[l,] = apply(grids[region_idx_list[[i]],],1,matern_kernel,y=grids[region_idx_list[[i]],][l,],nu = nu,l=scale)
      }
    }
    diag(kernel_mat) = 1
    K = eigs_sym(kernel_mat,L_vec[i])
    K_QR = qr(K$vectors)
    Phi_Q[[i]] = qr.Q(K_QR )
    Phi_D[[i]] = K$values
    Lt = c(Lt, length(Phi_D[[i]]))
    pt = c(pt, dim(Phi_Q[[i]])[1])
  }
  return(list(Phi_D = Phi_D,
              region_idx_block = region_idx_list,
              Phi_Q = Phi_Q,L_all = Lt,p_length=pt))
}


InclusionMap = function(mcmc_sample, true_beta, thresh = "auto", fdr_target = 0.15,max.iter = 100){
  InclusionProb = 1-apply(mcmc_sample, 1, function(x){mean(abs(x)==0)})
  true_beta = 1*(true_beta!=0)
  thresh_final = thresh
  if(thresh=="auto"){
    thresh = 0.5
    for(i in 1:max.iter){
      # InclusionProb_bin = 1*(abs(InclusionProb)>0)
      mapping = 1*(abs(InclusionProb)>thresh)
      fdr = FDR(mapping, true_beta)
      if(is.na(fdr)){
        print("Error: fdr=NA, target FDR is too small")
        return(list(tuning = F, 
                    InclusionProb=InclusionProb))
      }
      if(fdr<=fdr_target){
        thresh_final = thresh
        break
      }
      thresh = thresh*1.01
    }
  }else{
    mapping = 1*(InclusionProb>thresh)
  }
  return(list(mapping = mapping, thresh=thresh_final,
              InclusionProb=InclusionProb,tuning = T))
}

delta_idx_to_bin =  function(delta_sample){
  n = dim(delta_sample)[2]
  p = dim(delta_sample)[1]
  delta_bin = matrix(0,nrow = p, ncol=n)
  for(i in 1:n){
    delta_bin[delta_sample[,i]+1,i] = 1
  }
  return(delta_bin)
}

beta_mcmc = function(theta_beta_sample,delta_sample,basis,lambda){
  M = dim(theta_beta_sample)[2]
  S = dim(delta_sample)[1]
  
  beta_mcmc1 = basis$Q_mat %*% theta_beta_sample
  beta_mcmc1 = delta_sample*beta_mcmc1
  return(beta_mcmc1)
}

FDR = function(active_region, true_region){
  sum(active_region!=0 & true_region==0)/sum(active_region!=0)
}
Precision = function(active_region, true_region){
  mean(I(active_region!=0) == I(true_region!=0))
}
Power = function(active_region, true_region){
  sum(active_region !=0 & true_region!=0)/sum(true_region!=0)
}


