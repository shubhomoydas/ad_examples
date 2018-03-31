algo_plotdefs_norm_leaf = function(dataset, budget, basepath, n_win=30, sz_win=512) {
  
  # batch settings
  fsig_batch = list(file="%s-iforest_tau_instance-trees100_samples256_nscore4_leaf-top-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd300-tau0_030-topK0-pseudoanom_always_False-optim_scipy-norm-%s.csv",
                    dir="if_aad_trees100_samples256_i7_q1_bd300_nscore4_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma1000_mn1000_d100_norm")
  fsig_batch_adapt_prior = list(file=sprintf("%%s-iforest_tau_instance-trees100_samples256_nscore4_leaf-top-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd%d-tau0_030-topK0-pseudoanom_always_False-optim_scipy-norm-%%s.csv", budget),
                                dir=sprintf("if_aad_trees100_samples256_i7_q1_bd%d_nscore4_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma1000_mn1000_d100_norm", budget))
  fsig_batch_noprior_zero = list(file=sprintf("%%s-iforest_tau_instance-trees100_samples256_nscore4_leaf-top-noprior-init_zero-Ca1-1_1-fid1-runidx10-bd%d-tau0_030-topK0-pseudoanom_always_False-optim_scipy-norm-%%s.csv", budget),
                                 dir=sprintf("if_aad_trees100_samples256_i7_q1_bd%d_nscore4_leaf_tau0.03_xtau_noprior_init0_ca1_cx1_ma1000_mn1000_d100_norm", budget))
  fsig_batch_noprior_unif = list(file=sprintf("%%s-iforest_tau_instance-trees100_samples256_nscore4_leaf-top-noprior-init_uniform-Ca1-1_1-fid1-runidx10-bd%d-tau0_030-topK0-pseudoanom_always_False-optim_scipy-norm-%%s.csv", budget),
                                 dir=sprintf("if_aad_trees100_samples256_i7_q1_bd%d_nscore4_leaf_tau0.03_xtau_noprior_init1_ca1_cx1_ma1000_mn1000_d100_norm", budget))
  fsig_batch_noprior_rand = list(file=sprintf("%%s-iforest_tau_instance-trees100_samples256_nscore4_leaf-top-noprior-init_random-Ca1-1_1-fid1-runidx10-bd%d-tau0_030-topK0-pseudoanom_always_False-optim_scipy-norm-%%s.csv", budget),
                                 dir=sprintf("if_aad_trees100_samples256_i7_q1_bd%d_nscore4_leaf_tau0.03_xtau_noprior_init2_ca1_cx1_ma1000_mn1000_d100_norm", budget))
  fsig_batch_noprior_no_xtau_unif = list(file="%s-iforest_no_constraints-trees100_samples256_nscore4_leaf-top-noprior-init_uniform-Ca1-1_1-fid1-runidx10-bd300-tau0_030-topK0-pseudoanom_always_False-optim_scipy-norm-%s.csv",
                                         dir="if_aad_trees100_samples256_i7_q1_bd300_nscore4_leaf_tau0.03_noprior_init1_ca1_cx1_ma1000_mn1000_d100_norm")
  
  # streaming settings - fixed uniform prior
  fsig_stream_ovr = list(file=sprintf("%%s-iforest_tau_instance-trees100_samples256_nscore4_leaf-top-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd%d-tau0_030-topK0-pseudoanom_always_False-optim_scipy-norm-sw%d_asuTrue_mw%df2_20_overwrite_tillbudget-%%s.csv", budget, sz_win, n_win),
                         dir=sprintf("if_aad_trees100_samples256_i7_q1_bd%d_nscore4_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma1000_mn1000_d100_stream%dasu_mw%df2_20_ret0_tillbudget_norm", budget, sz_win, n_win))
  fsig_stream_top = list(file=sprintf("%%s-iforest_tau_instance-trees100_samples256_nscore4_leaf-top-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd%d-tau0_030-topK0-pseudoanom_always_False-optim_scipy-norm-sw%d_asuTrue_mw%df2_20_anomalous_tillbudget-%%s.csv", budget, sz_win, n_win),
                         dir=sprintf("if_aad_trees100_samples256_i7_q1_bd%d_nscore4_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma1000_mn1000_d100_stream%dasu_mw%df2_20_ret1_tillbudget_norm", budget, sz_win, n_win))
  
  # streaming settings - adaptive uniform prior
  fsig_stream_adapt_ovr = list(file=sprintf("%%s-iforest_tau_instance-trees100_samples256_nscore4_leaf-top-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd300-tau0_030-topK0-pseudoanom_always_False-optim_scipy-norm-sw%d_asuTrue_mw%df2_20_overwrite-%%s.csv", sz_win, n_win),
                               dir=sprintf("if_aad_trees100_samples256_i7_q1_bd300_nscore4_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma1000_mn1000_d100_stream%dasu_mw%df2_20_ret0_norm", sz_win, n_win))
  fsig_stream_adapt_top = list(file=sprintf("%%s-iforest_tau_instance-trees100_samples256_nscore4_leaf-top-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd300-tau0_030-topK0-pseudoanom_always_False-optim_scipy-norm-sw%d_asuTrue_mw%df2_20_anomalous-%%s.csv", sz_win, n_win),
                               dir=sprintf("if_aad_trees100_samples256_i7_q1_bd300_nscore4_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma1000_mn1000_d100_stream%dasu_mw%df2_20_ret1_norm", sz_win, n_win))
  
  # QueryTopRandom
  fsig_batch_adapt_prior_qt = list(file="%s-iforest_tau_instance-trees100_samples256_nscore4_leaf-toprandom-unifprior_adapt-init_uniform-Ca1-1_1-fid1-runidx10-bd300-tau0_030-topK0-pseudoanom_always_False-optim_scipy-norm-%s.csv",
                                   dir="if_aad_trees100_samples256_i7_q2n2_bd300_nscore4_leaf_tau0.03_xtau_s0.5_adapt_init1_ca1_cx1_ma1000_mn1000_d100_norm")
  fsig_stream_adapt_top_qt = list(file=sprintf("%%s-iforest_tau_instance-trees100_samples256_nscore4_leaf-toprandom-unifprior-init_uniform-Ca1-1_1-fid1-runidx10-bd300-tau0_030-topK0-pseudoanom_always_False-optim_scipy-norm-sw%d_asuTrue_mw%df2_20_anomalous-%%s.csv", sz_win, n_win),
                                  dir=sprintf("if_aad_trees100_samples256_i7_q2n2_bd300_nscore4_leaf_tau0.03_xtau_s0.5_init1_ca1_cx1_ma1000_mn1000_d100_stream%dasu_mw%df2_20_ret1_norm", sz_win, n_win))
  
  plotdefs = list()
  filename = sprintf(fsig_batch_adapt_prior$file, dataset, "baseline")
  plotname = "baseline"
  plotdefs[[plotname]] = list(
    name=plotname, def="Unsupervised Baseline", 
    path=file.path(basepath, dataset, fsig_batch_adapt_prior$dir),
    fname=filename)
  
  filename = sprintf(fsig_batch$file, dataset, "num_seen")
  plotname = "aad_batch"
  plotdefs[[plotname]] = list(
    name=plotname, def="BAL", 
    path=file.path(basepath, dataset, fsig_batch$dir),
    fname=filename)
  
  filename = sprintf(fsig_batch_adapt_prior$file, dataset, "num_seen")
  plotname = "aad_batch_adapt_prior"
  plotdefs[[plotname]] = list(
    name=plotname, def="BAL (Adaptive Prior)", 
    path=file.path(basepath, dataset, fsig_batch_adapt_prior$dir),
    fname=filename)
  
  filename = sprintf(fsig_batch_noprior_zero$file, dataset, "num_seen")
  plotname = "aad_batch_noprior_zero"
  plotdefs[[plotname]] = list(
    name=plotname, def="BAL (No Prior - Zero)", 
    path=file.path(basepath, dataset, fsig_batch_noprior_zero$dir),
    fname=filename)
  
  filename = sprintf(fsig_batch_noprior_unif$file, dataset, "num_seen")
  plotname = "aad_batch_noprior_unif"
  plotdefs[[plotname]] = list(
    name=plotname, def="BAL (No Prior - Unif)", 
    path=file.path(basepath, dataset, fsig_batch_noprior_unif$dir),
    fname=filename)
  
  filename = sprintf(fsig_batch_noprior_rand$file, dataset, "num_seen")
  plotname = "aad_batch_noprior_rand"
  plotdefs[[plotname]] = list(
    name=plotname, def="BAL (No Prior - Rand)", 
    path=file.path(basepath, dataset, fsig_batch_noprior_rand$dir),
    fname=filename)
  
  filename = sprintf(fsig_batch_noprior_no_xtau_unif$file, dataset, "num_seen")
  plotname = "aad_batch_noprior_no_xtau_unif"
  plotdefs[[plotname]] = list(
    name=plotname, def="BAL No Prior - No xtau Unif", 
    path=file.path(basepath, dataset, fsig_batch_noprior_no_xtau_unif$dir),
    fname=filename)
  
  filename = sprintf(fsig_batch_adapt_prior_qt$file, dataset, "num_seen")
  plotname = "aad_batch_adapt_prior_qt"
  plotdefs[[plotname]] = list(
    name=plotname, def="BAL QueryTopRandom", 
    path=file.path(basepath, dataset, fsig_batch_adapt_prior_qt$dir),
    fname=filename)
  
  if (T) {
    filename = sprintf(fsig_stream_ovr$file, dataset, "baseline")
    plotname = "baseline_stream"
    plotdefs[[plotname]] = list(
      name=plotname, def="SAL Baseline", 
      path=file.path(basepath, dataset, fsig_stream_ovr$dir),
      fname=filename)
    
    filename = sprintf(fsig_stream_ovr$file, dataset, "num_seen")
    plotname = "aad_stream_ovr"
    plotdefs[[plotname]] = list(
      name=plotname, def="SAL (Naive)", 
      path=file.path(basepath, dataset, fsig_stream_ovr$dir),
      fname=filename)
    
    filename = sprintf(fsig_stream_ovr$file, dataset, "num_total_anoms")
    plotname = "aad_stream_ovr_total"
    plotdefs[[plotname]] = list(
      name=plotname, def="SAL Total Anoms (Naive)", 
      path=file.path(basepath, dataset, fsig_stream_ovr$dir),
      fname=filename)
    
    filename = sprintf(fsig_stream_top$file, dataset, "num_seen")
    plotname = "aad_stream_top"
    plotdefs[[plotname]] = list(
      name=plotname, def="SAL (Retain Top)", 
      path=file.path(basepath, dataset, fsig_stream_top$dir),
      fname=filename)
    
    filename = sprintf(fsig_stream_top$file, dataset, "num_total_anoms")
    plotname = "aad_stream_top_total"
    plotdefs[[plotname]] = list(
      name=plotname, def="SAL Total Anoms (Retain Top)", 
      path=file.path(basepath, dataset, fsig_stream_top$dir),
      fname=filename)
    
    # adaptive uniform priors
    filename = sprintf(fsig_stream_ovr$file, dataset, "baseline")
    plotname = "baseline_stream_adapt"
    plotdefs[[plotname]] = list(
      name=plotname, def="SAL Baseline Adapt (Naive)", 
      path=file.path(basepath, dataset, fsig_stream_ovr$dir),
      fname=filename)
    
    filename = sprintf(fsig_stream_adapt_ovr$file, dataset, "num_seen")
    plotname = "aad_stream_adapt_ovr"
    plotdefs[[plotname]] = list(
      name=plotname, def="SAL Adapt (Naive)", 
      path=file.path(basepath, dataset, fsig_stream_adapt_ovr$dir),
      fname=filename)

    filename = sprintf(fsig_stream_adapt_ovr$file, dataset, "num_total_anoms")
    plotname = "aad_stream_adapt_ovr_total"
    plotdefs[[plotname]] = list(
      name=plotname, def="SAL Total Anoms Adapt (Naive)", 
      path=file.path(basepath, dataset, fsig_stream_adapt_ovr$dir),
      fname=filename)
    
    filename = sprintf(fsig_stream_adapt_top$file, dataset, "num_seen")
    plotname = "aad_stream_adapt_top"
    plotdefs[[plotname]] = list(
      name=plotname, def="SAL Adapt (Retain Top)", 
      path=file.path(basepath, dataset, fsig_stream_adapt_top$dir),
      fname=filename)
    
    filename = sprintf(fsig_stream_adapt_top$file, dataset, "num_total_anoms")
    plotname = "aad_stream_adapt_top_total"
    plotdefs[[plotname]] = list(
      name=plotname, def="SAL Total Anoms Adapt (Retain Top)", 
      path=file.path(basepath, dataset, fsig_stream_adapt_top$dir),
      fname=filename)
  }
  return (plotdefs)
}

get_defs = function(dataset, budget, plotnames, basepath, n_win=30, sz_win=512) {
  fpaths = c()
  dispnames = c()
  defs = algo_plotdefs_norm_leaf(dataset, budget, basepath, n_win=n_win, sz_win=sz_win)
  for (plotname in plotnames) {
    def = defs[[plotname]]
    # print (def$def)
    dispnames = c(dispnames, def$def)
    fpaths = c(fpaths, file.path(def$path, def$fname))
    fpaths_bw = c(fpaths, file.path(def$path, def$fname_bw))
    fpaths_nw = c(fpaths, file.path(def$path, def$fname_nw))
  }
  return (list(plotnames=plotnames, dispnames=dispnames, fpaths=fpaths, 
               fpaths_bw=fpaths_bw, fpaths_nw=fpaths_nw))
}

load_result_summary <- function(datasetinfo, flist, basepath) {
  dataset = datasetinfo$dataset
  budget = datasetinfo$budget
  defs = get_defs(dataset, budget, flist, basepath, 
                  n_win=datasetinfo$n_win, sz_win=datasetinfo$sz_win)
  budgets = c()
  numseen = matrix(0, nrow=length(flist), ncol=max(budget*2, 1000))
  numseen_sd = matrix(0, nrow=length(flist), ncol=ncol(numseen))
  max_nc = 0
  nruns = c()
  for (i in 1:length(flist)) {
    # print(defs$fpaths[i])
    if (!file.exists(defs$fpaths[i])) {
      print(sprintf("not found: %s", defs$fpaths[i]))
    }
    fdata = read.csv(defs$fpaths[i], header=F)
    tmp = as.matrix(fdata[, 3:ncol(fdata)], nrow=nrow(fdata), ncol=ncol(fdata)-2)
    tmp = tmp * 100 / datasetinfo$anoms  # fraction of total anomalies
    nc = ncol(tmp)
    max_nc = max(nc, max_nc)
    if (nrow(tmp) == 1) {
      numseen[i,1:nc] = tmp[1, 1:nc]
      numseen_sd[i,1:nc] = 0
    } else {
      # print(c(nc, nrow(tmp)))
      numseen[i,1:nc] = apply(tmp, MAR=2, FUN=mean)[1:nc]
      numseen_sd[i,1:nc] = apply(tmp, MAR=2, FUN=sd)[1:nc]
    }
    # print(numseen[1:1, 1:8])
    budgets = c(budgets, nc)
    nruns = c(nruns, nrow(tmp))
  }
  return (list(dataset=dataset, max_budget=budget, budgets=budgets, anoms=min(budgets, datasetinfo$anoms),
               plotnames=defs$plotnames, dispnames=defs$dispnames,
               numseen=numseen[, 1:max_nc], numseen_sd=numseen_sd[, 1:max_nc], 
               nruns=nruns))
}

load_conf_result_summary <- function(datasetinfo, flist, basepath) {
  dataset = datasetinfo$dataset
  budget = datasetinfo$budget
  defs = get_defs(dataset, budget, flist, basepath)
  # print(defs)
  budgets = c()
  numseen = matrix(0, nrow=length(flist), ncol=1000)
  numseen_sd = matrix(0, nrow=length(flist), ncol=1000)
  ntimes = matrix(0, nrow=length(flist), ncol=1000)
  max_nc = 0
  nruns = c()
  for (i in 1:length(flist)) {
    # print(defs$fpaths[i])
    if (defs$plotnames[i] %in% c("window-baseline", "window")) {
      fdata = read.csv(defs$fpaths[i], header=F)
      nc = max(fdata)
      max_nc = max(max_nc, nc)
      tmp = matrix(0, nrow=nrow(fdata), ncol=nc)
      for (k in 1:nrow(fdata)) {
        n_w = fdata[k,]
        n_w = n_w[n_w >= 0]
        # print(table(n_w))
        tmp[k, 1:nc] = table(n_w)
      }
      
      tmp_mean = apply(tmp, MAR=2, FUN=mean)
      tmp_sd = apply(tmp, MAR=2, FUN=sd)
      numseen[i, 1:nc] = tmp_mean
      numseen_sd[i, 1:nc] = tmp_sd
      ntimes[i, 1:nc] = nrow(tmp)
      budgets = c(budgets, max_nc)
      nruns = c(nruns, nrow(tmp))
    } else {
      fdata = read.csv(defs$fpaths[i], header=F)
      tmp = as.matrix(fdata[, 3:ncol(fdata)], nrow=nrow(fdata), ncol=ncol(fdata)-2)
      mx_anoms = max(tmp)  # max number of anomalies
      n_qs = rep(0, mx_anoms)  # stores number of queries to get to k true anomalies
      n_nn = rep(0, mx_anoms)  # stores number of queries to get to k true anomalies
      for (k in 1:mx_anoms) {
        l = c()
        for (j in 1:nrow(tmp)) {
          p = which(tmp[j,] == k)
          if (length(p) > 0) {
            l = c(l, p[1])
          }
        }
        numseen[i, k] = mean(l)
        numseen_sd[i, k] = sd(l)
        ntimes[i, k] = length(l)
      }
      nc = mx_anoms
      max_nc = max(nc, max_nc)
      # print(numseen[1:1, 1:8])
      budgets = c(budgets, nc)
      nruns = c(nruns, nrow(tmp))
    }
  }
  return (list(dataset=dataset, max_budget=budget, budgets=budgets, 
               plotnames=defs$plotnames, dispnames=defs$dispnames,
               numseen=numseen[, 1:max_nc], numseen_sd=numseen_sd[, 1:max_nc], 
               ntimes=ntimes[, 1:max_nc], nruns=nruns))
}

