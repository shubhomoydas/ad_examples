
rm(list=ls())

library(extrafont)
# Required first time
# font_import()
# loadfonts()

if (F) {
  source("/Users/moy/work/git/bb_python/ad_examples/R/aad_comparisons.R")
}

script_base = "/Users/moy/work/git/bb_python/ad_examples/R"
if (F) {
  input_base  = "/Users/moy/work/WSU/Aeolus/server/results-aad_stream"
  output_base = "/Users/moy/work/WSU/Aeolus/server"
} else {
  input_base  = "/Users/moy/work/git/bb_python/ad_examples/python/temp/aad"
  output_base = "/Users/moy/work/git/bb_python/ad_examples/python/temp/aad"
}
source(file.path(script_base, "aad_plot_defs.R"))

error.bar <- function(x, y, upper, lower=upper, len=0.1,...) {
  if(length(x) != length(y) | length(y) !=length(lower) | length(lower) != length(upper))
    stop("vectors must be same length")
  arrows(x,y+upper, x, y-lower, angle=90, code=3, length=len, ...)
}

get_n_intermediate <- function(x, n) {
  m <- length(x)
  p <- round((1:n) * (m / n))
  return(x[p])
}

all_datasets = list(
  "abalone"=list(dataset="abalone", budget=300, anoms=29, n_win=30, sz_win=512), 
  "ann_thyroid_1v3"=list(dataset="ann_thyroid_1v3", budget=300, anoms=73, n_win=30, sz_win=512), 
  "cardiotocography_1"=list(dataset="cardiotocography_1", budget=300, anoms=45, n_win=30, sz_win=512), 
  "yeast"=list(dataset="yeast", budget=300, anoms=55, n_win=30, sz_win=512), 
  "covtype"=list(dataset="covtype", budget=3000, anoms=2747, n_win=1000, sz_win=4096), 
  "kddcup"=list(dataset="kddcup", budget=3000, anoms=2416, n_win=30, sz_win=4096), 
  "mammography"=list(dataset="mammography", budget=1500, anoms=260, n_win=30, sz_win=4096), 
  "shuttle_1v23567"=list(dataset="shuttle_1v23567", budget=1500, anoms=867, n_win=30, sz_win=4096),
  "toy2"=list(dataset="toy2", budget=300, anoms=35, n_win=30, sz_win=512)
)

compare_types = c("batch", "to_reach", "stream", "stream_total", 
                  "stream_adapt_vs_not", "query_top_random", "batch_vs_stream",
                  "stream_adapt_vs_fixed_prior", "stream_fixed_vs_adapt_total")
compare_type = compare_types[1]  # 1|4
if (compare_type == compare_types[1]) {
  flist = c("baseline", 
            "aad_batch_adapt_prior"
            , "aad_batch_noprior_unif"
            , "aad_batch_noprior_zero", "aad_batch_noprior_rand"
            # , "aad_batch", "aad_batch_noprior_no_xtau_unif"
            )
  colrs = c("blue", "red", "green", "orange", "black", "magenta", "cyan", "brown", "grey80")
  lty = 1
  outpath = file.path(output_base, "plots-batch")
} else if (compare_type == compare_types[3]) {
  flist = c("baseline_stream", "aad_stream_top", "aad_stream_ovr")
  colrs = c("blue", "red", "green", "orange", "black", "magenta", "cyan", "brown", "grey80")
  lty = 1
  outpath = file.path(output_base, "plots-stream")
} else if (compare_type == compare_types[4]) {
  # flist = c("aad_stream_ovr", "aad_stream_top", "aad_stream_ovr_total", "aad_stream_top_total")
  flist = c("baseline", "aad_batch_adapt_prior", "aad_stream_top", "aad_stream_ovr") # , "aad_stream_top_total", "aad_stream_ovr_total")
  colrs = c("blue", "red", "darkgreen", "black", "darkgreen", "black")
  lty = c(1, 1, 1, 1, 2, 2)
  outpath = file.path(output_base, "plots-stream-total")
} else if (compare_type == compare_types[5]) {
  flist = c("aad_stream_adapt_ovr", "aad_stream_adapt_top", "aad_stream_adapt_ovr_total","aad_stream_adapt_top_total")
  colrs = c("blue", "red", "blue", "red")
  lty = c(1, 1, 2, 2)
  outpath = file.path(output_base, "plots-stream-total-adapt_v_not")
} else if (compare_type == compare_types[6]) {
  flist = c("aad_batch_adapt_prior", "aad_batch_adapt_prior_qt")
  colrs = c("blue", "red")
  lty = c(1, 1)
  outpath = file.path(output_base, "plots-batch-toprandom")
} else if (compare_type == compare_types[7]) {
  flist = c("aad_batch_adapt_prior", "aad_stream_adapt_top")
  colrs = c("blue", "red")
  lty = c(1, 1)
  outpath = file.path(output_base, "plots-batch_vs_stream")
} else if (compare_type == compare_types[8]) {
  flist = c("aad_stream_top", "aad_stream_adapt_top")
  colrs = c("blue", "red")
  lty = c(1, 1)
  outpath = file.path(output_base, "plots-stream_adapt_vs_fixed_prior")
} else if (compare_type == compare_types[9]) {
  flist = c("aad_stream_top_total", "aad_stream_adapt_top_total")
  colrs = c("blue", "red")
  lty = c(1, 1)
  outpath = file.path(output_base, "plots-stream_fixed_vs_adapt_total")
}

dir.create(outpath, recursive=T, showWarnings=F)

datasets = c("abalone", 
             "ann_thyroid_1v3", 
             "cardiotocography_1",
             "yeast"
             , "mammography"
             , "shuttle_1v23567"
             , "covtype"
             , "kddcup"
             )
# datasets = c("abalone", "ann_thyroid_1v3", "cardiotocography_1", "yeast")
# datasets = c("abalone")
# datasets = c("kddcup")

pchs = 1:7

diffs = data.frame(dataset=character(), algo=character(), diff=numeric(), sd=numeric())
for (datasetinfo in all_datasets[datasets]) {
  res = load_result_summary(datasetinfo, flist, input_base)
  # res = load_conf_result_summary(datasetinfo, flist, input_base)
  # stop("debug")
  if (compare_type == compare_types[2]) {
    fout_tmp = file.path(outpath, sprintf("tmp_num_seen-%s.pdf", datasetinfo$dataset))
    fout = file.path(outpath, sprintf("num_seen-%s.pdf", datasetinfo$dataset))
    pdf(fout_tmp, family="Arial")
    plot(0, type="n", 
         xlim=c(1, max(res$budgets)), ylim=c(0, max(res$numseen)), 
         xlab="k-th anomaly", ylab="#queries to reach k-th anomaly",
         cex.lab=1.8, cex.axis=1.4)
    lwds = c()
    for (i in 1:nrow(res$numseen)) {
      lwd = 2
      lwds = c(lwds, lwd)
      lines(1:res$budgets[i], res$numseen[i, 1:res$budgets[i]], col=colrs[i], lwd=lwd)
      pts = get_n_intermediate((1:(res$budgets[i]-nrow(res$numseen)))+i, 10)
      std_errs = 1.96*res$numseen_sd[i, pts]/sqrt(res$ntimes[i,pts])
      suppressWarnings(error.bar(pts, res$numseen[i, pts], upper=std_errs, 
                                 len=0.05, col=colrs[i]))
    } 
  } else {
      fout_tmp = file.path(outpath, sprintf("tmp_num_seen-%s.pdf", datasetinfo$dataset))
      fout = file.path(outpath, sprintf("num_seen-%s.pdf", datasetinfo$dataset))
      pdf(fout_tmp, family="Arial")
      if (compare_type == compare_types[4]) {
        # budget = ncol(res$numseen)
        budget = res$budgets[1]
      } else {
        budget = min(datasetinfo$budget, ncol(res$numseen))
      }
      # ymax = min(budget, datasetinfo$anoms)
      ymax = 100.0
      plot(0, type="n", xlim=c(1, budget), ylim=c(0, ymax), xlab="# queries", ylab="% of total anomalies seen",
           cex.lab=1.8, cex.axis=1.6)
      lwds = c()
      ltys = c()
      for (i in 1:nrow(res$numseen)) {
        lwd = 2
        lwds = c(lwds, lwd)
        lt = lty
        if (length(lty) > 1) {
          lt = lty[i]
        }
        lines(1:res$budgets[i], res$numseen[i, 1:res$budgets[i]], col=colrs[i], lwd=lwd, lty=lt)
        pts = get_n_intermediate((1:(res$budgets[i]-nrow(res$numseen)))+i, 10)
        std_errs = 1.96*res$numseen_sd[i, pts]/sqrt(res$nruns[i])
        suppressWarnings(error.bar(pts, res$numseen[i, pts], upper=std_errs, 
                                   len=0.05, col=colrs[i]))
      }
      if (F && datasetinfo$anoms <= 300) {
        lines(1:res$budgets[i], rep(datasetinfo$anoms, res$budgets[i]), col="grey", lwd=1, lty=1)
      }
  }
  legend("bottomright", legend=res$dispnames, 
         col=colrs[1:nrow(res$numseen)], cex=1.8, lwd=lwds, lty=lty 
         )
  dev.off()
  
  embed_fonts(fout_tmp, outfile=fout)
  print(datasetinfo$dataset)
}

