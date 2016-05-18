# Elio Amicarelli
# "From Fog Nets to Neural Nets" functions set 1 of 2

# FUNCTION 1 - fog.seqmod
# Description: f() trains, optimizes and produces predictions by using the fog data sequentially. 
# Arguments: target.path (path for the entire target data file), predictors.path (path for the entire IV data file), 
 # chunks.begin(num for starting chunk), chunk.end(num for ending chunk), chunks.threshold.path (path to file with splitting points for the entire predictors dataset),
 # model (caret model to be trained "rf" or "xgbTree" ), downsampling (TRUE or FALSE), downsampling.start (chunk to start downsampling)
 # downsampling.end (chunk to end downsampling), upsampling.start (as downsampling.start but upsampling), upsampling.end (as downsampling.end but upsampling)
 # multicore (TRUE/FALSE is good for rf)
# Note: This function requires to prespecify the trainControl and parameters' grid (caret)

fog.seqmod<-function(target.path,predictors.path, round.n, chunks.begin, chunks.end, chunks.threshold.path, model, downsampling, downsampling.start,downsampling.end, upsampling, upsampling.start, upsampling.end, multicore = TRUE){
  
  print("Important note: this function requires to prespecify the trainControl and parameters' grid (see caret documentation).")
  
  # import libraries
  library(caret)
  library(randomForest)
  library(xgboost)
  
  # keep a log of the fitting operations
  file.path<-paste("/FogNets/",round.n,"/log.txt", sep="")
  cat(round.n,"\n",file=file.path)
  
  # import raw data for predictors and format date 
  #[date var required to be named "X" with format Y-m-d H:M:S]
  data<-read.csv(predictors.path)
  data$X<-as.POSIXct(data$X, format="%Y-%m-%d %H:%M:%S") 
  
  # import target and format date 
  #[ditto date var]
  target<-read.csv(target.path)
  target$X<-as.POSIXct(target$X,format="%Y-%m-%d %H:%M:%S")
  
  # import traint/test split dates and format them 
  # [begin and end dates require to be named "threshold" and "limit" respectively with format Y-m-d H:M:S]
  thresholds<-read.csv(chunks.threshold.path)
  thresholds$threshold<-as.POSIXct(thresholds$threshold,format="%Y-%m-%d %H:%M:%S")
  thresholds$limit<-as.POSIXct(thresholds$limit,format="%Y-%m-%d %H:%M:%S")

  i=0 # the nostalgic note
  # Main loop
  for(i in chunks.begin:chunks.end){
    # create directory for chunk i
    dir.create(paste("/FogNets/",round.n,"/",as.character(i),sep=""))
    # flag
    #print (paste("i: ",i," split date: ", as.character(thresholds[i,1]), sep=""))
    cat("** i:",i,"split date: ", as.character(thresholds[i,1]),"\n",file=file.path,append=TRUE)
    cat("** i:",i,"split date: ", as.character(thresholds[i,1]),"\n")
    # flag
    cat("reading data...\n")
    # take predictors before "threshold" i 
    data.train<-data[data$X<thresholds[i,1], ]
    # take predictors only for training observations, needed when i > 1 in order to filter out test observations
    data.train<-data.train[data.train$Set==0,]
    # match predictors with targets
    data.train<-merge(data.train, target, by="X", all.x=TRUE)
    if(sum(is.na(data.train$yield)) >= 1){
      data.train<-data.train[-which(is.na(data.train$yield)), ]
    } 
    
    # remove dates and train/test index; do not want "X" during train but stored it in tmpX anyway
    data.train<-data.train[,-which(names(data.train)%in%c("X","X.1","Set"))]
    # flag
    #print(paste("train dim: ",as.character(nrow(data.train)),"*",as.character(ncol(data.train)), sep=""))
    cat("train dim: ",as.character(nrow(data.train)),"*",as.character(ncol(data.train)),"\n",file=file.path,append=TRUE)
    cat("train dim: ",as.character(nrow(data.train)),"*",as.character(ncol(data.train)),"\n")
    # save the i raw train data
    train.path<-paste("/FogNets/",round.n,"/", as.character(i), "/","train",as.character(i),".csv", sep="")
    write.csv(data.train,train.path)
    
    # downsample train data i
    if(downsampling == TRUE & i >= downsampling.start & i <= downsampling.end ){
      #data.train.complete<-data.train # $1 data.train.test<-data.train.complete 
      seed<-sample(100:15000,1)
      cat("Downsampling, selected seed:",seed,"\n",file=file.path,append=TRUE)
      cat("Downsampling, selected seed:",seed,"\n")
      set.seed(seed)
      data.train$Class<-as.factor(ifelse(data.train$yield>0,1,0))
      data.train<-downSample(data.train, data.train$Class)
      # print(table(data.train$Class))
      cat("balance 0 1 classes",table(data.train$Class),"\n",file=file.path,append=TRUE)
      cat("balance 0 1 classes",table(data.train$Class),"\n")
      data.train<-data.train[,names(data.train)!="Class"]
      train.path<-paste("/FogNets/",round.n,"/", as.character(i), "/","train_dwn",as.character(i),".csv", sep="")
      write.csv(data.train,train.path)
    }else{
      cat("No downsampling...","\n",file=file.path,append=TRUE)
      cat("No downsampling...","\n")
    }
    
    # upsample train data i
    if(upsampling == TRUE & i >= upsampling.start & i <= upsampling.end){
      #data.train.complete<-data.train # $1 data.train.test<-data.train.complete 
      seed<-sample(100:15000,1)
      cat("Upsampling, selected seed:",seed,"\n",file=file.path,append=TRUE)
      cat("Upsampling, selected seed:",seed,"\n")
      set.seed(seed)
      data.train$Class<-as.factor(ifelse(data.train$yield>0,1,0))
      data.train<-upSample(data.train, data.train$Class)
      # print(table(data.train$Class))
      cat("balance 0 1 classes",table(data.train$Class),"\n",file=file.path,append=TRUE)
      cat("balance 0 1 classes",table(data.train$Class),"\n")
      data.train<-data.train[,names(data.train)!="Class"]
      train.path<-paste("/FogNets/",round.n,"/", as.character(i), "/","train_up",as.character(i),".csv", sep="")
      write.csv(data.train,train.path)
    }else{
      cat("No upsampling...","\n",file=file.path,append=TRUE)
      cat("No upsampling...","\n")
    }
    
    # take predictors between "threshold" i and "limit" i that is predictors for test i data
    data.test<-data[data$X>=thresholds[i,1] & data$X<=thresholds[i,2] , ]
    # remove useless predictors but mantain dates "X"
    data.test<-data.test[,-which(names(data.test)%in%c("X.1","Set"))]
    # flag
    #print(paste("test dim: ",as.character(nrow(data.test)),"*",as.character(ncol(data.test)), sep=""))
    cat("test dim:",as.character(nrow(data.test)),"*",as.character(ncol(data.test)),"\n",file=file.path,append=TRUE)
    cat("test dim:",as.character(nrow(data.test)),"*",as.character(ncol(data.test)),"\n")
    
    if(any(is.na(data.test))){
      print("OOps! You have NAs in your test set...")
    }
    
    cat("test dim:",as.character(nrow(data.test)),"*",as.character(ncol(data.test)),"\n",file=file.path,append=TRUE)
    cat("test dim:",as.character(nrow(data.test)),"*",as.character(ncol(data.test)),"\n")
    test.path<-paste("/FogNets/",round.n,"/", as.character(i), "/","test",as.character(i),".csv", sep="")
    write.csv(data.test,test.path)
    
    cat("fittig data on train dim:",as.character(nrow(data.train)),"*",as.character(ncol(data.train)),"\n",file=file.path,append=TRUE)
    cat("fittig data on train dim:",as.character(nrow(data.train)),"*",as.character(ncol(data.train)),"\n")
    
    if (multicore == TRUE){
      set.seed(1243)
      library(doMC)
      registerDoMC(cores = 5)
    }
    
    modelspace = train(yield ~ ., data = data.train, 
                       method = model,        
                       trControl = ctrl, 
                       tuneGrid = myGrid,
                       metric="RMSE")
    
    if(model=="rf"){
      MDL<-modelspace$finalModel
      MDLpred<-predict(MDL, data.test)
      data.test$predictions<-MDLpred
      submission.chunk<-data.test[,names(data.test)%in%c("X","predictions")]
      submission.chunk.path<-paste("/FogNets/",round.n,"/", as.character(i), "/","predictions",as.character(i),".csv", sep="")
      
      # save models
      env.path1<-paste("/FogNets/",round.n,"/", as.character(i), "/",model,as.character(i),".R", sep="")
      save(MDL,file=env.path1)  
    }
    if(model=="xgbTree"){
      MDLpred<-predict(modelspace, data.test)
      data.test$predictions<-MDLpred
      submission.chunk<-data.test[,names(data.test)%in%c("X","predictions")]
      submission.chunk.path<-paste("/FogNets/",round.n,"/", as.character(i), "/","predictions",as.character(i),".csv", sep="")
    }
    write.csv(submission.chunk,submission.chunk.path)
    
    # save models
    env.path2<-paste("/FogNets/",round.n,"/", as.character(i), "/",model,"space",as.character(i),".R", sep="")
    save(modelspace,file=env.path2)
  }
}


# FUNCTION 2 - fog.bindpreds
# Description: f() binds together predictions obtained from different chunks 
# Arguments: round.path (path to the current round folder), preds.begin, preds.end (number of begin and end chunk)

fog.bindpreds<-function(round.path, preds.begin, preds.end){
  first.path<-paste(round.path,"/",as.character(preds.begin),"/","predictions",as.character(preds.begin),".csv",sep="")
  print(first.path)
  base.file<-read.csv(first.path)
  for(i in (preds.begin+1):preds.end){
    tmp.path<-paste(round.path,"/",as.character(i),"/","predictions",as.character(i),".csv",sep="")
    print(tmp.path)
    tmp.file<-read.csv(tmp.path)
    base.file<-rbind(base.file, tmp.file)
  }
  neg<-base.file$predictions<0
  base.file$predictions[neg]<-0
  write.path=paste(round.path,"/bindpreds.csv",sep="")
  write.csv(base.file,write.path)
}

# FUNCTION 3 - fog.seqrmse
# Description: f() sequentally calculates RMSE on unseen observations from next chunk and plots results
# Arguments: round.n (round number integer), model.begin (model number to start from)
 # model.end (model number to stop), stepsahead.n (test models on data from x steps ahead of each model),
 # obs.type (1 to test only on observations > 0)

fog.seqrmse<-function(round.n,model, model.begin, model.end, stepsahead.n, obs.type=0){
  print("load the libraries!")
  performances<-c()
  round.path<-paste("/FogNets/Round",as.character(round.n),"/", sep="")
  for(i in model.begin:model.end){
    cat("* model",i,"*","\n")
    # load model
    model.path<-paste(round.path,as.character(i),"/",model,as.character(i),".R",sep="")
    mymodel<-load(model.path)
    # load data i
    data.path<-paste(round.path,as.character(i),"/","train_dwn",as.character(i),".csv",sep="")
    datai<-read.csv(data.path)
    base<-nrow(datai)
    #print(paste("Original training observations:",base))
    cat("Original training observations:",base,"\n")
    # load stepsahead
    basesteps<-i+stepsahead.n
    datasteps.path<-paste(round.path,as.character(basesteps),"/","train_dwn",as.character(basesteps),".csv",sep="")
    datasteps<-read.csv(datasteps.path)
    #print(paste("Next step training observations:",nrow(datasteps)))
    cat("Next step training observations:",nrow(datasteps),"\n")
    datasteps<-datasteps[base:nrow(datasteps), ]
    #print(paste("Testing observations:",nrow(datasteps)))
    cat("Testing observations:",nrow(datasteps),"\n")
    datasteps$Class<-ifelse(datasteps$yield>0,1,0)
    if(obs.type==1){
      datasteps<-datasteps[datasteps$yield>0,]
      #print(paste("Testing observations > 0:",nrow(datasteps)))
      cat("Testing observations > 0:",nrow(datasteps),"\n")
    }
    # predictions
    if(nrow(datasteps)>0){
      p<-predict(eval(parse(text=mymodel)), newdata=datasteps)
      performance.rmse<-RMSE(p,datasteps$yield) 
      cat("RMSE",performance.rmse,"\n")
      performances<-append(performances,performance.rmse)
    }
  }
  plot(c(model.begin:model.end),performances, main=paste("Round",round.n),  ylab = "RMSE", type = "l",col="blue")
  abline(h= mean(performances), col="purple")
  abline(v= c(model.begin:model.end), lty= 2, col="gray")
}


# FUNCTION 4 - fog.printmodels
 # Description: Given a round, print the models' parameters
 # Arguments: round.path (string entire path for a given Round), model(string either rf or xgb), model.begin model.end (numeric) 
fog.printmodels<-function(round.path,model,model.begin, model.end, save = TRUE){
  mds<-model.begin:model.end

  if(save==TRUE){
    file.path<-paste(round.path,"/tunedModels.txt", sep="")
  }

  for(i in mds){
    if(model=="rf"){
      
      model.path<-paste(round.path,"/",as.character(i),"/rf",as.character(i),".R",sep="")
      mymodel<-load(model.path)
      mymodel.text1<-paste(mymodel,"$ntree",sep="")
      mymodel.text2<-paste(mymodel,"$mtry",sep="")
      cat("Chunk",i,"ntree",model.results1,"mtry", model.results2,"\n")
      
      if(save==TRUE){
        cat("Chunk",i,"ntree",model.results1,"mtry", model.results2,"\n",file=file.path,append=TRUE)
      }
    }
    
    if(model=="xgb"){
      model.path<-paste(round.path,"/",as.character(i),"/xgbTreespace",as.character(i),".R",sep="")
      mymodel<-load(model.path)
      mymodel.text<-paste(mymodel,"$finalModel$tuneValue", sep="")
      model.results<-eval(parse(text=mymodel.text))
      cat("Chunk",i,unlist(model.results),"\n")
      
      if(save==TRUE){
        cat("Chunk",i,unlist(model.results),"\n",file=file.path,append=TRUE)
      }
    }
  }
}

