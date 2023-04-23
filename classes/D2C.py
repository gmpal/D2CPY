
class D2C:
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs
        self.orig_X = None
        self.orig_Y = None

    

    # def is_what(self, iDAG, i, j, type):
    #     if type == "is_mb":
    #         return int(is_mb(iDAG, i, j))
    #     elif type == "is_parent":
    #         return int(is_parent(iDAG, i, j))
    #     elif type == "is_child":
    #         return int(is_child(iDAG, i, j))
    #     elif type == "is_descendant":
    #         return int(is_descendant(iDAG, i, j))
    #     elif type == "is_ancestor":
    #         return int(is_ancestor(iDAG, i, j))

    # def create_trainset(self):
    #     random.seed(self.iteration_counter)
    #     self.train_DAG = SimulatedDAG()
    #     self.train_DAG.generate(self.number_nodes, self.number_samples, self.number_features, self.max_s)
    #     self.train_DAG.generate_data()
    #     self.train_DAG.generate_DAG()




#     #' @docType methods
# setGeneric("makeModel", def=function(object,...) {standardGeneric("makeModel")})

# #' creation of a D2C object which preprocesses the list of DAGs and observations contained in sDAG and fits a  Random Forest classifier
# #' @name  D2C object
# #' @param .Object : the D2C object
# #' @param sDAG : simulateDAG object
# #' @param descr  : D2C.descriptor object containing the parameters of the descriptor
# #' @param max.features  : maximum number of features used by the Random Forest classifier \link[randomForest]{randomForest}. The features are selected by the importance returned by the function \link[randomForest]{importance}.
# #' @param ratioEdges  : percentage of existing edges which are added to the training set
# #' @param ratioMissingNode  : percentage of existing nodes which are not considered. This is used to emulate latent variables.
# #' @param goParallel : if TRUE it uses parallelism
# #' @param subset : if bivar it uses only bivariate descriptors
# #' @param verbose  : if TRUE it prints the state of progress
# #' @param EErep: Easy Ensemble size to deal with unbalancedness
# #' @references Gianluca Bontempi, Maxime Flauder (2015) From dependency to causality: a machine learning approach. JMLR, 2015, \url{http://jmlr.org/papers/v16/bontempi15a.html}
# #' @examples
# #' @export
# setMethod("makeModel",
#           "D2C",
#           function(object, max.features=30, names.features=NULL,
#                    classifier="RF",
#                    EErep=5,verbose=TRUE,subset="all") {
            
#             X<-object@origX
#             Y<-object@Y
            
#             features=1:NCOL(X)
            
#             if (subset=="bivar")
#               features<- grep('B.',colnames(X))
#             if (subset=="multivar")
#               features<- grep('M.',colnames(X))
#             if (subset=="noInt"){
#               featInt<- grep('Int.',colnames(X))
#               features<-setdiff(features,featInt)
#             }
            
#             X<-X[,features]
#             wna<-which(apply(X,2,sd)<0.001)
#             if (length(wna)>0){
#               features<-setdiff(features,features[wna])
#               X=X[,-wna] 
#             }
#             wna<-which(is.na(apply(X,2,mean)))
#             if (length(wna)>0){
#               features<-setdiff(features,features[wna])
#               X=X[,-wna] 
#             }
            
#             X<-scale(X)
#             N=NROW(X)
#             object@scaled=attr(X,"scaled:scale")
#             object@center=attr(X,"scaled:center")
#             object@classifier=classifier
            
#             object@features=features
            
            
#             listRF<-list()
#             I=sample(1:N,min(N-1,20000))
#             #featrank<-mrmr(X ,factor(Y),min(NCOL(X),3*max.features))
#             if (classifier=="RF"){
#               if (object@type=="is.distance"){
#                 RF <- randomForest(x =X[I,] ,y = Y[I],importance=TRUE)
#                 IM<-importance(RF)[,"%IncMSE"]
#               } else {
#                 RF <- randomForest(x =X[I,] ,y = factor(Y[I]),importance=TRUE)
#                 IM<-importance(RF)[,"MeanDecreaseAccuracy"]
#               }
#             }
#             if (classifier=="XGB.1"){
#               RF <- xgboost(data =X[I,] ,label = Y[I],nrounds=20,objective = "binary:logistic",eta=0.1)
#               IM<-numeric(NCOL(X))
#               names(IM)=colnames(X)
#               IM[xgb.importance(model = RF)[,1]]=xgb.importance(model = RF)[,2]
#             }
#             if (classifier=="XGB.2"){
#               RF <- xgboost(data =X[I,] ,label = Y[I],nrounds=20,objective = "binary:logistic",eta=0.2)
#               IM<-numeric(NCOL(X))
#               names(IM)=colnames(X)
#               IM[xgb.importance(model = RF)[,1]]=xgb.importance(model = RF)[,2]
#             }
#             featrank<- sort(IM,decr=TRUE,ind=TRUE)$ix
#             if (verbose)
#               cat("Best descriptors: ", colnames(X)[featrank], "\n")
            
#             ratio=3
#             if (classifier=="RF")
#               ratio=1.5
            
            
#             for (rep in 1:EErep){
#               w0<-which(Y==0)
#               w1<-which(Y==1)
#               if (length(w0)>=ratio*length(w1))
#                 w0<-sample(w0,round(ratio*length(w1)))
              
#               if (length(w1)>=length(w0))
#                 w1<-sample(w1,round(length(w0)))
#               Xb<-X[c(w0,w1),]
#               Yb<-Y[c(w0,w1)]
              
              
#               rank<-featrank
#               rank<-rank[1:min(max.features,length(rank))]
#               Xb=Xb[,rank]
#               if (classifier=="RF"){
#                 if (object@type=="is.distance")
#                   RF <- randomForest(x =X[,rank] ,y = Y)
#                 else
#                   RF <- randomForest(x =Xb ,y = factor(Yb))
#               }
#               if (classifier=="XGB.1")
#                 RF=xgboost(data =Xb ,label = Yb,nrounds=5,objective = "binary:logistic",eta=0.1)
#               if (classifier=="XGB.2")
#                 RF=xgboost(data =Xb ,label = Yb,nrounds=5,objective = "binary:logistic",eta=0.2)
              
#               listRF<-c(listRF,list(list(mod=RF,feat=rank)))
#               if (verbose)
#                 cat(classifier, " ", rep, ":",RF$confusion, " : (N,n)=", dim(Xb), "\n")
#             } ## for rep
            
#             object@mod=listRF
            
#             object
#           }
# )


# #' predict if there is a connection between node i and node j
# #' @param object : a D2C object
# #' @param i :  index of putative cause (\eqn{1 \le i \le n})
# #' @param j  : index of putative effect (\eqn{1 \le j \le n})
# #' @param data : dataset of observations from the DAG
# #' @return list with binary response and probability of the existence of a directed edge
# #' @examples
# #' require(RBGL)
# #' require(gRbase)
# #' require(foreach)
# #' data(example)
# #'## load the D2C object
# #' testDAG<-new("simulatedDAG",NDAG=1, N=50,noNodes=5,
# #'            functionType = "linear", seed=1,sdn=c(0.25,0.5))
# #' ## creates a simulatedDAG object for testing
# #' plot(testDAG@@list.DAGs[[1]])
# #' ## plot the topology of the simulatedDAG
# #' predict(example,1,2, testDAG@@list.observationsDAGs[[1]])
# #' ## predict if the edge 1->2 exists
# #' predict(example,4,3, testDAG@@list.observationsDAGs[[1]])
# #' ## predict if the edge 4->3 exists
# #' predict(example,4,1, testDAG@@list.observationsDAGs[[1]])
# #' ## predict if the edge 4->1 exists
# #' @references Gianluca Bontempi, Maxime Flauder (2015) From dependency to causality: a machine learning approach. JMLR, 2015, \url{http://jmlr.org/papers/v16/bontempi15a.html}
# #' @export
# setMethod("predict", signature="D2C",
#           function(object,i,j,data, rep=1){ 
#             out = list()
            
#             if (any(apply(data,2,sd)<0.01))
#               stop("Error in D2C::predict: Remove constant variables from dataset. ")
#             Response<-NULL
#             Prob<-NULL
            
#             for (repS in 1:rep){
#               ## repetition with different subsets of other variables
#               others=setdiff(1:NCOL(data),c(i,j))
#               if (repS>1)
#                 others=sample(others, round(2*length(others)/3))
#               D=data[,c(i,j,others)]
#               # move the concerned variables to the first two places
              
#               #X_descriptor = descriptor(data,i,j,
#               X_descriptor = descriptor(D,1,2,
#                                         lin = object@descr@lin,
#                                         acc = object@descr@acc,
#                                         ns=object@descr@ns,
#                                         maxs=object@descr@maxs,
#                                         struct = object@descr@struct,
#                                         pq = object@descr@pq, 
#                                         bivariate =object@descr@bivariate, 
#                                         boot=object@descr@boot,
#                                         errd=object@descr@residual, delta=object@descr@diff,
#                                         stabD=object@descr@stabD)
              
#               if (any(is.infinite(X_descriptor)))
#                 stop("Error in D2C::predict: infinite value ")
#               X_descriptor=X_descriptor[object@features]
              
#               X_descriptor=scale(array(X_descriptor,c(1,length(X_descriptor))),
#                                  object@center,object@scaled)
#               if (any(is.infinite(X_descriptor)))
#                 stop("error in D2C::predict")
              
#               for (r in 1:length(object@mod)){
#                 mod=object@mod[[r]]$mod
#                 fs=object@mod[[r]]$feat
#                 #Response = c( Response, predict(mod, X_descriptor[fs], type="response"))
#                 if (object@classifier=="RF"){
#                   if (object@type=="is.distance"){
                    
#                     Prob = c(Prob,predict(mod, X_descriptor[fs]))
#                   } else
#                     Prob = c(Prob,predict(mod, X_descriptor[fs], type="prob")[,"1"])
#                 }
#                 if (length(grep("XGB",object@classifier))>=1)
#                   Prob = c(Prob,predict(mod, array(X_descriptor[fs],c(1,length(fs)))))
                
#               }
#             }
#             if (object@type=="is.distance"){
#               out[["response"]] =mean(Prob)
#             } else{
#               out[["response"]] =round(mean(Prob))
#               out[["prob"]]=mean(Prob)
#             }
#             return(out)
#           })


# #' @docType methods
# setGeneric("joinD2C", def=function(object,...) {standardGeneric("joinD2C")})

# #' update of a "D2C" with a list of DAGs and associated observations
# #' @name join current D2C and input D2C
# #' @param object :  D2C to be updated
# #' @param input :  D2C to be joined
# #' @param verbose : TRUE or FALSE
# #' @param goParallel : if TRUE it uses  parallelism
# #' @export
# setMethod(f="joinD2C",
#           signature="D2C",
#           definition=function(object,input,
#                               verbose=TRUE, goParallel= FALSE){
            
#             `%op%` <- if (goParallel) `%dopar%` else `%do%`
            
#             X<-rbind(object@origX,input@origX)
#             Y<-c(object@Y,input@Y)
#             features<-intersect(object@features,input@features)
            
            
#             features<-1:NCOL(X)
#             wna<-which(apply(X,2,sd)<0.01)
#             if (length(wna)>0)
#               features<-setdiff(features,wna)
#             object@origX=X
#             X<-scale(X[,features])
#             object@scaled=attr(X,"scaled:scale")
#             object@center=attr(X,"scaled:center")
            
#             object@features=features
#             object@Y=Y
#             max.features=object@max.features
            
#             #listRF<-list()
#             #for (rep in 1:10){
#             #  w0<-which(Y==0)
#             #  w1<-which(Y==1)
#             #  if (length(w0)>length(w1))
#             #    w0<-sample(w0,length(w1))
            
#             #  if (length(w1)>length(w0))
#             #    w1<-sample(w1,length(w0))
#             #  Xb<-X[c(w0,w1),]
#             #  Yb<-Y[c(w0,w1)]
            
#             #  browser()
#             #  if (object.type=="is.distance")
#             #    RF <- randomForest(x =X ,y = Y,importance=TRUE)
#             #  else 
#             #    RF <- randomForest(x =Xb ,y = factor(Yb),importance=TRUE)
#             #  IM<-importance(RF)[,"MeanDecreaseAccuracy"]
#             #  rank<-sort(IM,decr=TRUE,ind=TRUE)$ix[1:min(max.features,NCOL(Xb))]
#             #  Xb=Xb[,rank]
            
#             #  RF <- randomForest(x =Xb ,y = factor(Yb))
            
#             #  listRF<-c(listRF,list(list(mod=RF,feat=rank)))
#             #}
            
#             # object@mod=listRF
            
#             object
#           }
# )




# #' @docType methods
# setGeneric("updateD2C", def=function(object,...) {standardGeneric("updateD2C")})

# #' update of a "D2C" with a list of DAGs and associated observations
# #' @name update D2C
# #' @param object :  D2C to be updated
# #' @param sDAG : simulatedDAG object to update D2C
# #' @param verbose : TRUE or FALSE
# #' @param goParallel : if TRUE it uses  parallelism
# #' @export
# setMethod(f="updateD2C",
#           signature="D2C",
#           definition=function(object,sDAG,
#                               verbose=TRUE, goParallel= FALSE){
            
#             `%op%` <- if (goParallel) `%dopar%` else `%do%`
#             ratioMissingNode=object@ratioMissingNode
#             ratioEdges=object@ratioEdges
#             descr=object@descr
#             FF<-foreach (i=1:sDAG@NDAG) %op%{
              
#               set.seed(i)
#               DAG = sDAG@list.DAGs[[i]]
#               observationsDAG =sDAG@list.observationsDAGs[[i]]
              
#               Nodes = nodes(DAG)
              
#               sz=max(2,ceiling(length(Nodes)*(1-ratioMissingNode)))
#               keepNode = sort(sample(Nodes,
#                                      size = sz ,
#                                      replace = F))
              
#               DAG2 =subGraph(keepNode, DAG)
              
              
#               ##choose wich edge to train / predict and find the right label
#               nEdge = length(edgeList(DAG))
#               sz=max(1,round(nEdge*ratioEdges))
              
#               edgesM = matrix(unlist(sample(edgeList(DAG2),
#                                             size = sz,replace = F)),ncol=2,byrow = TRUE)
#               edgesM = rbind(edgesM,t(replicate(n =sz ,
#                                                 sample(keepNode,size=2,replace = FALSE))))
              
#               nEdges =  NROW(edgesM)
#               labelEdge = numeric(nEdges)
#               for(j in 1:nEdges){
#                 I =edgesM[j,1] ;
#                 J =edgesM[j,2] ;
#                 labelEdge[j] = as.numeric(I %in% inEdges(node = J,DAG2)[[1]])
#               }
              
              
#               ##compute the descriptor for the edges
#               nNodes = length(labelEdge)
              
#               X.out = NULL
#               for(j in 1:nNodes){
#                 I =as(edgesM[j,1],"numeric") ;
#                 J =as(edgesM[j,2],"numeric") ;
                
                
#                 d<-descriptor(observationsDAG,I,J,lin=descr@lin,acc=descr@acc,
#                               struct=descr@struct,bivariate=descr@bivariate,
#                               pq=descr@pq,maxs=descr@maxs,ns=descr@ns,boot=descr@boot,
#                               errd=descr@residual, delta=descr@diff, stabD=descr@stabD)
                
                
                
                
                
#                 X.out = rbind(X.out,d)
#               }
#               if (verbose)
#                 cat("D2C:  DAG", i, " processed \n")
              
#               list(X=X.out,Y=labelEdge,edges=edgesM)
              
#             }
            
#             X<-do.call(rbind,lapply(FF,"[[",1))
#             Y<-do.call(c,lapply(FF,"[[",2))
#             allEdges<-lapply(FF,"[[",3)
            
            
            
#             X<-scale(X[,object@features],attr(object@X,"scaled:center"),attr(object@X,"scaled:scale"))
            
#             object@X=rbind(object@X,X)
#             object@Y=c(object@Y,Y)
#             object@allEdges=c(object@allEdges,allEdges)
#             RF <- randomForest(x =object@X ,y = factor(object@Y),importance=TRUE)
#             IM<-importance(RF)[,"MeanDecreaseAccuracy"]
#             rank<-sort(IM,decr=TRUE,ind=TRUE)$ix[1:min(object@max.features,NCOL(X))]
#             RF <- randomForest(x =object@X[,rank] ,y = factor(object@Y))
#             object@rank=rank
#             object@mod=RF
            
#             object
            
#           }
# )