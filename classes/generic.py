#TODO


# #' @docType methods
# setGeneric("compute", function(object,...) {standardGeneric("compute")})


# ##' generate N samples according to the network distribution
# ##' @name compute
# ##' @param N: the number of samples generated according to the network
# ##' @param object: a DAG.network object
# ##' @return a N*nNodes matrix
# ##' @export
# setMethod("compute", signature="DAG.network",  
#           function(object, N=50,bound=TRUE){
#             if(!is.numeric(N))
#               stop("N is not numeric")
#             save.seed <- get(".Random.seed", .GlobalEnv)
#             DAG = object@network
#             maxV= object@maxV
#             nNodes <- numNodes(DAG)
#             topologicalOrder <-tsort(DAG)
            
#             DD<-NULL
#             Nsamples<-0
            
#             #while (Nsamples < N & it <2){
#             D <- matrix(NA,nrow=N,ncol=nNodes)
#             colnames(D) <- 1:nNodes
            
            
#             for (ii in 1:length(topologicalOrder)){
              
#               i=topologicalOrder[ii]
#               bias = nodeData(DAG,n=i,attr="bias")[[1]]
#               sigma = nodeData(DAG,n=i,attr="sigma")[[1]]
#               seed = nodeData(DAG,n=i,attr="seed")[[1]]
#               inEdg <-  inEdges(node=i,object=DAG)[[1]]
              
#               if (length(inEdg)==0 ){
#                 set.seed(seed+ii+1)
#                 dr=rnorm(N,sd=object@exosdn)
#                 D[,i]<-bias + dr  #replicate(N,sigma())
#               } else  {
#                 D[,i]<-bias
#                 Xin<-NULL
#                 for (j in  inEdg){
#                   ## it computes the linear combination of the inputs
#                   inputWeight = edgeData(self=DAG,from=j,to=i,attr="weight")[[1]]
#                   H = edgeData(self=DAG,from=j,to=i,attr="H")[[1]]
                  
#                   if (object@additive){
#                     D[,i]<- D[,i] + H(D[,j]) *  inputWeight
#                   }else{
#                     ## it stacks inputs in a matrix
#                     Xin<-cbind(Xin,D[,j]*  inputWeight)
                    
#                   }
#                 }
#                 if (!object@additive){
#                   H = edgeData(self=DAG,from=inEdg[1],to=i,attr="H")[[1]]
#                   if (length(inEdg)==1)
#                     D[,i]<-  H(Xin)
#                   else
#                     D[,i]<-  H(apply(Xin,1,sum))
                  
#                 }
#                 set.seed(seed+ii)
                
#                 D[,i] <- (D[,i] + replicate(N,sigma()))  ## additive random noise
                
#               }
#             } ## for i
#             #col.numeric<-as(colnames(D),"numeric")
#             #D<-D[,topologicalOrder[order(col.numeric)]]
#             Dmax<-apply(abs(D),1,max)
#             wtoo<-union(which(Dmax>maxV),which(is.na(Dmax)))
#             if (length(wtoo)>0 & bound){
              
#               D=D[-wtoo,]
#             }
            
            
#             assign(".Random.seed", save.seed, .GlobalEnv)
            
            
#             return(D)
#           })

# #' @docType methods
# setGeneric("counterfact", function(object,...) {standardGeneric("counterfact")})
# ##' generate N samples according to the network distribution by modifying the original dataset
# ##' @name counterfact
# ##' @param DN: original dataset
# ##' @param knocked: the set of manipulated (e.g. knocked genes) nodes 
# ##' @param object: a DAG.network object
# ##' @return a N*nNodes matrix
# ##' @export
# setMethod("counterfact", signature="DAG.network",  