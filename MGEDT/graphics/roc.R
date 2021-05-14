library(rminer)
C1<<- "black"
C2<<- "blue"
myplot=function(d2,d3,name,models)
{
L=vector("list",2); 
pred=vector("list",1); test=vector("list",1);
test[[1]]=d2$Y
Y=as.factor(d2$Y)
#pred[[1]]=d1[,2]
#M1=list(pred=pred,test=test,runs=1)
pred[[1]]=d2[,2]
M2=list(pred=pred,test=test,runs=1)
pred[[1]]=d3[,2]
M3=list(pred=pred,test=test,runs=1)

L[[1]]=M2; L[[2]]=M3
pdf(name,width=5,height=5)
par(mar=c(4.0,4.0,0.2,0.2))
X=cbind(1-M2$pred[[1]],M2$pred[[1]])
auc2=mmetric(Y,X,metric="AUC",TC=2)
X=cbind(1-M3$pred[[1]],M3$pred[[1]])
auc3=mmetric(Y,X,metric="AUC",TC=2)
leg2=paste("AUC=",round(c(auc2,auc3)*100,2),sep="")
leg1=paste(models,leg2,sep=": ")
print(leg1)
mgraph(L,graph="ROC",TC=2,lty=c(2,1),leg=list(pos=c(0.4,0.2),leg=leg1),baseline=TRUE,Grid=10,col=c(C1,C2))
dev.off()
}

myplot2=function(d2,d3,name,models)
{
L=vector("list",3); 
L$runs=3
pred=vector("list",1); test=vector("list",1);
test[[1]]=d2$Y
Y=as.factor(d2$Y)

pdf(name,width=5,height=5)
par(mar=c(4.0,4.0,0.2,0.2))

X=cbind(1-d2[,2],d2[,2])
X=data.frame(X)
names(X)=c("0","1")
l2=mmetric(Y,X,metric="LIFT",TC=2)
pred[[1]]=X
M2=list(pred=pred,test=Y,runs=1)

X=cbind(1-d3[,2],d3[,2])
X=data.frame(X)
names(X)=c("0","1")
l3=mmetric(Y,X,metric="LIFT",TC=2)
pred[[1]]=X
M3=list(pred=pred,test=Y,runs=1)

L[[1]]=M2; L[[2]]=M3

leg2=paste("ALIFT=",round(c(l2$lift$area,l3$lift$area)*100,2),sep="")
leg1=paste(models,leg2,sep=": ")
print(leg1)
#mgraph(L,graph="LIFT",TC=2,lty=c(3,1,2),leg=list(pos=c(0.4,0.2),leg=leg1),baseline=TRUE,Grid=10)
plot(l2$lift$alift,xlim=c(0,1),ylim=c(0,1),lwd=2,type="l",xlab="Sample size",ylab="Responses",lty=2,panel.first=grid(10,10), col=C1)
#lines(l2$lift$alift,lty=1,lwd=2)
lines(l3$lift$alift,lty=1,lwd=2, col=C2)
legend("bottomright",leg1,lwd=2,lty=c(2,1))
dev.off()
}


mode="BEST2"
iter=7
		
#d1=read.table(paste0("DT/",mode,"/IDF/iteration",iter,"/preds.csv"),header=TRUE, sep=";")
d2=read.table(paste0("results/MOGEDT-",mode,"/iteration",iter,"/25_front/0_predictions.csv"),header=TRUE, sep=";")
d3=read.table(paste0("results/MOGEDTLM-",mode,"/iteration",iter,"/25_front/0_predictions.csv"),header=TRUE, sep=";")

myplot(d2,d3,paste0("graphics/roc/multi-",mode,"-roc-",iter,".pdf"), c("MGEDT","MGEDTL"))
myplot2(d2,d3,paste0("graphics/roc/multi-",mode,"-lift-",iter,".pdf"), c("MGEDT","MGEDTL"))


