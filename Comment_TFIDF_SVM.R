library(tm)

a <- read.csv("C:/temp/BLACK-DECKER.csv", header = FALSE)

# TF-IDF
ds <- DataframeSource(a)
corpus <- Corpus(ds)
dtm = DocumentTermMatrix(corpus,control = list(weighting = weightTfIdf, stopwords = TRUE, removePunctuation = TRUE, stemming = TRUE)) 
# EXPORT
m <- as.matrix(dtm)   
dim(m)   
write.csv(m, file="C:/temp/a.csv")
# Find Frequent Terms (over 1000 times)
tdm = TermDocumentMatrix(corpus,control = list(stopwords = TRUE, removePunctuation = TRUE, stemming = TRUE)) 
findFreqTerms(tdm,1000)
# Find Associated Terms with correlation scores (the degree of conﬁdence)
findAssocs(tdm,"dust", 0.5) 
# result: buster --> a dust buster

# labeling
x.df = as.data.frame(as.matrix(dtm)) 
dim(x.df) 
x.df$class_label = c(rep(0,4500),rep(1,4862))
# with no specilfic classification basis XD

# Random Sampling and grouping
index = sample(nrow(x.df))
index
x.split = split(x.df, index >= 4500)
train = x.split[[1]]
test = x.split[[2]] 
dim(train)
dim(test)

# see the distribution 
table(train$class_label)
table(test$class_label)

# SVM
s = findFreqTerms(dtm,100) 
length(s) 
s = c(s,'class_label') 
View(train[,s])
# learn a logistic regression(邏輯迴歸) classiﬁer using the glm() function
# look at some of the model coefﬁcients using the coef() function
model.glm = glm(class_label ~ ., train[,s], family='binomial') 
coef(model.glm)
p = ifelse(predict(model.glm,test) < 0,0,1)  
table(p == test$class_label)/nrow(test)
# Accuraccy 74.4%

# svm() function of the e1071 package
library(e1071)
# kernel = radial
model.svm = svm(class_label ~ ., train[,s], kernel = 'radial', cost=2) 
model.svm

p.svm = ifelse(predict(model.svm,test) < 0.5,0,1) 
table(p.svm == test$class_label)/nrow(test)
# Accuracy 86.7%

# kernel = linear
model.svm2 = svm(class_label ~ ., train[,s], kernel = 'linear', cost=1) 
model.svm2

p.svm2 = ifelse(predict(model.svm2,test) < 0.5,0,1) 
table(p.svm2 == test$class_label)/nrow(test)
# Accuracy 71.9%

# sc: <<Beginning Data Science with R>> --Manas A. Pathak

