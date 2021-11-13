library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)
library(dplyr)
library(h2o)
library(inspectdf)

df <- fread("nlpdata.csv")
df %>% inspect_na()
df <- df %>% rename(id = V1)

df$id %>% class()
df$id <- df$id %>% as.character()


# Split data
set.seed(123)
split <- df$Liked %>% sample.split(SplitRatio = 0.8)
train <- df %>% subset(split == T)
test <- df %>% subset(split == F)

it_train <- train$Review %>%tolower() %>% word_tokenizer() %>% itoken(ids = train$id,
                                                                      progressbar = F)

vocab <- it_train %>% create_vocabulary()
vocab %>% 
  arrange(desc(term_count)) %>% 
  head(100) 

vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

identical(rownames(dtm_train), train$id)

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,# high value is less accurate, but has faster training
            maxit = 1000)# again lower number of iterations for faster training

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

it_test <- test$Review %>% tolower() %>% word_tokenizer() %>% itoken(ids = test$id,progressbar = F)
dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$Liked, preds) %>% round(3)


stop_words <- c("i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "them",
                "my", "your", "yours", "his", "our", "ours",
                "myself", "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "or", "on", "by", "so",
                "from", "about", "to", "for", "of", 
                "that", "this", "is", "are")

vocab <- it_train %>% create_vocabulary(stopwords = stop_words)

pruned_vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5,
                   doc_proportion_min = 0.001)

pruned_vocab %>% 
  arrange(desc(term_count)) %>% 
  head(10)

vectorizer <- pruned_vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(vectorizer)


glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 4,
            thresh = 0.001,
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$Liked, preds) %>% round(2)
