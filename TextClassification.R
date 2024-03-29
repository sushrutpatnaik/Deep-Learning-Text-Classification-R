data <- readLines("http://utdallas.edu/~sxp175331/sentimentdataset/amazon_cells_labelled.txt")

fields<- strsplit(data, split = "\t")

library(tm)
for(i in 1:length(fields))
{
  fields[[i]][1]<-tolower(fields[[i]][1])
  fields[[i]][1]<-removeWords(fields[[i]][1], stopwords())#removes stopwords
  fields[[i]][1]<-gsub(pattern="\\W", replace=" ", fields[[i]][1]) #removes punctuation
  fields[[i]][1]<-gsub(pattern="\\d", replace=" ", fields[[i]][1]) #removes numbers
  fields[[i]][1]<-gsub(pattern="\\b[A-z]\\b{1}", replace=" ", fields[[i]][1]) #removes single letters  
  fields[[i]][1]<-stripWhitespace(fields[[i]][1]) #removes white spaces
  fields[[i]][1]<-trimws(fields[[i]][1])  #trims leading and trailing whitespaces
}


library(stringr)

k=1
word_list<-list()
label_list<-list()

#store the words and labels in 2 different lists
for(i in 1:length(fields))
{
  temp_list<-str_split(fields[[i]][1], pattern="\\s+")
  j=1
  while(j<=length(temp_list[[1]]))
  {
    word_list[k]<-temp_list[[1]][j]
    label_list[k]<-fields[[i]][2]
    j=j+1
    k=k+1
    
  }
}

#word_list
#label_list
word_list<-unlist(word_list)
label_list<-unlist(label_list)
label_list<-as.numeric(label_list)

# Creates a tokenizer, configured to only take into account the 10000 most common words, then builds the word index
library(keras)
tokenizer <- text_tokenizer(num_words = 10000) %>%
  fit_text_tokenizer(word_list)

#turns strings into lists of integer indices
library(keras)
sequences <- texts_to_sequences(tokenizer, word_list)

newlist<-unlist(sequences)



require(caTools)
set.seed(25) 
sample = sample.split(newlist, SplitRatio = .75)

train_data = subset(newlist, sample == TRUE)
test_data  = subset(newlist, sample == FALSE)
train_labels = subset(label_list, sample == TRUE)
test_labels  = subset(label_list, sample == FALSE)



#function to vectorize the input data to the Network
vectorize_sequences <- function(sequences,
                                dimension = 10000) {
  # Create an all-zero matrix of shape (len(sequences), dimension)
  results <- matrix(0, nrow = length(sequences),
                    ncol = dimension)
  for (i in 1:length(sequences))
    # Sets specific indices of results[i] to 1s
    results[i, sequences[[i]]] <- 1
  results
}

#Vectorized training data
x_train <- vectorize_sequences(train_data)
#Vectorized test data
x_test <- vectorize_sequences(test_data)

#Vectorized Labels
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)


#Validation dataset
val_indices <- 1:1500
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

#############################################################################################
#model 1
library(keras)
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu",
              input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(optimizer = "rmsprop",loss = "binary_crossentropy", metrics = c("accuracy"))

model %>% fit(x_train, y_train, epochs = 6,batch_size = 512)


#plot of history based on validation data
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

results <- model %>% evaluate(x_test, y_test)
#############################################################################################


#model 2
library(keras)
model2 <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu",
              input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model2 %>% compile(optimizer = "rmsprop",loss = "mse", metrics = c("accuracy"))

model2 %>% fit(x_train, y_train, epochs = 10,batch_size = 512)


#plot of history based on validation data
history <- model2 %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

results2 <- model2 %>% evaluate(x_test, y_test)
#############################################################################################

#model 3
library(keras)
model3 <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu",
              input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "tanh") %>%
  layer_dense(units = 16, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")

model3 %>% compile(optimizer = "rmsprop",loss = "poisson", metrics = c("accuracy"))

model3 %>% fit(x_train, y_train, epochs = 10,batch_size = 512)



#plot of history based on validation data
history <- model3 %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

results3 <- model3 %>% evaluate(x_test, y_test)

#############################################################################################

#model 4
library(keras)
model4 <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "tanh",
              input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "tanh") %>%
  layer_dense(units = 16, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")

model4 %>% compile(optimizer = "rmsprop",loss = "poisson", metrics = c("accuracy"))

model4 %>% fit(x_train, y_train, epochs = 10,batch_size = 512)


#plot of history based on validation data
history <- model4 %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

results4 <- model4 %>% evaluate(x_test, y_test)
#############################################################################################


#model 5
model5 <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "tanh") %>%
  layer_dense(units = 16, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")

model5 %>% compile(
  optimizer = "rmsprop",
  loss = "poisson",
  metrics = c("binary_accuracy")
)

model5 %>% fit(x_train, y_train, epochs = 10,batch_size = 512)


#plot of history based on validation data
history <- model5 %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

results5 <- model5 %>% evaluate(x_test, y_test)
#############################################################################################


