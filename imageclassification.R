#Basic Image Classification
library(keras)
fmnist=dataset_fashion_mnist()
str(fmnist)
c(train_image,train_label) %<-%fmnist$train
test_image=fmnist$test$x
test_label=fmnist$test$y

dim(train_image)
dim(train_label)
dim(test_image)
dim(test_label)
str(train_image)
sum(is.na(train_image))
unique(test_label)
unique(train_label)
class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

#preprocess

library(tidyr)
library(ggplot2)
image_1=data.frame(train_image[1,,])
#convert into long dataset(pivot)
colnames(image_1)=seq_len(ncol(image_1))
image_1$y=as.numeric(rownames(image_1))
image_1 = gather(image_1, "x", "value", -y)
str(image_1)
image_1$x <- as.integer(image_1$x)
plot(image_1$x,-image_1$y,col=image_1$value)

train_image=train_image/255
test_image=test_image/255

par(mfcol=c(6,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:30){
  image_i=data.frame(train_image[i,,])
  image_i=t(apply(image_i, 2, rev)) 
  image(1:28, 1:28, image_i, col = gray((0:255)/255),xaxt = 'n', yaxt = 'n',
         main = paste(class_names[train_label[i]+1]))
}

#set up the learning layers
model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%                          # transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array
  layer_dense(units = 128, activation = 'relu') %>%                   #128 nodes (or neurons)
  layer_dense(units = 10, activation = 'softmax')                     #0-node softmax layer.returns an array of 10 probability scores that sum to 1
#compile
model %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)
 
#train the model
fit(model,train_image,train_label,epochs =5,verbose=2)
#EVALUATE ACCURACY
evaluate(model,train_image,train_label, verbose = 0)
evaluate(model,test_image,test_label, verbose = 0) ###overfitting???
#predict
p=predict(model,test_image)
img_class=predict_classes(model,test_image)

