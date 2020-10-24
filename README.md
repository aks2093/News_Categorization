# News_Categorization
Data Set: https://drive.google.com/drive/folders/1hyz5WwE7nwlDZDRxI4uljkDwhx6DvVB7?usp=sharing


- Preprocessing steps performed on the dataset
  1. dropped the nan values from categories_mapping and news_details data set.
  2. merged the categories_mapping with news_details on news_id(inner join)
  
- Feature selection
  1. concatenated "snippet", "title", "news_description" in one combined column
  2. Used keras word embeddings on teh combined column(for LSTM and CNN)
  3. Used BERT encodings to fine-tune the BertSequenceClassifier

- Different types of models tested
  I tried below 3 models
	Below performance results were found:

	LSTM:

	  train_loss: 0.3022 
	  train_ccuracy: 0.9035
	  val_loss: 1.1194
	  val_accuracy: 0.7765
	  
	  Here we can see that this is a certain case of overfitting, 
	  But I tried out various experiments with drop_out rate , number of neurons/layers
	  but was not able to achieve more that 78% accuracy on validation data, while training with LSTMs

	CNN:

	  train_loss: 0.0425
	  train_accuracy: 0.9930
	  val_loss: 0.2438
	  val_accuracy: 0.9294

	  with CNN we are able to get better and more robust results than training with LSTMs but validation loss is still more.

	BERT:

	  train_loss: 0.06
	  train_accuracy: 0.99
	  validation_loss = 0.14
	  validation_accuracy: 0.97
		
	  Here we can see that fine-tuned BertForSequenceClassification model is the best performer among all other classifiers.
	  This model is more robust as we are fine-tuning the pretrained BertForSequenceClassification model on our dataset.
	  Hence results with BertForSequenceClassification model are more robust and loss function values are now acceptable.
	 
- Model Tuning techniques used to give best results:
	Note: Choosing a good Learning rate and good optimizer is the most important part of the model training, 
		  if we have optimize only parameter then I would certainly choose learning rate.
		  
	1. Tried out SGD, Adam optimizer with various values of beta1, beta2  and momentum to reduce the noise in the loss function values graph
    2. Tried out various values of learning rate to get the optimal performance of loss function
	3. Tried learning rate scheduler for to fine tune BERT model
	4. Tried values number of neurons/layers to get the optimal perfomance of the model
	5. Performed various experiments with drop_out rate to overcome overfitting


