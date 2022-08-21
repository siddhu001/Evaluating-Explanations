## Model Training 

This part of the code will help you reproduce parts of Table 1 in the paper.

### Commands 

For training BERT base cased model on the hotel reviews dataset, run  :

`python model_training/main.py --case_sensitive --data_dir ../datasets/hotel_reviews`

For calibrating BERT model :

`python model_training/calibrate_model.py --data_dir ../datasets/hotel_reviews  --load model_dumps/bert_cased.pth --case_sensitive`

For training logistic regression model on the hotel reviews dataset, run  :

`python model_training/train_log_reg.py --data_dir ../datasets/hotel_reviews`

For training linear student model using BERT predictions on the 13.7K hotel reviews downloaded by authors, run  :

`python model_training/train_log_reg_bert_tokeniser.py`
