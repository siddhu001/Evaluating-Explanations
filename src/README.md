## Run User Study Design for Evaluating Model Explanations

This part of the code will help you run the user study for Evaluating Explanations.

 Go to src 
```
 pip install -r requirements.txt
```
To run Logistic Regression control
```
 python app.py --explanation_type log_reg --data_dir ../datasets/hotel_reviews/ --filter True --control True
```

Upon executing this command, the user study would be hosted at http://localhost:<port_number>/. You can view it in your favourite browser. Similarly, the following commands would spawn different user studies at (possibly) different ports, which you would be able to see as an output on the console. 


To run Logistic Regression with feature coefficient explanation as treatment
```
 python app.py --explanation_type log_reg --data_dir ../datasets/hotel_reviews/ --filter True
```
To run BERT control
```
 python app.py --explanation_type none --data_dir <review_dataset> --filter True
```
To run LIME treatment with BERT
```
 python app.py --explanation_type lime --data_dir <review_dataset> --filter True
```
To run Integrated Gradients treatment with BERT
```
python app_IG.py --explanation_type integrated_gradients --data_dir ../datasets/hotel_reviews/ --filter True
```
To run feature coefficients from linear model treatment with BERT
```
 python app.py --explanation_type none --global_exp True --data_dir ../datasets/hotel_reviews/ --filter True --no_global_words True 
```
To run global cues from linear model treatment with BERT
```
 python app.py --explanation_type none --global_exp True --data_dir ../datasets/hotel_reviews/ --filter True 
```

### Models

BERT and logistic regression model that were used in our experiments are available at  https://drive.google.com/drive/folders/1np3zcMgiqNomg27rTQ8sV2JtVwDqxt1S?usp=sharing


### User Data

All database files saved to record user activity are available at https://drive.google.com/file/d/1g9EQpyG0dlfl6n2amy02mld8SGU3vUWM/view?usp=sharing. Unzip the database folder inside src/ directory to run analysis script.

### Analysis

Please refer to README files in the subdirectories to reproduce our results and Analysis.

### Contact:

If you need any support, please add an issue on the repository, and we'll respond to it. 
