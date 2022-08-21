## Percentage of Edit Statistics 

This part of the code will help you reproduce parts of Table 4 and Figure 2 in the paper.

### Commands 

To get the percentage of first edits performed on the top 20% highlighted words, go to `src` directory and run:

`python edits_highlighted_word_analysis_folder/word_edit_distance_highlighted_first_script.py  --exp_type <Explanation_name>`

To get the percentage of all edits performed on the top 20% highlighted words and their contribution towards the confidence reduction, go to `src` directory and run:

`python edits_highlighted_word_analysis_folder/word_edit_distance_highlighted_average_script.py --exp_type <Explanation_name>`

Explanation_name can be on of following: `log_reg`,`lime`,`integrated_gradients`,`global_exp_ablation`.
