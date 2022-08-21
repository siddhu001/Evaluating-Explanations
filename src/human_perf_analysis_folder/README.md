## Human Performance

This part of the code will help you reproduce parts of Table 2 in the paper.

### Commands

To get Simulation Accuracy, go to `src` directory and run:

`python human_perf_analysis_folder/correct_guesses_script.py  --exp_type <Explanation_name>`

To get the percentage of examples flipped, go to `src` directory and run:

`python human_perf_analysis_folder/examples_flipped_script.py  --exp_type <Explanation_name>`

To get average confidence reduced, go to `src` directory and run:

`python human_perf_analysis_folder/change_in_confidence_script.py  --exp_type <Explanation_name>`

Explanation_name can be on of following: `log_reg_control`,`log_reg`,`none`,`lime`,`integrated_gradients`,`global_exp_ablation`,`global_exp`.
