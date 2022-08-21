## Mixed Effect Analysis

This part of the code will help you reproduce parts of Table 3 in the paper.

### Commands 

To run mixed effects regression models for `BERT` based classifier, go to `src` directory and run:

`python mixed_effects_analysis_folder/mixed_run_mixed_effects_model.py`

Once the runs are complete, you should be able to check the last few output lines.

```
Explanation_Type_LIME            -0.029      0.029 -0.990 0.322       -0.086       0.028
Explanation_Type_IG               0.010      0.028  0.356 0.722       -0.046       0.066
Explanation_Type_global           0.143      0.028  5.061 0.000        0.087       0.198
Explanation_Type_global_ablation  0.050      0.029  1.731 0.083       -0.007       0.107
Group Var                         0.000                                                 
```

To run mixed effects regression models for the `Logistic Regression` model, go to the `src` directory and run:

`python mixed_effects_analysis_folder/mixed_run_mixed_effects_model_LR.py`

Once the runs are complete, you should be able to check the last few output lines.

```
Explanation_Type_Logreg 0.029       0.016 1.834 0.067       -0.002       0.060
Group Var               0.000                                                 
```

Lastly, note that the fixed effect term is produced for the simulation accuracy ratio, and to compute the fixed effect coefficient for the simulation accuracy percentage, multiply the `Coef` by 100.
