# Human-Subject Experiments for Evaluating Explanations

This is the code used for experiments in the following [paper](https://arxiv.org/abs/2112.09669): 

> Explain, Edit, and Understand: Rethinking User Study Design for Evaluating Model Explanations
> 
> *Siddhant Arora\*, Danish Pruthi\*, Norman Sadeh, William W. Cohen, Zachary C. Lipton, Graham Neubig*
> 
> The 36th AAAI Conference on Artificial Intelligence (AAAI-22).

For dependencies, please check the `environment.yml` file. To create the same conda environment you can run `conda env create -f environment.yml` (You might have to edit the prefix in the last line in the file.) Also, please look at `requirements.txt` file in `src` directory anr run `pip install -r requirements.txt` to install dependencies regarding user study.

Please refer to README files for our [user study](src/README.md).  

### Repo layout

    .
    ├── datasets                                            # All Datasets
    │   ├── hotel reviews                                   # Hotel reviews Dataset
    │   │   ├── original version                            # original version of hotel reviews dataset
    │   │   ├── additional_downloads                        # Downloaded by authors
    ├── src                                                 # All Source files
    │   ├── edits_highlighted_word_analysis_folder          # Show percentage of edit on top 20% highlighted word
    │   ├── global_stats_folder                             # Analysis regarding the usage of global cues
    │   ├── human_perf_analysis_folder                      # Outputs metrics of human understanding
    │   ├── mixed_effects_analysis_folder                   # Run mixed effects regression model
    │   ├── model_training                                  # Train ML model on reviews dataset
    │   ├── README.md                                       # Gives instruction for running user study
    └── README.md

### Note

If you use the code, please consider citing:

```
@article{arora2022explain,
  title={Explain, Edit, and Understand: Rethinking User Study Design for Evaluating Model Explanations},
  author={Arora, Siddhant and Pruthi, Danish and Sadeh, Norman and Cohen, William W and Lipton, Zachary C and Neubig, Graham},
  booktitle = {Thirty-Six AAAI Conference on Artificial Intelligence.},
  month = {February},
  year = {2022}
}
