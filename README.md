# FDKT: Towards an interpretable deep knowledge tracing via fuzzy reasoning

This is our implementation for the paper titled **FDKT: Towards an interpretable deep knowledge tracing via fuzzy reasoning**

**Author**: Fei Liu, Chenyang Bu, Haotian Zhang, Le Wu, Kui Yu, and Xuegang Hu

**Abstract**: In educational data mining, knowledge tracing (KT) aims to model learning performance based on student knowledge mastery. Deep-learning-based KT models perform remarkably better than traditional KT and have attracted considerable attention. However, most of them lack interpretability, making it challenging to explain why the model performed well in the prediction. In this paper, we propose an interpretable deep KT model, referred to as fuzzy deep knowledge tracing (FDKT) via fuzzy reasoning. Specifically, we formalize continuous scores into several fuzzy scores using the fuzzification module. Then, we input the fuzzy scores into the fuzzy reasoning module (FRM). FRM is designed to deduce the current cognitive ability, based on which the future performance was predicted. FDKT greatly enhanced the intrinsic interpretability of deep-learning-based KT through the interpretation of the deduction of student cognition. Furthermore, it broadened the application of KT to continuous scores. Improved performance with regard to both the advantages of FDKT was demonstrated through comparisons with 15 state-of-the-art models, using 4 real-world datasets.


    
##   Environment Settings
### Dependencies
- python==3.9
- torch==1.7
- torchnet

 ### Devices
 Our model supports running on both CPU and GPU devices. If you wish to run it on a CPU, please set `use_gpu = False` in the *config.py file*. Otherwise, please set `use_gpu = True`.
 
  ##   Hyper-parameters
  The majority of our hyperparameters are set in the config.py file. Please modify them according to your needs. Below are the default settings for some of the hyperparameters. Please note that these values can be adjusted as per your specific requirements.
```
    term_numbers = 6
    cog_numbers = 6
    rule_numbers = term_numbers * cog_numbers
    weight_decay = 0.001
    learning_rate = 0.01
    batch_size = 128
```
 
 ##   Run Steps
 If you want to run this program on the command line, please switch to the directory where main.py is located firstly.
For example:
```
    cd dir
    python main.py
```


### Datasets
The model has been proposed to solve the KT task in a continuous scoring scenario with subjective questions. Therefore, the dataset you use needs to include the following information: student id, exercise id, knowledge point id examined in the exercise, and the continuous score of the student answering the exercise.

 ### Data Format
Data folder contains two files：`training.txt` and `testing.txt`. In each files, there are response records of different students included. Here is an example of the data format in each file:
```
8
5 6 1 5 1 1 6 8 
0.2 1.0 0.0 0.5 0.4 0.8 0.7 0.0
5
5 3 2 2 5
0.5 0.4 0.8 0.5 1.0
```
The example shows the interaction record sequences of two students. Each student has three lines of data. The first line is an integer that represents the total number of interaction steps for that student. The second line represents the exercise IDs of the student's interactions at each time step. The third line represents the scores of the student's interactions at each time step.

##   Cite
@article{liu2024fuzzykt,
  title={FDKT: Towards an interpretable deep knowledge tracing via fuzzy reasoning},
  author={Fei Liu, Chenyang Bu, Haotian Zhang, Le Wu, Kui Yu, and Xuegang Hu},
  year={2023}
}
