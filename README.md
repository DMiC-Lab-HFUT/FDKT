# FDKT: Towards an interpretable deep knowledge tracing via fuzzy reasoning

This is our implementation for the paper titled **FDKT: Towards an interpretable deep knowledge tracing via fuzzy reasoning**

**Author**: Fei Liu, Chenyang Bu, Haotian Zhang, Le Wu, Kui Yu, and Xuegang Hu

**Abstract**: In educational data mining, knowledge tracing (KT) aims to model learning performance based on student knowledge mastery. Deep-learning-based KT models perform remarkably better than traditional KT and have attracted considerable attention. However, most of them lack interpretability, making it challenging to explain why the model performed well in the prediction. In this paper, we propose an interpretable deep KT model, referred to as fuzzy deep knowledge tracing (FDKT) via fuzzy reasoning. Specifically, we formalize continuous scores into several fuzzy scores using the fuzzification module. Then, we input the fuzzy scores into the fuzzy reasoning module (FRM). FRM is designed to deduce the current cognitive ability, based on which the future performance was predicted. FDKT greatly enhanced the intrinsic interpretability of deep-learning-based KT through the interpretation of the deduction of student cognition. Furthermore, it broadened the application of KT to continuous scores. Improved performance with regard to both the advantages of FDKT was demonstrated through comparisons with 15 state-of-the-art models, using 4 real-world datasets.
    
##   Environment Settings
### Packages
- torch: ‘1.7.1’
- torchnet: `pip install torchnet`

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

##   Cite
@article{liu2024fuzzykt,
  title={FDKT: Towards an interpretable deep knowledge tracing via fuzzy reasoning},
  author={Fei Liu, Chenyang Bu, Haotian Zhang, Le Wu, Kui Yu, and Xuegang Hu},
  year={2024}
}
