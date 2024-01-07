# FDKT: Towards an interpretable deep knowledge tracing via fuzzy reasoning

This is our implementation for the paper titled **FDKT: Towards an interpretable deep knowledge tracing via fuzzy reasoning**

**Author**: Fei Liu, Chenyang Bu, Haotian Zhang, Le Wu, Kui Yu, and Xuegang Hu

    
    
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