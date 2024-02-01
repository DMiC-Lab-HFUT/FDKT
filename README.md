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



## Dataset List

We collect the commonly used datasets and listed them here. The meaning of the fields in the table below is as follows:
- Exercise Text: contain textual information of exercise or not
- Concet Relation: contain relations among knowledge concepts or not (tree or prerequisite)
- Time: contain time for students to start answering questions or not
- Auto download: support download `middata` of the dataset  or not in EduStudio
- R2M Script: name of script to process the rawdata into middata  in EduStudio



| Dataset Name                                                 | Exercise Text | Concept Relation | Time | Auto Download | R2M Script Name          | Note                                                         |
| :----------------------------------------------------------- | :-----------: | :--------------: | :--: | :-----------: | :----------------------- | :----------------------------------------------------------- |
| [FrcSub](http://staff.ustc.edu.cn/~qiliuql/data/math2015.rar) |       ✖️       |        ✖️         |  ✖️   |       ✔️       | R2M_FrcSub               |                                                              |
| [Math1](http://staff.ustc.edu.cn/~qiliuql/data/math2015.rar) |       ✖️       |        ✖️         |  ✖️   |       ✔️       | R2M_Math1                |                                                              |
| [Math2](http://staff.ustc.edu.cn/~qiliuql/data/math2015.rar) |       ✖️       |        ✖️         |  ✖️   |       ✔️       | R2M_Math2                |                                                              |
| [AAAI_2023](https://docs.google.com/forms/d/e/1FAIpQLScWjxiXdSMAKBtlPJZm9MsudUG9CQS16lT0GVfajpVj-mWReA/viewform?pli=1) |       ✔️       |     ✔️(tree)      |  ✔️   |       ✔️       | R2M_AAAI_2023         | [AAAI2023 Global Knowledge Tracing Challenge](https://ai4ed.cc/competitions/aaai2023competition) |
| [ASSISTment_2009-2010](https://drive.google.com/file/d/0B2X0QD6q79ZJUFU1cjYtdGhVNjg/view?resourcekey=0-OyI8ZWxtGSAzhodUIcMf_g) |       ✖️       |        ✖️         |  ✔️   |       ✔️       | R2M_ASSIST_0910          |                                                              |
| [ASSISTment_2012-2013](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) |       ✖️       |        ✖️         |  ✔️   |       ✖️       | R2M_ASSIST_1213          |                                                              |
| [ASSISTment_2015-2016](https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data) |       ✖️       |        ✖️         |  ✔️   |       ✖️       | R2M_ASSIST_1516          |                                                              |
| [ASSISTment_2017](https://sites.google.com/view/assistmentsdatamining/dataset) |       ✖️       |        ✖️         |  ✔️   |       ✖️       | R2M_ASSIST_17            |                                                              |
| [Algebera_2005-2006](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp) |       ✖️       |        ✖️         |  ✔️   |       ✖️       | R2M_Algebera_0506        | [KDD Cup 2010](https://pslcdatashop.web.cmu.edu/KDDCup/rules_data_format.jsp) |
| [Algebera_2006-2007](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp) |       ✖️       |        ✖️         |  ✔️   |       ✖️       | R2M_Algebera_0607        | [KDD Cup 2010](https://pslcdatashop.web.cmu.edu/KDDCup/rules_data_format.jsp) |
| [Bridge2Algebra_2006-2007](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp) |       ✖️       |        ✖️         |  ✔️   |       ✖️       | R2M_Bridge2Algebra_0607  | [KDD Cup 2010](https://pslcdatashop.web.cmu.edu/KDDCup/rules_data_format.jsp) |
| [Junyi_AreaTopicAsCpt](https://pslcdatashop.web.cmu.edu/Project?id=244) |       ✖️       |     ✔️(tree)      |  ✔️   |       ✖️       | R2M_Junyi_AreaTopicAsCpt | Area&Topic field as concept                                  |
| [Junyi_ExerAsCpt](https://pslcdatashop.web.cmu.edu/Project?id=244) |       ✖️       | ✔️(prerequisite)  |  ✔️   |       ✖️       | R2M_Junyi_ExerAsCpt      | Exercice as concept                                          |
| EdNet_KT1                                                    |       ✖️       |        ✖️         |  ✔️   |       ✖️       | R2M_EdNet_KT1            | [download1](http://bit.ly/ednet-content), [download2](http://bit.ly/ednet-content) |
| [Eedi_2020_Task1&2](https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip) |       ✖️       |     ✔️(tree)      |  ✔️   |       ✖️       | R2M_Eedi_20_T12          | [NeurIPS 2020 Education Challenge: Task1&2](https://eedi.com/projects/neurips-education-challenge) |
| [Eedi_2020_Task3&4](https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip) |       ✔️(images)       |     ✔️(tree)      |  ✔️   |       ✖️       | R2M_Eedi_20_T34          | [NeurIPS 2020 Education Challenge: Task3&4](https://eedi.com/projects/neurips-education-challenge) |

##   Cite
```
@article{liu2024fuzzykt,
  title={FDKT: Towards an interpretable deep knowledge tracing via fuzzy reasoning},
  author={Fei Liu, Chenyang Bu, Haotian Zhang, Le Wu, Kui Yu, and Xuegang Hu},
  year={2023}
}
```