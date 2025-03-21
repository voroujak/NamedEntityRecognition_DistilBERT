
# Named-Entity Recognition using DistilBERT

## M. Farid.





# Description 

## 1 Problem definition
Named Entity Recognition (NER) is one of the principal problems in NLP, which is the problem of classifying each token of a sentence into a set of predefined classes. 

This is the implementation of MultiNERD Named Entity Recognition (NER) dataset in for English:
1. Implmentation is based on a BERT model (DistilBERT), using HuggingFace.
2. Non-English examples of the dataset are filtered out, and two case are compared with each other, Case A, and Case B.
3. In case A (systemA), the model is trained on whole English subset of dataset. 
4. In Case B (systemB), the model is trained to predict only five entity types: PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS),
ANIMAL(ANIM) (in addition to Null "O" tag).



## 2 Main Outlines

The systemB exhibits slightly better results in learning (almost all) five tags, leading to a better overall F1 score. However, it's important to note that the dataset has imbalanced labels, and systemB's performance may be influenced by pruning less-represented labels. Despite this, both models, including systemA, demonstrate good performance in Named Entity Recognition (NER), which is often considered a relatively simpler task for learning models, e.g., compared to translation. 

SystemA’s results indicate that the model effectively handles imbalanced data, and it is expected that the performance difference with models like LSTM to be more pronounced. Notably, some tags in SystemA show lower accuracy, which can be rooted in at least two things:

1. They are less-represented labels in the dataset; hence, the model learned them less, aligned with expectations of the nature of statistical models.
2. The "cased" model of DistilBERT is, which makes a difference between capitalized and lowercase words, ending in the tendency of the model in favor of recognizing capitalized names (e.g., often Person represented with capitalization) over uncapitalized ones. 

This observation is supported by examples with non-capitalized names, as well as lower F1 scores for labels that are commonly not capitalized.

Additionally, addressing Out-Of-Vocabulary words, typically handled by feeding the split OOV tokens to transformers, offers room for improvement with alternative approaches. Further investigations can also be made with different models, experimenting with different hyperparameters, penalizing the loss function towards solving the imbalanced nature of data, and so on.

## 3 Summary of Solution

The solution is implemented using TensorFlow GPU, two 3090 GPUs using Hugging Face DistilBERT model [(TensorFlow version)](https://huggingface.co/docs/transformers/model_doc/distilbert), which is a pre-trained knowledge-distilled BERT model. 

The solution includes a preprocessing step, training, and evaluation. The preprocessing steps include setting input and output in a proper Numpy-compatible shape and TensorFlow tensors while preserving the correspondence of tokenized words and labels. The training is done for seven epochs, with a 64 per-GPU batch size, which takes around two hours (a complete training of each model). 

The evaluation is done using the `seqeval` library and includes comparing the precision, recall, and F1 of each label and all labels together. Below are the results of the NER recognition over the test sets for both systems:

### Performance Comparison

| Tag (Number) | SystemA Precision | SystemA Recall | SystemA F1 | SystemB Precision | SystemB Recall | SystemB F1 |
|-------------|------------------|--------------|---------|------------------|--------------|---------|
| PER (10530) | 0.989 | 0.992 | 0.991 | 0.99 | 0.993 | 0.992 |
| ORG (6618) | 0.968 | 0.976 | 0.972 | 0.972 | 0.976 | 0.974 |
| LOC (24048) | 0.992 | 0.993 | 0.993 | 0.992 | 0.993 | 0.993 |
| ANIM (3208) | 0.714 | 0.774 | 0.743 | 0.717 | 0.762 | 0.739 |
| DIS (1518) | 0.733 | 0.777 | 0.754 | 0.743 | 0.796 | 0.768 |
| BIO (16) | 0.5 | 0.875 | 0.636 | - | - | - |
| CEL (82) | 0.75 | 0.805 | 0.776 | - | - | - |
| EVE (704) | 0.931 | 0.96 | 0.945 | - | - | - |
| FOOD (1132) | 0.618 | 0.668 | 0.642 | - | - | - |
| INST (24) | 0.6 | 0.75 | 0.667 | - | - | - |
| MEDIA (916) | 0.952 | 0.963 | 0.958 | - | - | - |
| MYTH (64) | 0.818 | 0.844 | 0.831 | - | - | - |
| PLANT (1788) | 0.624 | 0.697 | 0.659 | - | - | - |
| TIME (578) | 0.813 | 0.844 | 0.829 | - | - | - |
| VEHI (64) | 0.743 | 0.812 | 0.776 | - | - | - |
| **Overall** | **0.934** | **0.95** | **0.942** | **0.96** | **0.968** | **0.964** |

**Table 1:** Precision, Recall, F1, and Number for SystemA and SystemB across different tags.




# Structure of project
You can run the file by running in terminal: python ner.py


## Installing requirements
You can install the requirements by pip install requirements.txt. 

### Note 1 

The detailed list of all installed packages are given in requiremenetsDetailed.txt. Note that there are other packages that are not used in this project, and installing with requirements.txt is preferred.

## Note 2

The requirements of GPU has be handled individually. This code is run on two 3090 GPU, with core-i9 cpu, with cuda version 11.4, and each epoch approximately takes 20-30 minutes.



## Structure of files

### Python files
The main python file is the ner.py, which also calls the utils module. 

### Folders 
The two folders, "reduced_model" and "unreduced_model" corresponds to system-A and system-B. In each folder, the tensorboard log, model, and training plots of tensorboard can be found. 

### Files
The terminal output of the code is recorded in outputlogTerminal.txt, and the two files of results_tags_reduced.txt, and results_tags_unreduced.txt represent the resultant evaluation of the two systems, respectively. 
