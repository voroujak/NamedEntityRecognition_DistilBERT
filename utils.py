

from datasets import load_dataset
from transformers import DistilBertTokenizerFast, TFDistilBertForTokenClassification
from transformers import TFTrainer, TFTrainingArguments
#from transformers import RobertaTokenizerFast, TFRobertaForTokenClassification
#from transformers import XLNetTokenizerFast, TFXLNetForTokenClassification
import tensorflow as tf

tokenizerClass = DistilBertTokenizerFast
classificationClass = TFDistilBertForTokenClassification
import numpy as np
import evaluate
import copy


model_name=  'distilbert-base-cased'#'roberta-base' #'bert-base-cased' #'xlnet-base-cased' # 'distilbert-base-cased'

#loading the data
dataset = load_dataset('Babelscape/multinerd')
tokenizer = tokenizerClass.from_pretrained(model_name)

train_set = dataset['train']
test_set= dataset['test']

#filtering English data
train_set_en = train_set.filter(lambda sample: sample['lang'] == 'en')
test_set_en = test_set.filter(lambda sample: sample['lang'] == 'en')


#number of traning epochs.
num_train_epochs=7



debugMode = False #set True for reducing size of dataset, in debugging
if debugMode:
    num_train_epochs=5
    datasetSizeLimiter = 5000 
    train_set_en = train_set_en[:datasetSizeLimiter]
    test_set_en = test_set_en[:datasetSizeLimiter]




###Degining lists of tags, complete and reduced

unreduced_tag2id={
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-ANIM": 7,
    "I-ANIM": 8,
    "B-BIO": 9,
    "I-BIO": 10,
    "B-CEL": 11,
    "I-CEL": 12,
    "B-DIS": 13,
    "I-DIS": 14,
    "B-EVE": 15,
    "I-EVE": 16,
    "B-FOOD": 17,
    "I-FOOD": 18,
    "B-INST": 19,
    "I-INST": 20,
    "B-MEDIA": 21,
    "I-MEDIA": 22,
    "B-MYTH": 23,
    "I-MYTH": 24,
    "B-PLANT": 25,
    "I-PLANT": 26,
    "B-TIME": 27,
    "I-TIME": 28,
    "B-VEHI": 29,
    "I-VEHI": 30,
  }



reduced_tag2id={
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-ANIM": 7,
    "I-ANIM": 8,
    "B-DIS": 9,
    "I-DIS": 10,
    #
    #"B-BIO": 9,
    #"I-BIO": 10,
    #"B-CEL": 11,
    #"I-CEL": 12,
    #"B-DIS": 13,
    #"I-DIS": 14,
  }

#id of tags that are not in the five tags
removeable_ids= [9,10,11,12,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

#select function from list of tags, and providee id2tad and tag2id
def selectTagSet(reduce_tags= False):
    if not reduce_tags:
        tag2id = unreduced_tag2id
        #id2tag = {id: tag for tag, id in tag2id.items()}
    if reduce_tags:
        tag2id = reduced_tag2id
    id2tag = {id: tag for tag, id in tag2id.items()}

    return id2tag, tag2id

#preprocess dataset with following steps: reduce tags if needed, fix label of DIS tags in order to keep the continuousity of ids, tokenize inputs, fix labels to keep the correspondance with words, and put things into tensorflow dataset or numpy array.
def preprocessDataset(train_set_en, test_set_en, reduce_tags, id2tag, mode= 'train'):
    train_dataset, test_tokens, test_labels=None, None, None
    train_tokens = train_set_en['tokens']
    train_ner_tags = train_set_en['ner_tags']

    test_tokens = test_set_en['tokens']
    test_ner_tags = test_set_en['ner_tags']

    if reduce_tags:
        #filtered_ids = list(id2tag.keys())
        
        train_ner_tags= [[0 if tag in removeable_ids else tag for tag in sentence] for sentence in train_ner_tags]
        test_ner_tags= [[0 if tag in removeable_ids else tag for tag in sentence] for sentence in test_ner_tags]

        train_ner_tags= [[9 if tag==13  else tag for tag in sentence] for sentence in train_ner_tags]
        train_ner_tags= [[10 if tag==14  else tag for tag in sentence] for sentence in train_ner_tags]

        test_ner_tags= [[9 if tag==13  else tag for tag in sentence] for sentence in test_ner_tags]
        test_ner_tags= [[10 if tag==14  else tag for tag in sentence] for sentence in test_ner_tags]

        #test_ner_tags= [[tag if tag in filtered_ids else 0 for tag in sentence] for sentence in test_ner_tags]




    if mode=='train':
        train_encodings = tokenizer(train_tokens, 
                                    is_split_into_words=True, return_offsets_mapping=True,
                                    padding='max_length', truncation=True, max_length=336)
        train_labels = fix_labels(train_ner_tags, train_encodings)
        train_encodings.pop("offset_mapping") # we don't want to pass this to the model
        train_dataset = tf.data.Dataset.from_tensor_slices(( dict(train_encodings), train_labels))

    if mode =='test':
        test_encodings = tokenizer(test_tokens, 
                                is_split_into_words=True, return_offsets_mapping=True, 
                                padding='max_length', truncation=True, max_length=336)
        test_labels = fix_labels(test_ner_tags, test_encodings)
        test_encodings.pop("offset_mapping")


    return train_dataset, test_tokens, test_labels

#fix the correspondance of labels and tokenized (encoded) inputs
def fix_labels(tags, encodings):
    fixed_labels = np.ones((len(encodings.input_ids), len(encodings.input_ids[0])), dtype=np.int16)*-100
    for i, (sentence_tags, sentence_offsets) in enumerate(zip(tags, encodings.offset_mapping)):
        tag = -100
        tmpTags = copy.deepcopy(sentence_tags)
        for j in range(len(sentence_offsets)):
            offset = sentence_offsets[j]
            
            if (int(offset[0]) ==0) & (int(offset[1])==0):
                continue
            if (int(offset[0]) != 0) & (int(offset[1])!= 0):
                continue # tag = last_tag if to prefer the tag come along all broken parts of entities.
            if (int(offset[0])== 0) & (int(offset[1]) != 0):
                tag = tmpTags.pop(0)


            fixed_labels[i,j] = tag
    return fixed_labels


#computing the metrics while ignoring the paddings
def compute_metrics(predictions, labels, id2tag):
    #predictions = np.argmax(predictions, axis=2)
    true_predictions = [[id2tag[p] for (p, l) in zip(prediction, label) if (l != -100)]
        for prediction, label in zip(predictions, labels)]
    true_labels = [[id2tag[l] for (p, l) in zip(prediction, label) if (l != -100)]
        for prediction, label in zip(predictions, labels)]

    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    return results



#train size: 262560
#with batch_size 64,2 gpu, 1 epoch is around 2k steps

#train parameters
def trainModel(train_dataset, id2tag, tag2id, model_save_path):
    training_args = TFTrainingArguments(
        output_dir=model_save_path + '/results',          # output directory
        num_train_epochs=num_train_epochs,              # total number of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=250,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=model_save_path +'/logs',            # directory for storing logs
        logging_steps=10,
        eval_steps=1000,
        #max_grad_norm=0.1,
        #label_smoothing_factor=0.5,
        #learning_rate=5e-8
        evaluation_strategy='no'
        #warmup 250, weight decay 0.01
    )

    with training_args.strategy.scope():
        model = classificationClass.from_pretrained(
            model_name, len(tag2id), id2label=id2tag, label2id=tag2id)

    trainer = TFTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        #eval_dataset=test_dataset,             # evaluation dataset
        #compute_metrics=compute_metrics,
    )

    trainer.train()


    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(model_save_path)
    return trainer.model


#evaluaion by loading the model, and pass test set for prediction
def evalModel(model, model_path, test_tokens, test_labels, id2tag):
    #model= trainer.model
    if model ==None:
        model = classificationClass.from_pretrained(model_path)

    test_encodings = tokenizer(test_tokens, is_split_into_words=True, 
                               padding='max_length', max_length=336, truncation=True, return_tensors='tf')
    #test_encodings.pop('offset_mapping')
    test_predictions = model.predict(dict(test_encodings))
    predClasses = test_predictions.logits.argmax(-1)

    results = compute_metrics(predClasses, test_labels, id2tag)
    print(results)
    return results

#evaluating a single sentence, takes a non-tokenized string sentence.
def evalSingleSentence(sentence, model, id2tag):
    tokenized_sentence = tokenizer(sentence, padding=True, truncation=True,return_tensors="tf")
    output= model.predict(dict(tokenized_sentence))
    predictions = output.logits.argmax(-1)

    predictions = [[id2tag[p] for p in sentence] for sentence in predictions]
    print('THE SENTENCE IS: ',  tokenizer.decode(tokenized_sentence['input_ids'][0]))
    print('TOKENIED INTO: ', tokenized_sentence.tokens())
    print('NER PREDICTIONS ARE: ', predictions)
    return predictions

def pathFixer(reduce_tags):    
    if reduce_tags:
        reducePath = '/reduced'
    else:
        reducePath = '/unreduced'
    model_path = '.'+  reducePath + '_model'
    return model_path


