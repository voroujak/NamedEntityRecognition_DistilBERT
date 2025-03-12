#Author: Mohamadreza Farid Ghasemnia

from utils import *

# a simple training pipeline, for preprocessing, and training a model
def simpleTrainPipeline(reduce_tags):
    model_path = pathFixer(reduce_tags)
    id2tag, tag2id= selectTagSet(reduce_tags)
    train_dataset, test_tokens, test_labels = preprocessDataset(train_set_en, test_set_en,reduce_tags, id2tag, 'train')
    
    model = trainModel(train_dataset, id2tag, tag2id, model_path)
    

    evalSingleSentence('The city name is Stockholm.', model, id2tag)
    evalSingleSentence('My name is MohamadrezaFaridGhasemnia.', model, id2tag)


# a simple evaluation pipeline, for preprocessing test set and computing metrics.
def simpleEvalPipeline(reduce_tags):
    model_path = pathFixer(reduce_tags=reduce_tags)
    id2tag, tag2id= selectTagSet(reduce_tags)
    train_dataset, test_tokens, test_labels = preprocessDataset(train_set_en, test_set_en,reduce_tags, id2tag, 'test')
    
    results= evalModel(model=None, model_path=model_path, test_tokens=test_tokens, test_labels=test_labels, id2tag=id2tag)
    
    with open('tags_reduced_'+str(reduce_tags)+'.txt', 'w') as f:
        f.write(str(results))


#TRAIN MODE
print('BEGIN TRAIN UNREDUCED MODE')
simpleTrainPipeline(reduce_tags=False)
print('BEGIN TRAIN REDUCED MODE')
simpleTrainPipeline(reduce_tags=True)



#eval mode
print('BEGIN EVAL UNREDUCED MODE')
simpleEvalPipeline(reduce_tags = False)
print('BEGIN EVAL REDUCED MODE')
simpleEvalPipeline(reduce_tags = True)

print('finished!')
