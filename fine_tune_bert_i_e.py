import torch
from torch.utils.data import DataLoader, Dataset

from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import time
import pandas as pd
import random
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import torch
import random
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


# load datasets
df = pd.read_excel('./data/phase_1_paraphrased.xlsx')
df2 = pd.read_excel('./data/phase_2_paraphrased.xlsx')

df_copy = df.copy()

df.drop(['paraphrased_answer'], axis=1, inplace=True)
df_copy.drop(['Answer'], axis=1, inplace=True)
df_copy.rename(columns={'paraphrased_answer': 'Answer'}, inplace=True)

user_id = list(df_copy['User_ID'])
val = user_id[-1]
user_id_new = [(val + i) for i in user_id]

df_copy['User_ID'] = user_id_new


phase1_df = pd.concat([df, df_copy], ignore_index=True)

# I need to make the user_id different to make sure that I append it to the training data as seperate entries
df2_copy = df2.copy()
df2.drop(['paraphrased_answer'], axis=1, inplace=True)
df2_copy.drop(['Answer'], axis=1, inplace=True)
df2_copy.rename(columns={'paraphrased_answer': 'Answer'}, inplace=True)

user_id = list(df2_copy['User_ID'])
val = user_id[-1]
user_id_new = [(val + i) for i in user_id]

df2_copy['User_ID'] = user_id_new

phase2_df = pd.concat([df2, df2_copy], ignore_index=True)

"""# Search for "# Comment-me" and comment it if you are loading from the fine-tuned model"""

# Comment-me
tokenizer = BertTokenizer.from_pretrained("kykim/bert-kor-base")
model = BertForSequenceClassification.from_pretrained("kykim/bert-kor-base", num_labels=2, output_attentions=False, output_hidden_states=True)
# model.cuda()

# Load question dataset
question_df = pd.read_excel('./data/Question.xlsx')
questions = list(question_df['Question'])
questions_temp = [i.split("?") for i in questions]

short_q = []
descriptive_q = []
for idx, q in enumerate(questions_temp):
  short_q.append(q[0] + '?')
  descriptive_q.append(q[1])

questions_map = {}
for idx, val in enumerate(short_q):
  questions_map[idx] = val

# This is for combining short question and answer
for index, row in phase1_df.iterrows():
  phase1_df.at[index, 'Answer'] = questions_map.get(row['Q_number']-1) + ' ' + row['Short_Answer'] + ", " + row['Answer']

for index, row in phase2_df.iterrows():
  phase2_df.at[index, 'Answer'] = questions_map.get(row['Q_number']-1) + ' ' + row['Short_Answer'] + ", " + row['Answer']

# split train and test dataframe
train_df_list = []
test_df_list = []
for idx in phase1_df['User_ID'].unique():
    train_df_list.append(phase1_df[phase1_df['User_ID']==idx][0:40])
    test_df_list.append(phase1_df[phase1_df['User_ID']==idx][40:])

for idx in phase2_df['User_ID'].unique():
    train_df_list.append(phase2_df[phase2_df['User_ID']==idx])

train_df = pd.concat(train_df_list, ignore_index=True)
test_df = pd.concat(test_df_list, ignore_index=True)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

def convert_mbti_to_label(mbti: str):
    """
    :param mbti: string. length=4
    :return:
    """
    stand = 'ISTJ'  # [0, 0, 0, 0]
    result = []
    for i in range(4):
        if stand[i] == mbti[i]:
            result.append(0)
        else:
            result.append(1)

    # FIXME #CHANGEHERE --> 0 or 1 or 2 or 3
    # Change the result here to train 4 different models
    return result[0]

# Comment-me
# FIXME Change max_length to experiment the hyperparameters
train_tensor = tokenizer.batch_encode_plus(train_df['Answer'].to_list(), padding='longest', max_length=64, return_tensors='pt')
test_tensor = tokenizer(test_df['Answer'].to_list(), padding='longest', max_length=64, return_tensors='pt')

train_label = train_df['MBTI'].map(convert_mbti_to_label)
test_label = test_df['MBTI'].map(convert_mbti_to_label)

# Comment-me
train_dataset = Dataset(train_tensor, train_label)
val_dataset = Dataset(test_tensor, test_label)

# Comment-me
num_labels=2
max_length = 64
batch_size = 32
num_epochs = 4
log_interval = 200
learning_rate =  5e-5


def accuracy_and_auc(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)

    auc_score = roc_auc_score(labels_flat, preds[:, 1])  # Assuming binary classification
    return accuracy, auc_score


train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

optimizer = AdamW(model.parameters(),
                  lr = 4e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

# Number of training epochs (authors recommend between 2 and 4)
epochs = 3

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dl) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch for plotting
loss_values = []

for epoch_i in tqdm(range(0, epochs)):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss, accuracy and epoch for this epoch.
    total_loss = 0
    total_accuracy = 0
    total_auc = 0

    model.train()

    total_steps = len(train_dl)
    for step, batch in tqdm(enumerate(train_dl), total=total_steps):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dl), elapsed))

        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        model.zero_grad()

        # Output format: (loss, logits)
        outputs = model(b_input_ids,
                    token_type_ids=batch['token_type_ids'].to(device),
                    attention_mask=b_input_mask,
                    labels=b_labels)

        loss = outputs[0]
        total_loss += loss.item()

        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_train_accuracy, tmp_train_auc = accuracy_and_auc(logits, label_ids)
        total_accuracy += tmp_train_accuracy
        total_auc += tmp_train_auc

        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dl)
    avg_train_accuracy = total_accuracy / len(train_dl)
    avg_train_auc = total_auc / len(train_dl)
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Average training accuracy: {0:.2f}".format(avg_train_accuracy))
    print("  Average training auc: {0:.2f}".format(avg_train_auc))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================
    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    eval_loss, eval_accuracy, eval_auc = 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    total_test_steps = len(test_dl)
    for batch in tqdm(test_dl, total=total_test_steps):

        batch = tuple(val.to(device) for (key, val) in batch.items())
        b_input_ids, b_token_type_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            # No need to token_type_ids since it is a good practice not to.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy, tmp_auc = accuracy_and_auc(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        eval_auc += tmp_auc

        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  ROC_AUC: {0:.2f}".format(eval_auc/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")

for name, param in model.named_parameters():
    if name not in ['bert.encoder.layer.11.attention.self.query.weight', 'bert.encoder.layer.11.attention.self.query.bias', 'bert.encoder.layer.11.attention.self.key.weight', 'bert.encoder.layer.11.attention.self.key.bias', 'bert.encoder.layer.11.attention.self.value.weight', 'bert.encoder.layer.11.attention.self.value.bias', 'bert.encoder.layer.11.attention.output.dense.weight', 'bert.encoder.layer.11.attention.output.dense.bias', 'bert.encoder.layer.11.attention.output.LayerNorm.weight', 'bert.encoder.layer.11.attention.output.LayerNorm.bias', 'bert.encoder.layer.11.intermediate.dense.weight', 'bert.encoder.layer.11.intermediate.dense.bias', 'bert.encoder.layer.11.output.dense.weight', 'bert.encoder.layer.11.output.dense.bias', 'bert.encoder.layer.11.output.LayerNorm.weight', 'bert.encoder.layer.11.output.LayerNorm.bias', 'bert.pooler.dense.weight', 'bert.pooler.dense.bias', 'classifier.weight', 'classifier.bias']:
        param.requires_grad = False


train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

optimizer = AdamW(model.parameters(),
                  lr = 4e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

# Number of training epochs (authors recommend between 2 and 4)
epochs = 1

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dl) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch for plotting
loss_values = []

for epoch_i in tqdm(range(0, epochs)):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss, accuracy and epoch for this epoch.
    total_loss = 0
    total_accuracy = 0
    total_auc = 0

    model.train()

    total_steps = len(train_dl)
    for step, batch in tqdm(enumerate(train_dl), total=total_steps):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dl), elapsed))

        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        model.zero_grad()

        # Output format: (loss, logits)
        outputs = model(b_input_ids,
                    token_type_ids=batch['token_type_ids'].to(device),
                    attention_mask=b_input_mask,
                    labels=b_labels)

        loss = outputs[0]
        total_loss += loss.item()

        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_train_accuracy, tmp_train_auc = accuracy_and_auc(logits, label_ids)
        total_accuracy += tmp_train_accuracy
        total_auc += tmp_train_auc

        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dl)
    avg_train_accuracy = total_accuracy / len(train_dl)
    avg_train_auc = total_auc / len(train_dl)
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Average training accuracy: {0:.2f}".format(avg_train_accuracy))
    print("  Average training auc: {0:.2f}".format(avg_train_auc))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================
    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    eval_loss, eval_accuracy, eval_auc = 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    total_test_steps = len(test_dl)
    for batch in tqdm(test_dl, total=total_test_steps):

        batch = tuple(val.to(device) for (key, val) in batch.items())
        b_input_ids, b_token_type_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            # No need to token_type_ids since it is a good practice not to.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy, tmp_auc = accuracy_and_auc(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        eval_auc += tmp_auc

        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  ROC_AUC: {0:.2f}".format(eval_auc/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")

"""### Change the model name here before saving"""

# model.save_pretrained("./models/IE_model_finetuned_with_augmented_dataset_extra_trained")

# tokenizer.save_pretrained("./models/IE_model_finetuned_with_augmented_dataset_extra_trained")

