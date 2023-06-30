import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
import time
import random
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import torch
import random
from torch.optim import AdamW
from tqdm import tqdm
from src.mbti_prediction.dataprep.train_test_df import train_df, test_df
from utils import Dataset, convert_mbti_to_label, accuracy_and_auc, format_time
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

# Define Hyperparameters
NUM_LABELS = 2
MAX_LENGTH = 64
BATCH_SIZE = 32
NUM_EPOCHS = 4
LOG_INTERVAL = 200
LEARNING_RATE = 5e-5
ADAM_EPSILON = 1e-8

tokenizer = BertTokenizer.from_pretrained("kykim/bert-kor-base")
model = BertForSequenceClassification.from_pretrained("kykim/bert-kor-base", num_labels=NUM_LABELS, output_attentions=False, output_hidden_states=True)
# model.cuda()

# FIXME Change max_length to experiment the hyperparameters
train_tensor = tokenizer.batch_encode_plus(train_df['Answer'].to_list(), padding='longest', max_length=MAX_LENGTH, return_tensors='pt')
test_tensor = tokenizer(test_df['Answer'].to_list(), padding='longest', max_length=MAX_LENGTH, return_tensors='pt')

train_label = train_df['MBTI'].map(lambda mbti: convert_mbti_to_label(mbti, 'ie'))
test_label = test_df['MBTI'].map(lambda mbti: convert_mbti_to_label(mbti, 'ie'))

# Comment-me
train_dataset = Dataset(train_tensor, train_label)
val_dataset = Dataset(test_tensor, test_label)

train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(),
                  lr = LEARNING_RATE, # args.learning_rate - default is 5e-5
                  eps = ADAM_EPSILON # args.adam_epsilon  - default is 1e-8.
                )

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dl) * NUM_EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


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
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch for plotting
loss_values = []

for epoch_i in tqdm(range(0, NUM_EPOCHS)):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, NUM_EPOCHS))
    print('Training...')

    t0 = time.time()
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
    
    # Adaptive finetuning with freezing
    if epoch_i == 2:
        for name, param in model.named_parameters():
            if name not in ['bert.encoder.layer.11.attention.self.query.weight', 'bert.encoder.layer.11.attention.self.query.bias', 'bert.encoder.layer.11.attention.self.key.weight', 'bert.encoder.layer.11.attention.self.key.bias', 'bert.encoder.layer.11.attention.self.value.weight', 'bert.encoder.layer.11.attention.self.value.bias', 'bert.encoder.layer.11.attention.output.dense.weight', 'bert.encoder.layer.11.attention.output.dense.bias', 'bert.encoder.layer.11.attention.output.LayerNorm.weight', 'bert.encoder.layer.11.attention.output.LayerNorm.bias', 'bert.encoder.layer.11.intermediate.dense.weight', 'bert.encoder.layer.11.intermediate.dense.bias', 'bert.encoder.layer.11.output.dense.weight', 'bert.encoder.layer.11.output.dense.bias', 'bert.encoder.layer.11.output.LayerNorm.weight', 'bert.encoder.layer.11.output.LayerNorm.bias', 'bert.pooler.dense.weight', 'bert.pooler.dense.bias', 'classifier.weight', 'classifier.bias']:
                param.requires_grad = False


print("")
print("Training complete!")

# """### Change the model name here before saving"""
# model.save_pretrained("./models/IE_model_finetuned_with_augmented_dataset_extra_trained")
# tokenizer.save_pretrained("./models/IE_model_finetuned_with_augmented_dataset_extra_trained")

