from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd


# load datasets
df = pd.read_excel('../../../data/phase_1_paraphrased.xlsx')
df2 = pd.read_excel('../../../data/phase_2_paraphrased.xlsx')

df_copy = df.copy()

df.drop(['paraphrased_answer'], axis=1, inplace=True)
df_copy.drop(['Answer'], axis=1, inplace=True)
df_copy.rename(columns={'paraphrased_answer': 'Answer'}, inplace=True)

user_id = list(df_copy['User_ID'])
val = user_id[-1]
user_id_new = [(val + i) for i in user_id]

df_copy['User_ID'] = user_id_new

phase1_df = pd.concat([df, df_copy], ignore_index=True)

# Makes user ID unique to append to the training data as seperate entries
df2_copy = df2.copy()
df2.drop(['paraphrased_answer'], axis=1, inplace=True)
df2_copy.drop(['Answer'], axis=1, inplace=True)
df2_copy.rename(columns={'paraphrased_answer': 'Answer'}, inplace=True)

user_id = list(df2_copy['User_ID'])
val = user_id[-1]
user_id_new = [(val + i) for i in user_id]

df2_copy['User_ID'] = user_id_new

phase2_df = pd.concat([df2, df2_copy], ignore_index=True)

tokenizer = BertTokenizer.from_pretrained("kykim/bert-kor-base")
model = BertForSequenceClassification.from_pretrained("kykim/bert-kor-base", num_labels=2, output_attentions=False, output_hidden_states=True)
# model.cuda()

# load question dataset
question_df = pd.read_excel('../../../data/Question.xlsx')
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

# Combines short question and answer
for index, row in phase1_df.iterrows():
  phase1_df.at[index, 'Answer'] = questions_map.get(row['Q_number']-1) + ' ' + row['Short_Answer'] + ", " + row['Answer']

for index, row in phase2_df.iterrows():
  phase2_df.at[index, 'Answer'] = questions_map.get(row['Q_number']-1) + ' ' + row['Short_Answer'] + ", " + row['Answer']

# Splits train and test dataframe
train_df_list = []
test_df_list = []
for idx in phase1_df['User_ID'].unique():
    train_df_list.append(phase1_df[phase1_df['User_ID']==idx][0:40])
    test_df_list.append(phase1_df[phase1_df['User_ID']==idx][40:])

for idx in phase2_df['User_ID'].unique():
    train_df_list.append(phase2_df[phase2_df['User_ID']==idx])

train_df = pd.concat(train_df_list, ignore_index=True)
test_df = pd.concat(test_df_list, ignore_index=True)
