import pandas as pd
import numpy as np
from scipy import stats
#load  in data and just consider columsn of importance for analysis
df_task=pd.read_csv('task.csv')
df_task_r=df_task[['Participant Public ID','display','forced_choice','Response','image2','test_image1', 'test_image2','test_image1_value',
                   'test_image2_value','image_query2', 'image_query1']]
df_task_r=df_task_r.replace('response_text_entry','query_internal_probability')

# Define best-action dictionary for present design
# best action per condition (6 conditions)
best_answer_key={'rare_threat_1': [[0, 'Pinecone 1.jpg'], [1, 'Pumpkin 1.jpg']],
                 'rare_threat_2': [[0, 'Keyboard 3.jpg'], [1, 'Office supplies 2.jpg']],
                 'common_threat_1': [[0, 'Fire hydrant 1.jpg'], [1, 'Fence 2.jpg']],
                 'common_threat_2': [[0, 'Bricks 1.jpg'], [1, 'Barrels 1.jpg']],
                 'neutral_1': [[0, 'Snow 3.jpg'], [1, 'Skyscraper 1.jpg']],
                 'neutral_2': [[0, 'Clean 1.jpg'], [1, 'Cotton swabs 3.jpg']]}

#best answers per condition: lists
rt1=[]
rt2=[]
ct1=[]
ct2=[]
n1=[]
n2=[]

tally=0
invalid_scores={'NaN'}
counter=0
conditions=[]
start_new_test_set=0
start_new_subject=0
best_action_tally=0
condition_counter=0
sub_counter=0
current_subject=1

#subject specific data
rt_sub=[]
ct_sub=[]
neut_sub=[]
current_choice_data=[]
choice_data_3d=np.zeros((13,3,40)) #to be populated below for bernoulli formulation
choice_data_binomial_n=np.zeros((13,3))
choice_data_binomial_rt=np.zeros((13,3))
choice_data_binomial_ct=np.zeros((13,3))


for row,data in df_task_r.iterrows():

    if str(df_task_r['display'][row]).startswith('test'):

        if counter==0:
            conditions.append(df_task_r['display'][row][5:])
            condition_info=best_answer_key[conditions[counter]]
            new_condition=0
            counter+=1


        elif df_task_r['display'][row][5:]!=conditions[counter-1]:
                    conditions.append(df_task_r['display'][row][5:])
                    condition_info=best_answer_key[conditions[counter]]
                    if conditions[counter-1]=='rare_threat_1':
                        rt1.append(best_action_tally)
                        rt_sub.append(current_choice_data)

                    elif conditions[counter-1]=='rare_threat_2':
                        rt2.append(best_action_tally)
                        rt_sub.append(current_choice_data)

                    elif conditions[counter-1]=='common_threat_1':
                        ct1.append(best_action_tally)
                        ct_sub.append(current_choice_data)

                    elif conditions[counter-1]=='common_threat_2':
                        ct2.append(best_action_tally)
                        ct_sub.append(current_choice_data)

                    elif conditions[counter-1]=='neutral_1':
                        n1.append(best_action_tally)
                        neut_sub.append(current_choice_data)

                    elif conditions[counter-1]=='neutral_2':
                        n2.append(best_action_tally)
                        neut_sub.append(current_choice_data)
                    counter+=1
                    best_action_tally=0
                    current_choice_data=[]
                    condition_counter+=1
                    #after 6 blocks, new subject
                    if condition_counter>5:
                        current_subject+=1
                        neut_sub=neut_sub[0]+neut_sub[1]
                        if len(neut_sub)<40:
                            neut_sub=[int(x) for x in neut_sub+np.zeros(40-len(neut_sub)).tolist()]
                        sum_neutral=np.sum(neut_sub)


                        rt_sub=rt_sub[0]+rt_sub[1]
                        if len(rt_sub)<40:
                            rt_sub=[int(x) for x in rt_sub+np.zeros(40-len(rt_sub)).tolist()]
                        sum_positive=np.sum(rt_sub)

                        ct_sub=ct_sub[0]+ct_sub[1]
                        if len(ct_sub)<40:
                            ct_sub=[int(x) for x in ct_sub+np.zeros(40-len(ct_sub)).tolist()]
                        sum_negative=np.sum(ct_sub)

                        choice_data_3d[sub_counter,0]=neut_sub
                        choice_data_3d[sub_counter,1]=ct_sub
                        choice_data_3d[sub_counter,2]=rt_sub

                        choice_data_binomial_n[sub_counter,0]=sum_neutral
                        choice_data_binomial_n[sub_counter,1]=1
                        choice_data_binomial_n[sub_counter,2]=sub_counter+1
                        choice_data_binomial_rt[sub_counter,0]=sum_negative
                        choice_data_binomial_rt[sub_counter,1]=2
                        choice_data_binomial_rt[sub_counter,2]=sub_counter+1
                        choice_data_binomial_ct[sub_counter,0]=sum_positive
                        choice_data_binomial_ct[sub_counter,1]=3
                        choice_data_binomial_ct[sub_counter,2]=sub_counter+1

                        sub_counter+=1
                        condition_counter=0
                        rt_sub=[]
                        ct_sub=[]
                        neut_sub=[]

        else:
            new_condition=0

        #Get values and convert from strings to floating point
        value1=df_task_r['test_image1_value'][row]
        if "p" in value1:
            value1=float(value1[0:2])*0.01
        else:
            value1=float(value1[1])
        value2=df_task_r['test_image2_value'][row]
        if "p" in value2:
            value2=float(value2[0:2])*0.01
        else:
            value2=float(value2[1])

        if value1>value2:
            best_option=df_task_r['test_image1'][row]
        else:
            best_option=df_task_r['test_image2'][row]


        #get response and convert to integer
        try:
            current_response=int(df_task_r['Response'][row])
        except:
            current_response='missing'




        # determine if participant made best choice
        for info_total in condition_info:
            for info in info_total:
                if best_option == str(info):
                    best_action=info_total[0]

        #for last subject only that doesn't meet the condition above for indexing
        if row==17255:
            ct2.append(best_action_tally)
            ct_sub.append(current_choice_data)
            neut_sub=neut_sub[0]+neut_sub[1]
            if len(neut_sub)<40:
                neut_sub=[int(x) for x in neut_sub+np.zeros(40-len(neut_sub)).tolist()]


            rt_sub=rt_sub[0]+rt_sub[1]
            if len(rt_sub)<40:
                rt_sub=[int(x) for x in rt_sub+np.zeros(40-len(rt_sub)).tolist()]

            ct_sub=ct_sub[0]+ct_sub[1]
            if len(ct_sub)<40:
                ct_sub=[int(x) for x in ct_sub+np.zeros(40-len(ct_sub)).tolist()]

            sum_neutral=np.sum(neut_sub)
            sum_positive=np.sum(rt_sub)
            sum_negative=np.sum(ct_sub)
            choice_data_3d[sub_counter,0]=neut_sub
            choice_data_3d[sub_counter,1]=ct_sub
            choice_data_3d[sub_counter,2]=rt_sub
            choice_data_binomial_n[sub_counter,0]=sum_neutral
            choice_data_binomial_n[sub_counter,1]=1
            choice_data_binomial_n[sub_counter,2]=sub_counter+1
            choice_data_binomial_rt[sub_counter,0]=sum_negative
            choice_data_binomial_rt[sub_counter,1]=2
            choice_data_binomial_rt[sub_counter,2]=sub_counter+1
            choice_data_binomial_ct[sub_counter,0]=sum_positive
            choice_data_binomial_ct[sub_counter,1]=3
            choice_data_binomial_ct[sub_counter,2]=sub_counter+1
            sub_counter+=1
            condition_counter=0
            rt_sub=[]
            ct_sub=[]
            neut_sub=[]

        else:
            if current_response==best_action:
                best_action_tally+=1
                current_choice_data.append(1.0)
            elif current_response=='missing':
                x='missing'
            else:
                current_choice_data.append(0.0)


#convert to numpy arrays
rt1=np.array(rt1)
rt2=np.array(rt2)
ct1=np.array(ct1)
ct2=np.array(ct2)
n1=np.array(n1)
n2=np.array(n2)
choice_data_3d = choice_data_3d.astype(int)
choice_data_binomial=np.concatenate((choice_data_binomial_n,choice_data_binomial_rt,choice_data_binomial_ct))

choice_data_binomial_std=np.concatenate((((choice_data_binomial[:,0]-np.mean(choice_data_binomial[:,0]))/np.std(choice_data_binomial[:,0])).reshape((39,1)),choice_data_binomial[:,1].reshape((39,1)),choice_data_binomial[:,2].reshape((39,1))),axis=1)
# print('Binomial-ready data standardized: \n{}'.format(choice_data_binomial_std))
# print(choice_data_binomial_std.shape)
choice_data_binomial=choice_data_binomial.astype(int)
# print('Binomial raw data:\n {}'.format(choice_data_binomial))
# print('mean: {}, sd: {}'.format(np.mean(choice_data_binomial[:,0]),np.std(choice_data_binomial[:,0])))
np.save('choice_data',choice_data_binomial)
