import pandas as pd
import numpy as np
import shutil



root="/mount/neuron/Lamborghini/dir/pythonProject/MICCAI/artifact-generalization-skin/"
root_dir =  '/mount/neuron/Lamborghini/dir/pythonProject/CVPR/data/ISIC_2019_Training_Input/'
root_csv = root+"group_DRO/trap_sets_paper2021/train_bias_0_1.csv"

confounder_names = ["dark_corner", "hair", "gel_border", "gel_bubble", "ink", "ruler", "patches"]
# Read in attributes
attrs_df = pd.read_csv(root+'isic_inferred_wocarcinoma.csv')
data = pd.read_csv(root_csv)
attrs_df = attrs_df[attrs_df['image'].isin(data['image'])]
print('----',data)
print('+++++',attrs_df)
#
# # Split out filenames and attribute names
#
filename_array = attrs_df['image'].values
filename_array = np.array([name + '.jpg' for name in filename_array])
print(filename_array)
# attrs_df = attrs_df.drop(labels='image', axis='columns')
# attr_names = attrs_df.columns.copy()
# print('wwwwwwwwwwwwwww', attr_names)
#
# # Then cast attributes to numpy array and set them to 0 and 1
# # (originally, they're -1 and 1)
# # attrs_df = attrs_df.values
def add(a, b, c,d,e,f,g):
    return a + b + c + d + e + f + g
env_df=attrs_df.drop(labels='image', axis='columns')
env_df=env_df.drop(labels='Unnamed: 0',axis='columns')
env_df=env_df.drop(labels='vasc',axis='columns')
env_df=env_df.drop(labels='label',axis='columns')
env_df=env_df.drop(labels='label_string',axis='columns')
env_df=env_df.drop(labels='gel_border',axis='columns')
env_df=env_df.drop(labels='ink',axis='columns')
env_df=env_df.drop(labels='patches',axis='columns')
attrs_df['max_prob']=env_df.max(axis=1)
# for cfd in confounder_names:
    #
    # attrs_df[cfd][attrs_df[cfd]>0.8]  = 1
    # attrs_df[cfd][attrs_df[cfd] <= 0.8] = 0
# # # attrs_df[attrs_df == -1] = 0
# print('eeeeee', attrs_df)
max_g=env_df.max(axis=1)
attrs_df['max_prob'][attrs_df['max_prob']>0.65]=0
attrs_df['max_prob'][(attrs_df['max_prob']<=0.65) & (attrs_df['max_prob']>0)]=1
attrs_df.rename(columns={'max_prob':'clean'}, inplace=True)

print('///',attrs_df)
#

df_final=pd.concat([env_df,attrs_df['clean']],axis=1)
print(')))))))',df_final)
env_df = df_final.mask(
    df_final.eq(df_final.max(axis=1), axis=0), # the mask (True locations will get replaced)
    1,       # the replacements
    axis=0)
print(env_df)


# attrs_df['add'] = attrs_df.apply(lambda row: add(row['dark_corner'], row['hair'], row['gel_border'], row['gel_bubble'],row['ruler'], row['ink'],row['patches'] ), axis=1)
# # # # attrs_df = attrs_df.values


#
df_final=pd.concat([attrs_df['image'],env_df,attrs_df['label']],axis=1)
print(df_final)
df_final.to_csv('train_0bias.csv')
print('dark corner',df_final['dark_corner'][df_final['dark_corner']==1].sum())
print('hair',df_final['hair'][df_final['hair']==1].sum())
# print('gel border',df_final['gel_border'][df_final['gel_border']==1].sum())
print('gel bubble',df_final['gel_bubble'][df_final['gel_bubble']==1].sum())
print('ruler',df_final['ruler'][df_final['ruler']==1].sum())
print('clean',df_final['clean'][df_final['clean']==1].sum())
# print('ink',df_final['ink'][df_final['ink']==1].sum())
# print('patches',df_final['patches'][df_final['patches']==1].sum())
#
source_dir='/mount/neuron/Lamborghini/dir/pythonProject/CVPR/data/ISIC_2019_Training/'
target_dir='/mount/neuron/Lamborghini/dir/pythonProject/MICCAI/prompt_derm/domainbed/data/ISIC2019_train/'
#
# for index,row in df_final.iterrows():
#     filename=source_dir+row['image']+'.jpg'
#     sub_dir=row[row==1].index[0]
#     label=row['label']
#     if label==1:
#         label_name='mel'
#     elif label==0:
#         label_name='ben'
#     else:
#         label_name='!!!!!!!!!!!!!'
#     dest = shutil.copy2(filename, target_dir+sub_dir+'/'+label_name+'/')
#     print(dest)

