import time
import pdb
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

start_time = time.time()

csv_path = "/home/dipesh/Desktop/wgan/csv_data/RealWorld_Art.csv"
real_world= pd.read_csv(csv_path).to_numpy()
gen_feats = np.load("gen_vect_lab.npy")
legend_dict = {"first_feat":"real_world",
            "second_feat":"gan_gen"
                }

first_feat = real_world[:,:-1]
second_feat = gen_feats[:,:-1]
first_lab = real_world[:,-1]
second_lab = gen_feats[:,-1]
# first_feat = "{}_x_mask_loaded_{}_d".format(save_name_prefix,layer)
# second_feat = "{}_x_nomask_{}_d".format(save_name_prefix,layer)

# legend_dict = {"first_feat":"mask",
#             "second_feat":"no_mask"
#                 }

# first_feat = feats_dict[first_feat]
# second_feat = feats_dict[second_feat]

print("first_feat shape {}".format(first_feat.shape)) 
print("second_feat shape {}".format(second_feat.shape)) 
print("first_lab shape {}".format(first_lab.shape)) 
print("second_lab shape {}".format(second_lab.shape)) 

all_features = np.concatenate((first_feat, second_feat), axis = 0)
all_labels = np.concatenate((first_lab, second_lab), axis = 0)

dataset_label = np.zeros((all_features.shape[0],1))
dataset_label[first_feat.shape[0]:] = 1


tsne = TSNE(n_jobs=16)

embeddings = tsne.fit_transform(all_features)

vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]
vis_x_1 = vis_x[:first_feat.shape[0]]
vis_x_2 = vis_x[first_feat.shape[0]:]
vis_y_1 = vis_y[:first_feat.shape[0]]
vis_y_2 = vis_y[first_feat.shape[0]:]

sns.set(rc={'figure.figsize':(11.7,8.27)})
# NUM_CLASSES = 2
NUM_CLASSES = 65
palette = sns.color_palette("bright", NUM_CLASSES)

label_dict={0:legend_dict["first_feat"],
            1:legend_dict["second_feat"]
}

pdb.set_trace()
# hue_label = [label_dict[i] for i in dataset_label[:,0]]
# hue_label = [i for i in all_labels]
hue_label = [i for i in second_lab]

# plot = sns.scatterplot(vis_x, vis_y, hue=hue_label, legend='full', palette=palette)
plot = sns.scatterplot(vis_x_2, vis_y_2, hue=hue_label, legend='full', palette=palette)
tsne_save_path = "tsne.png"

plt.savefig(tsne_save_path)
# plt.savefig("generated_feats_only.png")
print("saved {} plot".format(tsne_save_path))
plt.clf()

print("--- {} mins {} secs---".format((time.time() - start_time)//60,(time.time() - start_time)%60))