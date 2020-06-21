import pandas as pd
import torch.utils.data as data_utils
import scipy

def load_data(batch_size=64):

    art= pd.read_csv('/home/dipesh/data/office_home_csv/Art_Art.csv')
    clipart= pd.read_csv('/home/dipesh/data/office_home_csv/Clipart_Clipart.csv')
    product= pd.read_csv('/home/dipesh/data/office_home_csv/Product_Product.csv')
    real_world= pd.read_csv('/home/dipesh/data/office_home_csv/RealWorld_RealWorld.csv')


    # print("art.shape",art.shape)
    # print("clipart.shape",clipart.shape)
    # print("product.shape",product.shape)
    # print("real_world.shape",real_world.shape)

    art = torch.tensor(art.values)
    clipart = torch.tensor(clipart.values)
    product = torch.tensor(product.values)
    real_world = torch.tensor(real_world.values)


    dict_name = {"source_2":"art",
            "source_1":"real_world"  ,
            "source_3":"clipart"  ,
            "target":"product"
    }
    # dict_name = {"source_1":"art",
    #         "source_2":"clipart",
    #         "target":"product"            

    dict_data = {"art":art,
                "clipart":clipart,
                "real_world":real_world,
                "product":product

    }

    source_1 = dict_data[dict_name["source_1"]]
    source_2 = dict_data[dict_name["source_2"]]
    source_3 = dict_data[dict_name["source_3"]]
    target = dict_data[dict_name["target"]]


    target[target[:,-1] > 24,-1] = 25

    source_1 = source_1[source_1[:,-1] <= 25,:]
    source_2 = source_2[source_2[:,-1] <= 25,:]
    source_3 = source_3[source_3[:,-1] <= 25,:]


    source_1_plus_source_2 = torch.cat((source_1, source_2),0)
    # print('source_1_plus_source_2.shape',source_1_plus_source_2.shape)
    source_features = (source_1_plus_source_2[:,:-1])
    source_labels = source_1_plus_source_2[:,-1]
    target_features = target[:,:-1]
    target_labels = target[:,-1]
    # print('source_features.shape,source_labels.shape',source_features.shape,source_labels.shape)
    # print('target_features.shape, target_labels.shape',target_features.shape, target_labels.shape)

    # exit()

    train1 = data_utils.TensorDataset(source_1[:,:-1], source_1[:,-1])
    train2 = data_utils.TensorDataset(source_2[:,:-1], source_2[:,-1])
    train3 = data_utils.TensorDataset(source_3[:,:-1], source_3[:,-1])
    test = data_utils.TensorDataset(target_features, target_labels)
    source_1_loader = data_utils.DataLoader(train1, batch_size=batch_size, shuffle=False)
    source_2_loader = data_utils.DataLoader(train2, batch_size=batch_size, shuffle=False)
    source_3_loader = data_utils.DataLoader(train3, batch_size=batch_size, shuffle=False)
    target_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=False)
    return source_1_loader, source_2_loader, source_3_loader, target_loader, dict_name

