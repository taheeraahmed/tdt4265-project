from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader

def create_box_plot(data_frame, label_text):
    plt = matplotlib.pyplot

    # plot bg
    sns.set_style("whitegrid")

    #Size of the plot
    plt.figure(figsize=(10, 8))

    # setting color of the plot
    color = sns.color_palette('pastel')

    # Using seaborn to plot it horizontally with 'color'
    ax = sns.catplot(data=data_frame, x="size", y="label", kind="box", palette=color, showfliers=False, order = list(range(0,9)))

    # Title of the graph
    plt.title('Spread of boxsizes given a label', size = 20)

    # Horizontal axis Label
    plt.xlabel('Size of boxes', size = 17)
    # Vertical axis Label
    plt.ylabel('Labels', size = 17)
    ax.set_yticklabels(label_text['text_labels'], rotation='horizontal', fontsize=5)

    # x-axis label size
    plt.xticks(size = 17)
    #y-axis label siz
    plt.yticks(size = 15)

    # display plot
    plt.show()

def create_sum_plot(observations): 
    sns.set_style({'grid.color': 'white'})
    
    # Sum the data, plot bar with given size using color defined
    ax = sns.catplot(kind='bar', data=observations, x='labels', y='observations', color='#CE389C')

    # Title of the graph
    plt.title('No. of observations given a label', size = 20)

    # Horizontal axis Label
    plt.xlabel('Labels', size = 17)
    # Vertical axis Label
    plt.ylabel('No.of Observation', rotation = 'vertical', size = 17)

    # x-axis label size, setting label rotations
    plt.xticks(rotation = 'horizontal', size = 14)
    # y-axis label size
    plt.yticks(size = 14)

    # removing the top and right axes spines, which are not needed
    sns.despine()

    # display plot
    plt.show()
    
def create_ratio_width_and_height(data_frame, label_text):
    plt = matplotlib.pyplot

    # plot bg
    sns.set_style("whitegrid")

    #Size of the plot

    # setting color of the plot
    color = sns.color_palette('pastel')

    # Using seaborn to plot it horizontally with 'color'
    ax = sns.catplot(data=data_frame, x="ratio", y="label", kind="box", palette=color, showfliers=False, order = list(range(0,9)))

    # Title of the graph
    plt.title('Spread of ratios (height/width) given a label', size = 20)

    # Horizontal axis Label
    plt.xlabel('Ratio of boxes', size = 17)
    # Vertical axis Label
    plt.ylabel('Labels', size = 17)
    ax.set_yticklabels(label_text['text_labels'], rotation='horizontal', fontsize=5)

    # x-axis label size
    plt.xticks(size = 17)
    #y-axis label siz
    plt.yticks(size = 15)

    # display plot
    plt.show()

def analyze_something(dataloader, cfg):
    """
    TODO: 
    [X] Lage en dataframe som ser sånn her ut
    +-------+-------+
    | Label | Boxes |
    +-------+-------+
    
    [X] Lage en kolonne til som heter size

    Begynne å analysere dataframe-en
    [X] Type of task -> Type of diagram (x-akse,y-akse) 
    [X] Size of boxes of given the same label -> Box plot (size, labels)
    [X] No. of observations given a label in general -> Bar diagram (labels, amount)
    """
    # Creating a dataframe to store the data
    data_frame = pd.DataFrame()
    for batch in tqdm(dataloader):
        for labels, boxes in zip(batch["labels"], batch["boxes"]):
            for label, box in zip(labels, boxes):
                data_frame = data_frame.append({"label": int(label.detach().numpy()), 
                                                "x_min": box[0].detach().numpy(), 
                                                "y_min": box[1].numpy(), 
                                                "x_max": box[2].numpy(), 
                                                "y_max": box[3].numpy(),
                                                }, ignore_index=True)

    # Changing the dtype for label to 'category'
    data_frame["label"] = data_frame["label"].astype("category")

    # Create a new column with the size of each boxes   
    height = 128
    width = 1024
    data_frame["size"] = data_frame.apply(lambda row: (width*(row["x_max"] - row["x_min"])) * (height*(row["y_max"] - row["y_min"])), axis=1)
    data_frame["height"] = data_frame.apply(lambda row: (height*(row["y_max"] - row["y_min"])), axis=1)  
    data_frame["width"] = data_frame.apply(lambda row: (width*(row["x_max"] - row["x_min"])), axis=1) 
    data_frame["ratio"] = data_frame.apply(lambda row: (row["height"]/row["width"]), axis=1)  
    
    data_frame.to_csv('data_frame.csv', index=False)

    return data_frame



def main():
    config_path = "configs/tasks/task21_baseline.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"
    
    # Reading or creating data_frame
    try:
        print("Loading dataframe from file...")
        data_frame = pd.read_csv('dataset_exploration/dataset/data_frame.csv')
        observations = pd.read_csv('dataset_exploration/dataset/observations.csv')
        print("Dataframe loaded")
    except:
        print("Creating dataframe...")
        dataloader = get_dataloader(cfg, dataset_to_analyze)
        data_frame = analyze_something(dataloader, cfg)
        print("Dataframe created")

    data_frame["label"] = data_frame["label"].astype("category")
    data_frame["x_min"] = data_frame["x_min"].astype("float64")
    data_frame["y_min"] = data_frame["y_min"].astype("float64")
    data_frame["x_max"] = data_frame["x_max"].astype("float64")
    data_frame["y_min"] = data_frame["y_min"].astype("float64")

    # Creating the plots
    label_text = pd.read_csv("dataset_exploration/dataset/labels.csv")
    create_sum_plot(observations)
    create_box_plot(data_frame,label_text)
    create_ratio_width_and_height(data_frame, label_text)


if __name__ == '__main__':
    main()
