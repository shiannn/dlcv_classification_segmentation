import argparse
import pandas as pd

def count_acc(csv_path):
    pred_csv = pd.read_csv(csv_path)
    print(pred_csv)
    acc = 0
    for index, row in pred_csv.iterrows():
        image_id = row['image_id']
        pred_label = row['label'] ### int
        ground_truth = int(image_id.split('_')[0])
        acc += (pred_label == ground_truth)
    print("Acc: {}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-g', '--labels', help='ground truth masks directory', type=str)
    parser.add_argument('-p', '--pred', help='prediction csv', type=str)
    args = parser.parse_args()
    count_acc(args.pred)