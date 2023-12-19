import pickle
import torch
import argparse
import json
import matplotlib.pyplot as plt
from ipapy.ipachar import IPAConsonant

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

import utils
from extract_representations import extract_phoneme_embeds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):

    utils.set_seed(args.random_seed)

    accs_per_layer = {
        layer_idx:
            None
        for layer_idx in range(13)
    }

    for model_name in args.model_names:

        # Get dictionary of embeddings sorted by phoneme label
        phoneme_embeds_train, phoneme_embeds_test = extract_phoneme_embeds(model_name, 1000, 200, args.random_seed, args.batch_size, args.timit_path, args.num_layers)

        # Define layer-wise probes, to be trained and tested on frame-level Wav2Vec2 embeddings
        layer_probes = {
            layer_idx: LogisticRegression(solver="liblinear", penalty="l2", max_iter=10)
            for layer_idx in range(args.num_layers)
        }

        print("Training and evaluating probes...")

        for layer_idx in range(7, 19):

            # Put train data in the right format for our probing classifier + balance classes
            train_X, train_y = utils.data_loader(phoneme_embeds_train, layer_idx=layer_idx,
                                                 target_phonemes=args.target_phonemes)
            train_X, train_y = utils.balance_classes(train_X, train_y)

            # Train model
            layer_probes[layer_idx].fit(train_X, train_y)

            # save the model to disk
            filename = f'probing_models/{args.model_type}/{args.target_phonemes[0]}_{args.target_phonemes[1]}_probe_{layer_idx}.sav'
            pickle.dump(layer_probes[layer_idx], open(filename, 'wb'))

            # probing_model = pickle.load(
            #     open(f'probing_models/finetuned/{args.target_phonemes[0]}_{args.target_phonemes[1]}_probe_{layer_idx}.sav', 'rb'))

            # Put test data in the right format for our probing classifier + balance classes
            test_X, test_y = utils.data_loader(phoneme_embeds_test, layer_idx, target_phonemes=args.target_phonemes)
            test_X, test_y = utils.balance_classes(test_X, test_y)

            # Predict
            test_pred = layer_probes[layer_idx].predict(test_X)
            test_acc = accuracy_score(test_y, test_pred)
            print(f'Accuracy for layer {layer_idx}:', test_acc)
            accs_per_layer[layer_idx-7] = test_acc

    with open(f'{args.target_phonemes[0]}_{args.target_phonemes[1]}_accs_per_layer.json', 'w') as fp:
        json.dump(accs_per_layer, fp)

    if args.plot_results:
        with open(f"n_m_accs_per_layer.json", 'r') as fp:
            n_m_data = json.load(fp)
        with open(f"n_ng_accs_per_layer.json", 'r') as fp:
            n_ng_data = json.load(fp)
        with open(f"d_b_accs_per_layer.json", 'r') as fp:
            d_b_data = json.load(fp)
        with open(f"d_g_accs_per_layer.json", 'r') as fp:
            d_g_data = json.load(fp)
        with open(f"t_p_accs_per_layer.json", 'r') as fp:
            t_p_data = json.load(fp)
        with open(f"t_k_accs_per_layer.json", 'r') as fp:
            t_k_data = json.load(fp)

        plt.style.use('ggplot')
        fig = plt.figure(figsize=(8,3))
        gs = fig.add_gridspec(1, 2, hspace=0, wspace=0.05)
        plt.grid(False, axis='x')
        # plot model accuracies
        plt.plot(list(n_m_data.values()))
        plt.plot(list(n_ng_data.values()))
        plt.plot(list(d_b_data.values()))
        plt.plot(list(d_g_data.values()))
        plt.plot(list(t_p_data.values()))
        plt.plot(list(t_k_data.values()))

        ipa_symbol = IPAConsonant(name="velar nasal", descriptors=u"voiced velar nasal", unicode_repr=u"\u014B")
        plt.xlabel('Transformer Layer')
        plt.xticks(range(13))
        plt.ylabel('Accuracy')
        legend = ['/n/ vs /m/', f'/n/ vs /{ipa_symbol}/', '/d/ vs /b/', '/d/ vs /g/', '/t/ vs /p/', '/t/ vs /k/']
        plt.legend(legend, loc='lower right')
        plt.ylim((0.7, 1.0))

        plt.savefig(f'probing-accs.pdf', dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timit_path", default="/home/charlotte/Documents/TIMIT", type=str, help="path to directory where TIMIT data is stored"
    )
    parser.add_argument(
        "--model_names", default=["facebook/wav2vec2-base-960h"], type=list, help="list of huggingface checkpoints of wav2vec 2.0"
    )
    parser.add_argument(
        "--batch_size", default=1, type=int,
    )
    parser.add_argument(
        "--random_seed", default=42, type=int,
    )
    parser.add_argument(
        "--num_layers", default=20, type=int, help="number of model layers from which hidden representations are extracted"
    )
    parser.add_argument(
        "--target_phonemes", default=['d', 'g'], type=list, help="phoneme labels that probing model needs to predict"
    )
    parser.add_argument(
        "--model_type", default='finetuned', type=str, help="whether to use the pretrained or finetuned model version"
    )
    parser.add_argument(
        "--plot_results", default=False, type=bool, help="whether to plot the probing accuracies over layers for each phoneme contrast"
    )
    args = parser.parse_args()
    main(args)

