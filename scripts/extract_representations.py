import torch
import math
import random
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config
from collections import defaultdict

import utils
features = defaultdict()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def extract_hidden_states(model, processor, inputs, batch_size, num_layers):
    '''
    Extract hidden states from Wav2Vec 2.0 encoder layers.
    :param model: Wav2Vec 2.0 model
    :param processor: Wav2Vec 2.0 processor
    :param inputs: list of TIMIT instances (i.e. timit['train'] or timit['test'])
    :param batch_size: number of instances per batch
    :return: dictionary containing frame-level embeddings saved per transformer layer and per utterance
    '''

    # Get batches of audio arrays
    input_batches = utils.get_batches(inputs, batch_size)

    # Here we will save all frame-level embeddings, sorted by layer and utterance
    frame_embeddings = {
        layer_idx: {
            utterance_idx: []
            for utterance_idx in range(batch_size * len(input_batches))
        }
        for layer_idx in range(num_layers)
    }

    for batch_idx, batch in enumerate(input_batches):

        print(f'Extracting hidden states from batch {batch_idx} out of {len(input_batches)} batches...')

        # Process input batch using the Wav2Vec2 processor
        processed_input_batch = processor(batch, sampling_rate=16000, return_tensors="pt", padding='longest').input_values

        # Run the model on the batch and extract hidden states (last cnn layer + transformer layers)
        for i in range(7):
            model.feature_extractor.conv_layers[i].activation.register_forward_hook(get_features(f'cnn_{i}'))

        with torch.no_grad():
            input_tensor = torch.tensor(processed_input_batch, device=DEVICE)

            # forward pass [with feature extraction]
            model_output = model(input_tensor, output_hidden_states=True, output_attentions=False)

            # get all hidden outputs
            cnn_outputs = [features[f'cnn_{i}'].cpu() for i in range(7)]
            cnn_outputs_transposed = [cnn_output.transpose(1, 2) for cnn_output in cnn_outputs]
            transformer_layers = model_output.hidden_states

            # Define the number of frames in which the audio signal is split
            num_frames = len(transformer_layers[0][0])

            window_sizes = {
                0: 64,
                1: 32,
                2: 16,
                3: 8,
                4: 4,
                5: 2,
                6: 1
            }

            # Get moving averages of CNN layers
            averaged_cnn_layers = []
            for layer_idx in range(7):
                window_size = window_sizes[layer_idx]
                windows = [cnn_outputs_transposed[layer_idx][0][window_size * y:window_size * (y + 1)] for y in range(num_frames)]
                averaged_windows = [torch.mean(window, dim=0).cpu() for window in windows]
                averaged_cnn_layers.append(averaged_windows)

            # Cast each CNN layer into tensor and add extra dimension at position 0 --> dims = (number_of_utterances, frames_per_utterance, embedding_size)
            tensorified_averaged_cnn_layers = [utils.tensorify(l).unsqueeze(0) for l in averaged_cnn_layers]

            # # Send CNN output through feature projection
            # projected_cnn_layers = [model.feature_projection(layer)[1] for layer in tensorified_averaged_cnn_layers]

            # Add CNN layers to all_layers
            all_layers = list(transformer_layers)
            all_layers[:0] = [l for l in tensorified_averaged_cnn_layers]

        # Save frame embeddings and labels for each layer
        for layer_idx, layer in enumerate(all_layers):

            for utterance_idx, utterance in enumerate(layer):

                # Get the index of the current utterance
                total_utterance_idx = (batch_idx * batch_size) + utterance_idx

                for frame_embedding in utterance:
                    frame_embeddings[layer_idx][total_utterance_idx].append(frame_embedding.cpu())

    return frame_embeddings

def sort_embeds_per_phoneme(data, frame_embeddings, num_layers):

    frame_embeds_per_phoneme = {
        layer_idx: defaultdict(list)
        for layer_idx in range(num_layers)
    }

    for layer_idx, layer in frame_embeddings.items():

        for utterance_idx, utterance in layer.items():

            # retrieve phoneme annotation for the current utterance
            phonemes = data[utterance_idx]["phonetic_detail"]
            # text = data[utterance_idx]["word_detail"]["utterance"]
            # print(text)
            phoneme_indeces = defaultdict(list)

            for start, stop, phoneme in zip(phonemes['start'], phonemes['stop'], phonemes['utterance']):

                # divide start and stop point by sample rate (16000 hz) and frame length (0.020 sec)
                start_index = math.floor((start / 16000) / 0.020)
                stop_index = math.ceil((stop / 16000) / 0.020)
                phoneme_indeces[phoneme].extend(range(start_index, stop_index))

            # find embeddings corresponding to phoneme indeces and save them per layer and per phoneme
            for phoneme, indeces in phoneme_indeces.items():
                phoneme_embeds = [utterance[idx] for idx in indeces if idx < len(utterance)]
                frame_embeds_per_phoneme[layer_idx][phoneme].extend(phoneme_embeds)

    return frame_embeds_per_phoneme

def extract_phoneme_embeds(model_name, train_size, test_size, random_seed, batch_size, timit_path, num_layers, averaged=False):

    utils.set_seed(random_seed)

    print("Loading dataset...")
    timit = load_dataset("timit_asr", data_dir=timit_path).shuffle(seed=random_seed)

    # Generate random indeces to select a subset of the train and test data
    print(len(timit['train']), len(timit['test']))
    train_indeces = random.sample(range(0, len(timit['train'])), train_size)
    test_indeces = random.sample(range(0, len(timit['test'])), test_size)

    train_subset = timit['train'].select(train_indeces)
    test_subset = timit['test'].select(test_indeces)

    # Load processor and model
    print("Loading model and processor...")
    config = Wav2Vec2Config.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name, config=config)
    model = Wav2Vec2Model.from_pretrained(model_name, config=config)

    # Put model in evaluation mode
    model.eval()
    model.to(DEVICE)

    # Extract frame-level representations from each layer of the model and label them with their respective phoneme
    frame_embeds_train = extract_hidden_states(model, processor, train_subset, batch_size, num_layers)
    frame_embeds_test = extract_hidden_states(model, processor, test_subset, batch_size, num_layers)

    if averaged:
        phoneme_embeds_train = sort_embeds_per_phoneme_averaged(train_subset, frame_embeds_train, num_layers)
        phoneme_embeds_test = sort_embeds_per_phoneme_averaged(test_subset, frame_embeds_test, num_layers)
    else:
        phoneme_embeds_train = sort_embeds_per_phoneme(train_subset, frame_embeds_train, num_layers)
        phoneme_embeds_test = sort_embeds_per_phoneme(test_subset, frame_embeds_test, num_layers)

    return phoneme_embeds_train, phoneme_embeds_test
