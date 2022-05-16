import torch
from torch import nn
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import csv



@dataclass
class Instance:
    text: list
    sentiment: str
    def __iter__(self):
        return iter((self.text, self.sentiment))

class NLPDataset(Dataset):
    def __init__(self, data_path):
        self.instances:list(Instance) = []
        self.vocab = None
        """
        with pd.read_csv(data_path, sep=",", names=["review","sentiment"]) as data_csv:
            for index, row in self.data_csv.iterrows():
                self.instances.append(Instance(row["review"], row["sentiment"]))
        """
        with open(data_path, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                self.instances.append(Instance(row[0].split(" "), row[1].strip()))
        self.data_path = data_path

    def __getitem__(self, idx):
        # print(self.data_csv.iloc[0]["sentiment"])
        # tekst, sentiment
        return self.vocab.encode_text(self.instances[idx].text), self.vocab.encode_sentiment(self.instances[idx].sentiment)

    def set_vocab(self, vocab):
        self.vocab = vocab
    def __len__(self):
        return len(self.instances)


class Vocab:
    def __init__(self, max_size, min_freq, dataset_instances):
        self.max_size = max_size
        self.min_freq = min_freq
        self.freq_dict = {}
        self.stoi_text = {"<PAD>": 0, "<UNK>": 1}
        self.stoi_sentiment = {"positive": 0, "negative": 1}
        freq_dict = {}
        # creating the sorted freq_dict
        for row in dataset_instances:
            # print(f"row mi je: {row}")
            for item in row.text:
                # print(f"item mi je: {item}")
                if (item in freq_dict):
                    freq_dict[item] += 1
                else:
                    freq_dict[item] = 1
        sorted_dict = {}
        sorted_keys = sorted(freq_dict, key=freq_dict.get, reverse=True)
        for w in sorted_keys:
            if freq_dict[w] < min_freq:
                continue
            if len(sorted_dict) >= max_size and max_size != -1:
                break
            sorted_dict[w] = freq_dict[w]
        self.freq_dict = sorted_dict
        self.dict_size = len(self.freq_dict)
        self.vocabulary = list(self.freq_dict)
        for i in range(0, self.dict_size):
            self.stoi_text[list(self.freq_dict.keys())[i]] = i + 2
        self.itos_text = {v: k for k, v in self.stoi_text.items()}
        self.itos_sentiment = {v: k for k, v in self.stoi_sentiment.items()}

    def encode_text(self, text):
        ints = []
        [ints.append(self.stoi_text.get(token, self.stoi_text["<UNK>"])) for token in text]
        return torch.tensor(ints, device=device, dtype=torch.long)

    def encode_sentiment(self, sentiment):
        return torch.tensor(self.stoi_sentiment[sentiment], device=device)


def generate_embedding_matrix(vocabulary: Vocab, pretrained_file: str, random_matrix=False):
    if random_matrix:
        embeddings_tensor = torch.rand((len(vocabulary.vocabulary) + 2, 300), dtype=torch.float32, device=device)
        embeddings_tensor[0] = torch.zeros((1, 300), dtype=torch.float32, device=device)
        return torch.nn.Embedding.from_pretrained(embeddings_tensor, freeze=False, padding_idx=0)
    embedding_dictionary = {}
    with open(pretrained_file, "r") as file:
        word_embeddings = file.readlines()
    for line in word_embeddings:
        split_line = line.split(" ")
        embeddings_float_vector = [float(x.strip()) for x in split_line[1:]]
        # embedding_dictionary[vocabulary.stoi_text[split_line[0]]] = torch.tensor(embeddings_float_vector, device=device, dtype=torch.float32)
        embedding_dictionary[vocabulary.stoi_text[split_line[0]]] = embeddings_float_vector
    sorted_dict = {}
    sorted_keys = sorted(list(embedding_dictionary))
    for w in sorted_keys:
        sorted_dict[w] = torch.tensor(embedding_dictionary[w], device=device, dtype=torch.float32)
    embedding_dictionary = sorted_dict
    # INIT tensor reprezentacija i napuni ga vrijednostima
    embeddings_tensor = torch.rand((len(vocabulary.vocabulary) + 2, 300), dtype=torch.float32, device=device)
    embeddings_tensor[0] = torch.zeros((1, 300), dtype=torch.float32, device=device)
    for idx, embedding_vector in embedding_dictionary.items():
        embeddings_tensor[idx] = embedding_vector

    return torch.nn.Embedding.from_pretrained(embeddings_tensor, freeze=True, padding_idx=0)

def pad_collate_fn(batch):
    """
    Arguments:
      Batch:
        list of Instances returned by `Dataset.__getitem__`.
    Returns:
      A tensor representing the input batch.
    """
    texts, labels = zip(*batch) # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in texts], device=device) # Needed for later
    texts_tensor = torch.zeros((len(lengths), int(max(lengths))), dtype=torch.float32, device=device)
    for i, text in enumerate(texts):
        max_length = max(lengths)
        texts_tensor[i] = torch.cat( (text, torch.zeros(max_length - text.size()[0], device=device)))

    #texts = torch.tensor([text for text in texts])
    labels = torch.tensor([label for label in labels], device=device)
    # Process the text instances
    return texts_tensor, labels, lengths

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = NLPDataset("data/sst_train_raw.csv")
    train_vocab = Vocab(-1, 0, train_dataset.instances)
    train_dataset.set_vocab(train_vocab)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4,
                                  shuffle=False, collate_fn=pad_collate_fn)
    embedding_matrix = generate_embedding_matrix(train_vocab, "data/sst_glove_6b_300d.txt")
    texts, labels, lengths = next(iter(train_dataloader))
    print(f"Texts: {texts} ")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")
    for text in texts:
        pass
    #print(f"embedami textovi idk {embedding_matrix(torch.tensor(texts[1][2], device=device, dtype=torch.long))}")
    print(f"embedami text idk {embedding_matrix(texts[1][2].to(torch.long))}")
    a = embedding_matrix(texts[1][2].clone().detach().to(torch.long))
    batch_tensors = []
    for review in texts:
        word_tensors = []
        for word in review:
            word_embedding = embedding_matrix(word.to(torch.long))
            word_tensors.append(word_embedding)
        review_tensor = torch.stack(word_tensors, dim= 0)
        batch_tensors.append(review_tensor)
        pass
    batch_tensor = torch.stack(batch_tensors, dim=0)
    layer = nn.AdaptiveAvgPool2d((1, 300))
    batches_averaged = layer(batch_tensor)
    instance_text, instance_label = train_dataset.instances[3]
    print(f"Text: {instance_text}")
    print(f"Label: {instance_label}")
    print(f"Numericalized text: {train_vocab.encode_text(instance_text)}")
    print(f"Numericalized label: {train_vocab.encode_sentiment(instance_label)}")
    embedding_matrix = generate_embedding_matrix(train_vocab, "data/sst_glove_6b_300d.txt")
    numericalized_text, numericalized_label = train_dataset[3]
    # Koristimo nadjačanu metodu indeksiranja
    print(f"Numericalized text: {numericalized_text}")
    print(f"Numericalized label: {numericalized_label}")
    pass
