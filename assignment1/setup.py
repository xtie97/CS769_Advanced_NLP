import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_file", type=str, default="glove", choices=["glove", "fasttext"])
    return parser.parse_args()

def download_emb_file(emb_file):
    if 'glove' in emb_file:
        if os.path.exists("glove.42B.300d.txt"):
            exit(0)
        elif os.path.exists("glove.42B.300d.zip"):
            os.system("unzip glove.42B.300d.zip")
        else:
            os.system("wget http://nlp.stanford.edu/data/glove.42B.300d.zip") 
            os.system("unzip glove.42B.300d.zip")
    elif 'fasttext' in emb_file:
        if os.path.exists("crawl-300d-2M.vec"):
            exit(0)
        elif os.path.exists("crawl-300d-2M.vec.zip"):
            os.system("unzip crawl-300d-2M.vec.zip")
        else:
            os.system("wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip")
            os.system("unzip crawl-300d-2M.vec.zip")
    else:
        raise ValueError(f"Unknown embedding choice: {emb_file}. Pick from [glove, fasttext]")

if __name__ == '__main__':
    os.system("apt install unzip")
    args = get_args()
    download_emb_file(args.emb_file)


#
