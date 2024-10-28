# 🫠 I18n Decoder

[RimoChan](https://github.com/RimoChan) showed a way to encode strings in the ["i18n format"](https://github.com/RimoChan/i7h), where each word is encoded with `leading letter + number of middle letters + trailing letter` (e.g. "hello" -> "h3o").

This repository contains a decoder for such strings.

The decoder is based on a language model, and it is **not guaranteed** to reveal the original string.

## 🚀 QuickStart

```bash
# encode a string
$ python -m src.cli encode "Alice was beginning to get very tired of sitting by her sister on the bank"
A3e w1s b7g to g1t v2y t3d of s5g by h1r s4r on t1e b2k

# decode a string
$ python -m src.cli decode "A3e w1s b7g to g1t v2y t3d of s5g by h1r s4r on t1e b2k" --verbose
[nltk_data] Downloading package words to /home/username/nltk_data...
[nltk_data]   Package words is already up-to-date!
Decoded:  <-> Remaining: A3e w1s b7g to g1t v2y t3d of s5g by h1r s4r on t1e b2k
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 36/36 [00:02<00:00, 13.59it/s]
Decoded: Annie was <-> Remaining:  b7g to g1t v2y t3d of s5g by h1r s4r on t1e b2k
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:04<00:00, 13.09it/s]
Decoded: Annie was bemoaning <-> Remaining:  to g1t v2y t3d of s5g by h1r s4r on t1e b2k
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.04it/s]
Decoded: Annie was bemoaning to get <-> Remaining:  v2y t3d of s5g by h1r s4r on t1e b2k
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 12.90it/s]
Decoded: Annie was bemoaning to get very <-> Remaining:  t3d of s5g by h1r s4r on t1e b2k
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:02<00:00, 11.71it/s]
Decoded: Annie was bemoaning to get very tired <-> Remaining:  of s5g by h1r s4r on t1e b2k
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 115/115 [00:10<00:00, 11.03it/s]
Decoded: Annie was bemoaning to get very tired of sobbing <-> Remaining:  by h1r s4r on t1e b2k
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.61it/s]
Decoded: Annie was bemoaning to get very tired of sobbing by her <-> Remaining:  s4r on t1e b2k
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 204/204 [00:20<00:00, 10.09it/s]
Decoded: Annie was bemoaning to get very tired of sobbing by her sister <-> Remaining:  on t1e b2k
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00,  9.51it/s]
Decoded: Annie was bemoaning to get very tired of sobbing by her sister on the <-> Remaining:  b2k
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:03<00:00,  8.27it/s]
Annie was bemoaning to get very tired of sobbing by her sister on the back
```

## ⚙️ Installation

```bash
# tqdm, nltk, transformers, torch, protobuf
pip install -r requirements.txt
```

## 🪪 License

[MIT](LICENSE)

## ❤️ Acknowledgments

- [RimoChan/i7h](https://github.com/RimoChan/i7h)
