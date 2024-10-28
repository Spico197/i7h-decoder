import argparse

from src.i7h import i18n
from src.decoder import Decoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, choices=["encode", "decode"])
    parser.add_argument("string", type=str, help="string to encode or decode")
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM-135M",
        help="language model id from huggingface",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=1,
        help="lookahead of i18n words, >1 for multi-token prediction at each step, which would greatly increase combinatorial complexity and decoding time",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print middle process of decoding"
    )
    args = parser.parse_args()

    if args.action == "encode":
        print(i18n(args.string))
    else:
        decoder = Decoder(args.model, lookahead=args.lookahead)
        print(decoder.decode(args.string, verbose=args.verbose))
