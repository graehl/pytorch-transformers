 #!/usr/bin/python3

# pip3 install --user torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# pip3 install --user transformers



import torch
import argparse
import sys


modelname="finmodel_distilbert"  #"bert-base-cased-finetuned-mrpc"


parser = argparse.ArgumentParser(description='3-class sentiment analyzer')

parser.add_argument('--model-name', type=str, default=modelname, help='model directory, must include model type in name')

parser.add_argument('--input-file', '-i', type=str, default=None, help='input file name, default is 3 test lines')

parser.add_argument('--output-format-short', '-s', action='store_true', default=False, help='one line, three scores (adding to 1.0)')

parser.add_argument('--output-format-class', '-c', action='store_true', default=False, help='one line, single best class')

parser.add_argument('--keep-case', '-k', action='store_true', default=False, help='do not lowercase the input (default: lowercase it)')

parser.add_argument('--max-length', '-m', type=int, default=512, help='maximum length handled by the model')

args = parser.parse_args()


usecfg = False
if usecfg:
    from transformers import (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
    )
    config = DistilBertConfig.from_pretrained(args.model_name, finetuning_task='sentiment3', num_labels=3)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_name, config=config)
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name, do_lower_case=(not args.keep_case))
else:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=(not args.keep_case))
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

model.to("cpu")
model.eval()


classes = ["0", "1", "2"]


texts = ["I hate you", "I love you", "Isomorphic protein matrices"]


iter = (sys.stdin if args.input_file == '-' else open(args.input_file)) if args.input_file is not None else texts

for t in iter:

    proc = t.strip() # if args.keep_case else t.strip().lower()

    input = tokenizer.encode(proc, return_tensors="pt", add_special_tokens=True, max_length=args.max_length)  # only keep first 512 (sub)words


    tensors = model(input)


    sm = torch.softmax(tensors[0], dim=1).tolist()[0]


    out = None
    if args.output_format_short:
        out = classes[sm.index(max(sm))]

    if args.output_format_class:
        if out is not None:
            out += '\t'
        out += '\t'.join('{0:.4f}'.format(x) for x in sm)
        print(out)
    elif args.output_format_short:
        print(out)
    else:
        print(f"input: {proc}")

        for i, cls in enumerate(classes):

            print(f"  {cls}: {round(sm[i]*100)}%")
