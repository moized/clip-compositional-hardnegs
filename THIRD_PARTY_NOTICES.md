# Third-party code and data

Parts of this repository are adapted from the projects below. Each adapted file
names its source in its module docstring.

## Code

**ARO: "When and why vision-language models behave like bags-of-words, and what to do about it?"**
https://github.com/mertyg/vision-language-models-are-bows (commit 81fdd20), MIT License,
Copyright (c) 2023 Mert Yuksekgonul.
Adapted in `src/compbind/aro.py` (test-case loading, list of relations excluded from the
macro average) and `src/compbind/perturb.py` (word-order perturbations, caption pre-processing).

**SugarCrepe**
https://github.com/RAIVNLab/sugar-crepe (commit 0047054), MIT License,
Copyright (c) 2023 RAIVN Lab.
The evaluation protocol in `src/compbind/sugarcrepe.py`, the grammar-model baseline in
`src/compbind/textonly.py` and the idea of adversarial refinement in `src/compbind/refine.py`
follow this project.

**NegCLIP training code** (a fork of OpenCLIP)
https://github.com/vinid/neg_clip (commit e680f8e), MIT-style License,
Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman, Nicholas Carlini, Rohan Taori,
Achal Dave, Vaishaal Shankar, John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi,
Ludwig Schmidt.
The loss in `src/compbind/losses.py` and the optimiser settings in `src/compbind/training.py`
follow this code; no file was copied.

**CLIP_benchmark**
https://github.com/LAION-AI/CLIP_benchmark, MIT License.
Used as a library (datasets, class names, prompt templates); the zero-shot procedure in
`src/compbind/downstream.py` mirrors its `zeroshot_classification.py`.

**Consulted, no code used**
- https://github.com/tripletclip/TripletCLIP (TripletCLIP, NeurIPS 2024): its NegCLIP and NegCLIP++
  baselines are the precedent for comparing rule-based and fluent negatives.
- https://github.com/saic-fi/concept_centric_clip (C2LIP, CVPR 2026, MIT License, Copyright (c) Samsung
  AI Center - Cambridge, UK): the paper evaluates every model with CLIP_benchmark; the same tool's
  datasets, class names and prompts are used here for the zero-shot measurements.

The MIT License of the projects above:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this
> software and associated documentation files (the "Software"), to deal in the Software
> without restriction, including without limitation the rights to use, copy, modify, merge,
> publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
> to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
> INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
> PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
> FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
> OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
> DEALINGS IN THE SOFTWARE.

## Data and models (downloaded at run time, never stored in this repository)

| What | Source | Terms |
|---|---|---|
| COCO images and captions | https://cocodataset.org | images: Flickr terms of use; annotations: CC BY 4.0 |
| COCO Karpathy split files | BLIP project (Salesforce) storage | follows COCO |
| Visual Genome images and metadata (ARO's VG-Relation / VG-Attribution) | ARO repository links; https://visualgenome.org | CC BY 4.0 |
| COCO ids of ARO's Visual Genome images, `src/compbind/resources/aro_vg_to_coco.csv` (in this repository) | derived from Visual Genome's `image_data.json` (Krishna et al., *Visual Genome*, IJCV 2017) | CC BY 4.0 |
| SugarCrepe annotations | SugarCrepe repository | MIT |
| NegCLIP checkpoint and training file | ARO repository links | MIT (repository licence) |
| OpenAI CLIP ViT-B/32 weights | via OpenCLIP | MIT |
| GPT-2 | https://huggingface.co/gpt2 | MIT |
| CoLA grammar model | https://huggingface.co/textattack/distilbert-base-uncased-CoLA | see model card |
| CIFAR-10 / CIFAR-100 | https://www.cs.toronto.edu/~kriz/cifar.html | see dataset page |
| ImageNet-1k validation set (optional) | https://www.image-net.org | ImageNet terms of access |
