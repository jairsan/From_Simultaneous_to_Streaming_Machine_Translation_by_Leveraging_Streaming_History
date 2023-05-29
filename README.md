# From Simultaneous to Streaming Machine Translation by Leveraging Streaming History
This repository contains the code of the paper [From Simultaneous to Streaming Machine Translation by Leveraging Streaming History](https://aclanthology.org/2022.acl-long.480/).
```
@inproceedings{iranzo-sanchez-etal-2022-simultaneous,
    title = "From Simultaneous to Streaming Machine Translation by Leveraging Streaming History",
    author = "Iranzo-S{\'a}nchez, Javier  and
      Civera, Jorge  and
      Juan-C{\'\i}scar, Alfons",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.480",
    doi = "10.18653/v1/2022.acl-long.480",
    pages = "6972--6985",
}
```

## Code
This is the code used to train the wait-k models used for the experiments. This is based on the multi-path
wait-k implementation of https://github.com/elbayadm/attn2d. I have only copied the appropiate
example folder so that it can be adapted to a current fairseq version. I think that the most interesting part is 
incremental_simultaneous_beam_search.py, which is a beam search decoder
that can take a partial target prefix as input. The original fairseq decoder was not very optimized for this task
(prefix translation), so I wrote my own, in order to speed-up decoding and as a personal
challenge.
## Eval Results
This folder contains some simple scripts to reproduce the results reported on the paper.