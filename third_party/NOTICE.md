# Third-party material

`data/prompts/*.md` (the degree-of-belief question-generation and grading prompts), the universe
contexts in `data/contexts/` for cake_bake, roman_concrete, antarctic_rebound, fda_approval and
kansas_abortion, and the training defaults in `sdf/train.py` come from
[safety-research/believe-it-or-not](https://github.com/safety-research/believe-it-or-not)
(science-synth-facts, commit b22a45a), MIT licence, reproduced in `believe-it-or-not-LICENSE`.

The false contexts for military_submarine and italian_food are the five `fact` framings of
`model-organisms-for-real/synthetic-documents-<fact>`; their true contexts were written for this
project.
