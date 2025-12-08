# dsait4090Project

In the `requirements.txt` there is the `dexter-cqa` dependency. In order to run dexter, you need to use the environment of it as a local interpreter. 

It should be work by running the following commands:  

```aiignore
conda create -n dsait4090 python=3.11
conda activate dsait4090
pip install -r requirements.txt
```
Also installed a specific older version
```aiignore
pip install "huggingface_hub<0.26.0"
```

## File structure:

* `src/data` – loaders for 2WikiMultiHopQA dev set and corpus
* `src/experiments` – scripts for each experiment (baseline, random negs, hard negs, ADORE, etc.)
* `src/rag` – context construction logic (which docs go into the LLM for each experiment)
* `src/retrieval` – retrieval scripts using DEXTER (Contriever, later ADORE)
* `main.py` – optional entry point / CLI to run a chosen experiment
* `evaluation_metrics.py` – EM and CEM scoring for experiment_results/*.json


-----
AI File structure explaination: 

Three layers:
* restricted_data/ - Just the data:
  * dev.json – the 2WikiMultiHopQA questions + gold evidence
  * wiki_musique_corpus.json – all Wikipedia passages we retrieve from
* src/retrieval/ - One job: “given dev.json and the corpus, run a retriever and give me scores”. → contriever_retrieval.py = “run Contriever once, get:
  * queries (questions),
  * qrels (gold doc IDs),
  * corpus (all docs),
  * results (for each question: list of (doc_id, score))”.
  Every bulletpoint uses this instead of re-implementing retrieval.
* src/rag/ - Decision layer: “for this experiment, which doc IDs go into the LLM context?”
  * baseline: top-k from results
  * random negatives: top-k + random other docs
  * hard negatives: top-k gold docs + top-N high-score non-gold docs
* src/experiments/ - Full run layer: glue everything:
	1.	call retrieval (run_contriever)
	2.	use one contexts_*.py to pick doc IDs
	3.	turn them into text and call the LLM (generative_model_setup)
	4.	save JSON to experiment_results/ and score with evaluation_metrics.py

----
From the course Google Drive, download: dev.json and wiki_musique_corpus.json
!!! Add under `restricted_data` folder the `dev.json` and `wiki_musique_corpus.json` files


There is a known issue with HfRetriever - it returns a cuda-related error. 
I had to replace all cuda instances with .cpu / .device instances. For this purpose, I created the `[HfRetriever-copy.py](src/retrieval/HfRetriever-copy.py)`. If you have the same errors when following the onboarding guide, please copy that code in the official HfRetriever.py file.