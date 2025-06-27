# 📘 Project Title: Generalisability of TIGER


## 🧑‍💻 Team Members
- Jose Garcia – jose.garcia.carrillo@student.uva.nl  
- Karolina Hájková – karolina.hajkova2@student.uva.nl  
- Adriana Haralambieva – adriana.haralambieva@student.uva.nl  
- Lucia Šikulová – lucia.sikulova2@student.uva.nl   

## 👥 Supervising TAs
- Kidist Mekonnen (Main Supervisor) - k.a.mekonnen@uva.nl
- Akis Lionis (Co-supervisor) - akis.lionis@student.uva.nl


---

## 🧾 Project Abstract
Sequential recommendation systems try to predict what a user will want next based on their past actions. TIGER (Transformer Index for GEnerative Retrieval) is a new approach that uses a generative model to create meaningful item identifiers, called Semantic IDs, with a Transformer model. In this paper, we reproduce the original TIGER results and go beyond standard accuracy metrics to evaluate the model using fairness and diversity measures such as the Gini coefficient, KL divergence, $\Delta$GAP, and Coverage. We also test how well TIGER works with new unseen users and different datasets, especially in cold-start situations where there is limited user history. We evaluate TIGER's effectiveness, fairness and adaptability across datasets and cold-start setting.

---

## 📊 Summary of Results


### Reproducibility 
- TIGER’s reported results for recall and NDCG could not be fully reproduced due to several issues like differences in the model used to generate the item embeddings (sentence-t5-xl instead of sentence-t5-xxl) and limited textual semantices.
- Our replication, with corrected data splits, shows lower absolute performance but preserves ranking trends across datasets.
- Baseline results (SASRec and S3-Rec) were also slightly lower than reported in their original paper but followed similar patterns.


### Extensions
- **Fairness & Diversity**: Measured Gini, KL divergence, ∆GAP, and Coverage. TIGER achieves high brand diversity per user (lower Gini), reduces popularity bias (lower ΔGAP), but suffers from low catalog coverage and brand bias in cold-start settings.
- **Generalisation Settings**: In W-Base (all users), W-Reduced (80% users), and S-TIGER (80% users with unseen users for evaluation) setups, reduced data and unseen users lower accuracy but improve coverage and certain fairness metrics.
- **New Dataset Evaluation**: On Amazon Reviews 2023 (five categories), h@10 remains low (<0.04) due to fewer usable review sequences, highlighting preprocessing’s critical role.
- **Ablation Study on Temperature**: To assess how temperature affects our fairness and diversity metrics.


---

## 🛠️ Task Definition  
We frame sequential recommendation as a **generative retrieval** problem:  
- **Input**: A user’s past interaction history encoded as a sequence of Semantic ID token sequences (discretized via RQ-VAE from item title, brand, and category embeddings).  
- **Output**: The predicted Semantic ID for the next item, which is mapped back to an actual item and ranked based on generation probability.


---

## 📂 Datasets

- [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)  
  - **Pre-processing** (code not provided): Removed users/items with <5 interactions; used leave-one-out evaluation; max history length = 20.  
  - **Subsets**:  
    - Beauty: 22,363 users, 12,101 items  
    - Sports & Outdoors: 35,598 users, 18,357 items  
    - Toys & Games: 19,412 users, 11,924 items  
    - Cold-start splits: W-Reduced (80% users), S-TIGER (unseen users)    
  - **Dataset size**: varies by category (~19k–35k users, ~12k–18k items)  
  - **Attributes for item fairness**: brand identifiers.  
  - **Other attributes**: title, brand, category.  

- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/index.html)  
  - **Pre-processing**: Filtered out users with <3 reviews; limited to first 16,384 reviews per category.
  - **Subsets**:  
    - Beauty: 580 users, 10,653 items  
    - Sports & Outdoors: 1,253 users, 13,931 items  
    - Toys & Games: 873 users, 13,985 items  
    - Health & Personal Care: 450 users, 8,578 items  
    - Electronics: 1,326 users, 13,088 items 
  - **Dataset size**: varies by category (~400–1.5k users, ~8.5k–14k items)
  - **Other attributes**: rich metadata (timestamps, categories).  

---

## 📏 Metrics

We evaluate both traditional accuracy and fairness/diversity metrics to provide a holistic view of performance:

- **Hit Rate (h@k)**  
  - Measures if the true next item is in the top-k predicted items.
- **NDCG (Normalized Discounted Cumulative Gain)**  
  - Evaluates ranking quality, giving higher weight to correct items appearing higher in the list.
- **Gini Coefficient**  
  - Measures how equally different brands are recommended (lower = more equal).
- **KL Divergence**  
  - Measures the mismatch between historical and predicted brand distributions per user.
- **ΔGAP (Popularity Bias)**  
  - Compares popularity of recommended items to the user’s history.
- **Coverage**  
  - Measures the percentage of unique items recommended across all users.


---

## 🔬 Baselines & Methods

We compare against:

- [SASRec](https://github.com/kang205/SASRec) – A Transformer-based sequential recommender modeling short- and long-term preferences.
- [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec) – A self-supervised sequential recommender with bi-directional pretraining.

To run the baselines in a reproducible manner, we provide the commands needed to create the files needed for the new metrics. These work to run the evaluation for both SASRec and S3-Rec.

This command does not need to be run as the ouptut files are already provided but is here for reproducibility. Use it for all desired categories.

```
srun python build_fairness_data.py --data_name Beauty
srun python run_finetune_full.py --data_name Beauty 
```


## 🧠 High-Level Description of Method

1. **Content Encoding**: Extract item title, brand, category; embed with a pretrained text encoder (sentence-t5).  
2. **Semantic ID Generation**: Apply RQ-VAE to quantize embeddings into multi-level discrete tokens, forming each item’s Semantic ID.  
3. **Sequence Modeling**: Convert user histories into ordered Semantic ID sequences; train an autoregressive Transformer decoder (8 layers, 8 heads) to predict next ID.  
4. **Decoding & Ranking**: Generate the next Semantic ID via temperature-controlled sampling; map tokens back to item IDs and rank by generation probability. 
5. **Design Choices**:  
   - Vector quantization for structured, generalizable IDs.  
   - End-to-end generative retrieval replacing ANN indexing.  
   - Temperature tuning to balance diversity vs. popularity bias.

Our method can be run using the same configurations described on the README file of the original repository (copied on README_2.md).

## Installing
Clone the repository and run `pip install -r requirements.txt`. Dataset download is automatically handled by the code.

## Executing
RQ_VAE tokenizer model and the retrieval model are trained separately, using two separate training scripts. Configs are handled using `gin-config`. 

The `train` functions defined under `train_rqvae.py` and `train_decoder.py` are decorated with `@gin.configurable`, which allows all their arguments to be specified with `.gin` files. These include most parameters one may want to experiment with (e.g. dataset, model sizes, output paths, training length).

Sample configs for the `train.py` functions are provided under `configs/`. Configs are applied by passing the path to the desired config file as argument to the training command.

To train both models on the **Amazon Reviews** dataset, run the following commands:
* **RQ-VAE tokenizer model training:** Trains the RQ-VAE tokenizer on the item corpus. Executed via `python train_rqvae.py configs/rqvae_amazon.gin`
* **Retrieval model training:** Trains retrieval model using a frozen RQ-VAE: `python train_decoder.py configs/decoder_amazon.gin`

The same methodology can be used for training on the Amazon Reviews 2014 and Amazon Reviews 2023 datasets. To switch between them, change the dataset parameter of the config files. Set the `dataset_split` to the desired dataset category to use.

Additionally, we added 4 new parameters to the configuration files to make our extensions work.

-strong_generalization,
-reduce_users,
-split_dataset, and
-split_qty

Set `strong_generalization=True` to run the S-Tiger training configuration.
Set `reduce_users=True` to run the W-Reduced training configuration.
Set `split_dataset=True` when using the Amazon Reviews 2023 dataset, in order to limit the number of items read by the model. This helps with memory limitation. If False, the entire dataset for the category will be downloaded.
If `split_dataset=True`, set `split_qty` to the number of items to be read.

---

## 🌱 Proposed Extensions

- Incorporated fairness and diversity metrics (Gini, KL divergence, ∆GAP, Coverage) beyond accuracy.  
- Designed and evaluated generalisation configurations (W-Reduced, S-TIGER) to assess cold-start and data-reduction effects.  
- Extended experiments to Amazon Reviews 2023 across five categories, highlighting preprocessing’s impact.  
- Performed a temperature ablation study to analyze trade-offs between diversity and popularity alignment.


### References
* [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065) by Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy