"""
Experiment ladder roadmap for the reusable EEG dementia-classification project.

This file is intentionally a planning artifact, not an execution script.
It is meant to be imported/read alongside the project modules so the
experiments are run in a controlled staged order.

Grounding in current codebase
----------------------------
The roadmap below is aligned to the modules currently present in source:
- preprocessing.py: subject preparation, windowing, QC, macro grouping
- feature_extraction.py: node-feature extraction and aggregation
- connectivity_extraction.py: connectivity metrics and aggregation
- graph_construction.py: segment/macro/subject graphs, topology builders,
  edge-weight separation, graph-bank construction/fusion, PyG conversion
- dense.py: node-only, connectivity-only (MLP/CNN), dual-branch dense models
- gnn.py: simple fixed-graph GNN, fused graph-bank GNN, dual-branch graph model,
  graph readout options
- models_mil.py: mean / attention / gated MIL and subject fusion
- trainer.py, evaluate.py, metrics.py: training, aggregation, CV summaries

Design principle
----------------
At each block, vary one major idea while freezing the rest.
Do not start expensive topology-learning or fusion experiments until the
strongest graph level, model family, and basic topology/weight choices are known.

Practical note
--------------
For the first working version of the ladder, prefer direct imports from the
actual modules (dense.py, gnn.py, graph_construction.py, etc.) rather than a
registry-first workflow. The current registry.py still looks like a generic
placeholder layer and may need cleanup before it becomes the main entry point.

======================================================================
1. OVERALL EXPERIMENT PHILOSOPHY
======================================================================

The project should answer the following questions in order:

A. What graph construction level is the most stable and scientifically useful?
   - segment graph
   - macro graph
   - subject graph

B. Before using message passing, do dense baselines already explain most of the gain?
   - node-only
   - connectivity-only
   - node + connectivity dual-branch dense

C. If graph models help, is the gain coming from:
   - graph level
   - topology source
   - edge weights
   - connectivity metric / band choice
   - subject aggregation
   - graph readout
   - constrained topology fusion

The ladder is therefore ordered to answer high-level conceptual questions first,
then lower-level mechanism questions, then expensive refinement questions.

======================================================================
2. LOCKED EVALUATION PROTOCOL
======================================================================

Lock this before comparing ideas.

Dataset policy
--------------
- Start with AHEAP only for the first full ladder pass.
- Bring CAUEEG only after the first complete reduced ladder is stable.
- Do not tune separately on CAUEEG during early blocks.

Label policy
------------
- Keep the task definition fixed within one ladder pass.
- Do not mix binary and 3-class settings inside the same early comparison block.
- If the main project target is multiclass dementia classification, keep that fixed.

Split policy
------------
- Subject-level splitting only.
- Same outer folds across all models in a block.
- Same train/val/test subject identities for all compared settings.
- Freeze split seeds before Block 0 completes.

Seed policy
-----------
- Use one split seed for debugging.
- Use 3 train seeds for the first stable comparison blocks.
- Expand to 5 train seeds only for finalists.
- Keep split seed fixed while varying train seed.

Primary model-selection metric
------------------------------
- Validation balanced accuracy for checkpoint selection.

Primary reporting metric
------------------------
- Test balanced accuracy.

Secondary reporting metrics
---------------------------
- macro-F1
- accuracy
- PR-AUC macro OVR
- ROC-AUC macro OVR
- Brier score
- confusion matrix

Aggregation/reporting policy
----------------------------
- Always report subject-level metrics.
- For segment/macro experiments, save both instance-level outputs and aggregated
  subject outputs.
- Report mean and std across train seeds.
- Keep a per-fold results table and a per-subject prediction table.

Frozen preprocessing for early blocks
-------------------------------------
Use one preprocessing recipe across Blocks 0-4:
- same bandpass / notch / reference choices
- same segment length and overlap
- same QC thresholds
- same normalization mode
- same feature groups for node features
- same frequency bands for connectivity unless the block explicitly studies them

======================================================================
3. EXPERIMENT LADDER BLOCKS IN ORDER
======================================================================

----------------------------------------------------------------------
BLOCK 0. EVALUATION PROTOCOL LOCK + SANITY / REPRODUCIBILITY
----------------------------------------------------------------------
Scientific question
- Can the current pipeline produce stable subject-level outputs before we test ideas?

Freeze
- Dataset: AHEAP only
- One fixed preprocessing recipe
- One fixed feature set
- One fixed connectivity metric for debug runs (coherence)
- One fixed graph level for the first smoke test (subject graph is simplest)

Vary
- Only debugging / reproducibility checks

Recommended settings
- Subject graph + node-only MLP
- Subject graph + connectivity-only MLP
- Subject graph + simple fixed-graph GNN
- 1 split seed, then 3 train seeds after the first pass works

Must verify
- folds are stable across reruns
- output directories and prediction tables save correctly
- best checkpoint selection is consistent
- subject aggregation is correct when needed
- repeated run with same seeds reproduces the same results
- metrics summary and CV summary are internally consistent

Metrics to compare
- exact reproducibility of fold assignments
- exact reproducibility of predictions with same seed
- balanced accuracy / macro-F1 consistency across reruns
- presence and correctness of saved history / checkpoint / prediction files

Interpretation
- If this block fails, do not start scientific ablations.
- Most later disagreements become uninterpretable if seed behavior, saving,
  or aggregation are unstable.

Decision rule
- Move forward only after repeated identical runs give identical outputs and
  all required artifacts are saved correctly.

----------------------------------------------------------------------
BLOCK 1. GRAPH-LEVEL COMPARISON
----------------------------------------------------------------------
Scientific question
- At what temporal abstraction should one graph live: segment, macro, or subject?

Freeze
- Model family: use the simplest comparable models first
- Topology: fixed simple topology for graph models
- Edge weights: fixed simple choice
- Connectivity metric/band: one fixed choice
- Subject aggregation: appropriate default only

Vary
- Graph construction level only:
  1) segment graph
  2) macro graph
  3) subject graph

Recommended settings
A. Dense side
- segment-level node-only + mean MIL
- macro-level node-only + mean MIL or light subject fusion
- subject-level node-only without MIL

B. Connectivity side
- segment-level connectivity-only MLP + mean MIL
- macro-level connectivity-only MLP + light subject fusion
- subject-level connectivity-only MLP

C. Graph side
- segment-level simple fixed-graph GNN + mean MIL
- macro-level simple fixed-graph GNN + light subject fusion
- subject-level simple fixed-graph GNN without MIL

Recommended default for macro graphs
- large block such as 3-5 minutes
- aggregate short-window node features with mean
- aggregate short-window connectivity with mean

Metrics to compare
- balanced accuracy (primary)
- macro-F1
- calibration / Brier score
- variance across seeds
- runtime and memory cost

Interpretation
- If subject graphs win clearly, MIL becomes optional rather than central.
- If macro graphs win, they are probably reducing noisy segment variation while
  still retaining multiple observations per subject.
- If segment graphs win, the within-subject temporal diversity is important and
  MIL deserves later optimization.

Decision rule
- Keep the best 1 graph level, and at most one runner-up if the gap is small
  and the variance overlaps.
- Do not continue all three levels into later expensive blocks.

----------------------------------------------------------------------
BLOCK 2. MODEL-FAMILY COMPARISON
----------------------------------------------------------------------
Scientific question
- Before sophisticated graph learning, which representation family already works best?

Freeze
- Best graph level(s) from Block 1
- Same preprocessing
- Same fixed topology when a topology is needed
- Same edge-weight source
- Same connectivity metric and same band handling
- Same subject aggregation policy matched to graph level

Vary
- Model family only:
  1) node-only MLP
  2) connectivity-only MLP
  3) connectivity-only CNN
  4) dual-branch dense model
  5) simple fixed-graph GNN
  6) dual-branch graph model on fixed graph

Recommended settings
- Node-only MLP: baseline for node features
- Connectivity-only MLP: baseline for flattened connectivity
- Connectivity-only CNN: baseline for 19x19xB tensor
- DualBranchDenseModel with both MLP-fusion and CNN-fusion versions only if one is clearly useful
- SimpleFixedGraphGNN with shallow backbone, 2 layers
- DualBranchGraphModel only on the same fixed graph used by SimpleFixedGraphGNN

Metrics to compare
- balanced accuracy
- macro-F1
- PR-AUC macro OVR
- Brier score
- parameter count and runtime

Interpretation
- If dense dual-branch beats GNN, message passing is not yet justified.
- If simple fixed-graph GNN beats dense dual-branch, topology/message passing is adding value.
- If node-only already matches dual-branch, connectivity is not helping yet.
- If connectivity-only is weak but dual-branch helps, connectivity may be complementary rather than sufficient.

Decision rule
- Move forward with the best 2 model families only.
- One of them should usually be the strongest dense baseline.
- The second can be the best graph model if it is competitive.

----------------------------------------------------------------------
BLOCK 3. TOPOLOGY ABLATION
----------------------------------------------------------------------
Scientific question
- When graph structure matters, which source of topology helps most?

Freeze
- Best graph level
- Best graph-family candidate(s) from Block 2
- Same edge-weight strategy for all topology candidates
- Same connectivity metric / band source
- Same readout
- Same MIL strategy

Vary
- Topology source only:
  1) fixed topology
  2) connectivity-derived topology
  3) feature-induced topology

Recommended settings
A. Fixed topologies
- distance-based if available from your project setup
- user-defined electrode topology
- complete graph only as a stress-test reference, not as main finalist

B. Connectivity-derived topologies
- full
- top-k strongest neighbors
- MST / maximum spanning tree
- threshold only if threshold is fixed globally and not tuned too heavily

C. Feature-induced topologies
- cosine top-k
- pearson top-k
- optionally RBF top-k only if cosine/pearson are promising

Metrics to compare
- balanced accuracy
- macro-F1
- topology density statistics
- seed variance

Interpretation
- If fixed topology wins, learned biological priors are strong enough and dynamic topology may be unnecessary.
- If connectivity-derived topology wins, edge structure is carrying discriminative information.
- If feature-induced topology wins, node features are the main signal and topology is acting as a feature-similarity prior.

Decision rule
- Keep the best topology family and one best concrete topology instance.
- Do not keep many thresholds or many k values into later stages.

----------------------------------------------------------------------
BLOCK 4. EDGE-WEIGHT ABLATION
----------------------------------------------------------------------
Scientific question
- Given a topology mask, what should the edge values actually be?

Freeze
- Best graph level
- Best model family
- Best topology mask from Block 3
- Same connectivity metric source unless it is the tested weight source
- Same readout and aggregation

Vary
- Edge-weight source only:
  1) binary weights
  2) raw connectivity-valued weights
  3) normalized connectivity weights
  4) topology-source weights / similarity weights when applicable

Recommended settings
- binary mask
- raw connectivity weights from the chosen metric/band
- min-max or z-score normalized weights if implemented consistently
- topology-weight source only when topology construction already produced a meaningful weight matrix

Metrics to compare
- balanced accuracy
- macro-F1
- training stability
- calibration

Interpretation
- If binary works best, the presence of edges matters more than their magnitude.
- If weighted edges help, connection strength carries useful signal beyond sparsity pattern.
- If normalized beats raw, scale instability was hurting message passing.

Decision rule
- Keep one weight policy per surviving topology.

----------------------------------------------------------------------
BLOCK 5. CONNECTIVITY METRIC / BAND ABLATION
----------------------------------------------------------------------
Scientific question
- Which connectivity source is most useful once topology and weight policies are fixed?

Freeze
- Best graph level
- Best model family
- Best topology strategy
- Best edge-weight strategy
- Same readout and same aggregation

Vary
- Connectivity metric
- Band usage
- Single-band vs multiband usage

Recommended settings
Stage 5A: metric screen
- coherence
- pli
- wpli
- pearson or spearman as a non-phase reference

Stage 5B: band screen for the best 1-2 metrics
- delta
- theta
- alpha
- beta
- gamma

Stage 5C: representation style
- best single band
- all bands stacked
- small graph bank over a few selected metric-band candidates

Metrics to compare
- balanced accuracy
- macro-F1
- PR-AUC
- model variance across seeds

Interpretation
- If one band dominates, later topology-fusion should focus on that band first.
- If stacked multiband helps, dense/CNN branches or bank models are justified.
- If metrics disagree strongly across datasets later, that becomes a transferability question, not an early tuning question.

Decision rule
- Keep at most:
  - best single metric-band setting
  - best multiband setting
- Drop the rest before moving to richer aggregation or fusion experiments.

----------------------------------------------------------------------
BLOCK 6. MIL VS NON-MIL SUBJECT AGGREGATION
----------------------------------------------------------------------
Scientific question
- For the winning graph level, how much does subject aggregation strategy matter?

Freeze
- Best graph level(s)
- Best model family
- Best topology and edge-weight setup
- Best connectivity source from Block 5
- Same graph readout inside each instance encoder

Vary
- Subject aggregation only:
  1) none / identity (subject graph)
  2) mean MIL
  3) attention MIL
  4) gated attention MIL
  5) small subject fusion for macro graphs

Recommended settings
- For subject graphs: identity only; MIL is unnecessary
- For macro graphs: mean MIL vs subject_fusion
- For segment graphs: mean MIL vs attention MIL vs gated attention MIL

Metrics to compare
- balanced accuracy
- macro-F1
- Brier score
- sensitivity to bag size / number of windows

Interpretation
- If mean MIL is enough, later work should focus on better graph construction, not more elaborate bag pooling.
- If gated attention wins clearly on segment graphs, instance importance varies meaningfully across a subject.
- If subject graphs match or beat MIL-based segment graphs, simpler subject graphs are the preferred default.

Decision rule
- Keep exactly one subject aggregation strategy per surviving graph level.

----------------------------------------------------------------------
BLOCK 7. GRAPH-LEVEL READOUT / POOLING
----------------------------------------------------------------------
Scientific question
- Inside a graph encoder, how should node embeddings be pooled into one graph embedding?

Freeze
- Best graph level
- Best model family that still uses graph pooling
- Best topology, edge weights, connectivity source
- Best subject aggregation strategy

Vary
- Graph readout only:
  1) mean
  2) add/sum
  3) mean+max concat
  4) attention readout
  5) gated attention readout

Recommended settings
- Use mean as the simplest baseline
- Use mean_max_concat as the strongest non-learned baseline
- Use attention or gated_attention only on the finalists
- LSTM / hierarchical pooling should be deferred until the current readout set is exhausted

Metrics to compare
- balanced accuracy
- macro-F1
- calibration
- readout variance across seeds

Interpretation
- If mean_max_concat wins, the issue is richer summary statistics, not necessarily learned node importance.
- If attention/gated attention wins, some nodes consistently matter more than others.
- If differences are tiny, keep the simplest readout.

Decision rule
- Promote one readout only.

----------------------------------------------------------------------
BLOCK 8. FUSED GRAPH BANK / TOPOLOGY REFINEMENT
----------------------------------------------------------------------
Scientific question
- After strong fixed baselines are known, does constrained graph-bank fusion improve further?

Freeze
- Best graph level
- Best graph-family backbone
- Best node branch / best graph branch settings
- Best connectivity source shortlist
- Best MIL policy
- Best readout

Vary
- Graph-bank design only:
  1) non-learned fused bank from graph_construction.py
  2) FusedGraphBankGNN with static fusion
  3) FusedGraphBankGNN with summary-gated fusion
  4) DualBranchGraphModel with graph bank

Recommended bank contents
- Keep the bank small and curated
- Example bank:
  - fixed prior topology
  - connectivity top-k using best metric-band
  - connectivity MST using best metric-band
  - feature-induced cosine top-k

Recommended topology rules
- union
- vote
- intersection only as a strict sparsity stress test

Metrics to compare
- balanced accuracy
- macro-F1
- calibration
- fusion-weight stability across folds/seeds

Interpretation
- If static fusion helps, diversity of candidate topologies matters.
- If summary-gated fusion helps, subject-specific topology preference matters.
- If bank fusion does not beat the best fixed graph, topology learning should not be a project priority.

Decision rule
- Promote bank fusion only if it beats the best fixed-topology graph baseline by a consistent margin across seeds and folds.

----------------------------------------------------------------------
BLOCK 9. CROSS-DATASET CHECK (AHEAP -> CAUEEG)
----------------------------------------------------------------------
Scientific question
- Which conclusions survive when you move from the main dataset to the second dataset?

Freeze
- Only carry forward the finalists from Blocks 1-8
- No large retuning on CAUEEG

Vary
- Dataset only

Recommended settings
- best dense baseline
- best fixed-graph baseline
- best fusion-based model only if it already justified itself on AHEAP

Metrics to compare
- absolute performance on CAUEEG
- ranking stability of finalists
- failure modes relative to AHEAP

Interpretation
- If dense baselines transfer but complex graph models do not, the graph gains may be dataset-specific.
- If the same graph level and same model family remain strong, that is much better evidence for a reusable pipeline.

Decision rule
- Final project defaults should be chosen from the models that are both strong on AHEAP and reasonably stable on CAUEEG.

======================================================================
4. RECOMMENDED EXECUTION ORDER
======================================================================

Recommended order for actual running:
1. Block 0 on AHEAP
2. Block 1 on AHEAP
3. Block 2 on AHEAP
4. Block 3 on AHEAP using only surviving graph models
5. Block 4 on AHEAP
6. Block 5 on AHEAP
7. Block 6 on AHEAP
8. Block 7 on AHEAP
9. Block 8 on AHEAP only if fixed baselines are already solid
10. Block 9 on CAUEEG using only finalists

======================================================================
5. MINIMAL FIRST VERSION OF THE LADDER
======================================================================

If you want the smallest serious first pass, run only this reduced ladder:

Minimal Block A
- Block 0 sanity / reproducibility

Minimal Block B
- Block 1 graph-level comparison using:
  - node-only MLP
  - connectivity-only MLP
  - simple fixed-graph GNN

Minimal Block C
- Block 2 model-family comparison on the winning graph level using:
  - node-only MLP
  - connectivity-only CNN if multiband connectivity is available
  - dual-branch dense model
  - simple fixed-graph GNN

Minimal Block D
- Block 3 topology ablation for the winning graph model only:
  - fixed
  - connectivity top-k
  - connectivity MST
  - feature-induced cosine top-k

Minimal Block E
- Block 6 aggregation comparison only if the winner is segment/macro based:
  - mean MIL
  - gated attention MIL

Minimal Block F
- CAUEEG transfer check on the final 2-3 finalists

======================================================================
6. RESULTS THAT SHOULD BE USED AS GATES
======================================================================

Gate 1: from Block 0 to Block 1
- exact reproducibility established

Gate 2: from Block 1 to Block 2
- one graph level clearly survives

Gate 3: from Block 2 to Blocks 3-5
- at least one graph family is competitive with the best dense baseline
- if no graph family is competitive, shift effort to dense dual-branch and connectivity-source studies instead

Gate 4: from Blocks 3-5 to Blocks 6-7
- one concrete graph recipe is stable enough to justify finer tuning

Gate 5: from Blocks 6-7 to Block 8
- only if the best fixed-graph model is already strong and stable
- otherwise graph-bank fusion is too expensive and too unconstrained

Gate 6: from AHEAP to CAUEEG
- only finalists move forward

======================================================================
7. COMMON MISTAKES TO AVOID
======================================================================

1. Comparing segment, macro, and subject graphs while also changing MIL.
   That confounds graph level with aggregation.

2. Letting topology source and edge-weight source change together.
   A top-k graph with binary edges is not the same experiment as a top-k graph
   with coherence-valued edges.

3. Starting topology fusion too early.
   If fixed baselines are weak, graph-bank fusion will not be interpretable.

4. Using different split seeds across model families in the same block.
   That makes the comparison noisy and unfair.

5. Reporting instance-level accuracy as the main result.
   The scientific target is subject-level dementia classification.

6. Carrying too many candidates into later blocks.
   The point of the ladder is to prune aggressively.

7. Over-tuning CAUEEG before AHEAP conclusions are stable.
   That turns the project into two separate searches instead of one reusable pipeline.

8. Depending too early on registry.py as the orchestration layer.
   In the current codebase, direct imports are safer for the first ladder pass.

======================================================================
8. DEFAULT RECOMMENDATION FOR YOUR PROJECT
======================================================================

Based on the current project logic, the most sensible default hypothesis is:
- macro graph or subject graph will likely be more stable than raw segment graph
- dense dual-branch should be tested before assuming message passing helps
- simple fixed-graph GNN should be the first graph model benchmark
- topology and edge weights must be ablated separately
- fused graph bank should be treated as a late-stage refinement, not an early baseline

End of roadmap.
"""

# EXPERIMENT_LADDER_ROADMAP = __doc__
