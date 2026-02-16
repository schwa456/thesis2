# 📖 Text-to-SQL with Uncertainty-Aware Graph Routing & Adaptive Agents

This repository contains the official implementation of our master's thesis research: **"Uncertainty-Aware Schema Routing and Adaptive Multi-Agent Consensus for Robust Text-to-SQL."** This project tackles the critical challenges of structural hallucination and unanswerable queries in cross-domain Text-to-SQL benchmarks (e.g., BIRD, Spider). By synergizing Heterogeneous Graph Neural Networks (GAT), Prize-Collecting Steiner Tree (PCST) optimization, and an Adaptive Multi-Agent Workflow, this system drastically reduces latency while safely rejecting ambiguous or impossible queries.

## ✨ Key Features & Novelties
### 1. Edge-as-Node Graph Transformation (Offline)
- Database foreign keys are "verbalized" via LLMs and promoted to independent fk_node entities.
- A Heterogeneous GAT learns the topological structure, compressing DB schema contexts into dense vectors.
### 2. Symmetric Dual-Tower Alignment (Joint Space)
- Maps natural language query tokens and graph nodes into a shared latent space.
- Employs **MaxSim (Maximum Similarity)** operations with a strict threshold ($\tau$) to retrieve high-precision Initial Seed Nodes, eliminating stopword noise.
### 3. Semantic-Aware PCST Routing (Online)
- Extracts the optimal sub-graph using the GW-based PCST algorithm.
- **Dynamic Edge Cost: $c(e) = c_{base} - \alpha \cdot \text{sim}(Q, e)$**. The algorithm dynamically discounts the cost of traversing edges that are semantically relevant to the user's query.
### 4. Adaptive Multi-Agent Workflow with Jaccard Uncertainty
- **Parallel Execution**: A Semantic Analyst and a Structural Admin evaluate the sub-graph simultaneously via asyncio for near-zero extra latency.
- **Mathematical Uncertainty**: Calculates disagreement using Jaccard Distance ($U = 1 - \frac{|A \cap B|}{|A \cup B|}$).
- **Conditional Skeptic**: If $U > Threshold$, a "Conservative Skeptic" agent intervenes to explicitly output Unanswerable, preventing forced hallucinations.

# 📁 Repository Structure
```PlainText
thesis2/
├── data/                      # Raw datasets (.sqlite) and processed indices (.index, .pkl)
├── models/                    # PyTorch Neural Network architectures
│   ├── gat_network.py         # HeteroGAT for structural node/edge embeddings
│   ├── alignment_layer.py     # Dual-Tower Contrastive Learning (InfoNCE)
│   └── plm_encoder.py         # Token-level feature extraction (SentenceTransformers)
├── offline_indexing/          # Pipeline for converting DB to Vector Space
│   ├── schema_parser.py       # SQLite metadata extraction
│   ├── llm_verbalizer.py      # Edge description generation (via vLLM/OpenAI API)
│   ├── graph_builder.py       # PyG HeteroData construction
│   └── build_index.py         # FAISS & KV Store integration
├── online_inference/          # Real-time inference pipeline
│   ├── query_processor.py     # POS Tagging & Lexical filtering (SpaCy)
│   ├── retriever.py           # Threshold-based FAISS search & MaxSim scoring
│   ├── pcst_router.py         # Dynamic edge cost & Sub-graph extraction (pcst_fast)
│   └── agent_workflow.py      # Async Multi-Agent consensus & Uncertainty scoring
├── utils/                     # Centralized prompts, metrics, and loggers
│   ├── prompts.py
│   ├── logger.py
│   └── metrics.py
├── run_offline.py             # Script to build the offline schema index
└── run_online.py              # Script to run end-to-end inference & agent evaluation
```

# 🚀 Quick Start
## 1. Prerequisites
We recommend using a virtual environment (e.g., Conda).
```Bash
# Install PyTorch (Compute Platform dependent)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch_geometric

# Install other dependencies
pip install faiss-cpu pcst_fast sentence-transformers spacy openai numpy

# Download SpaCy English model
python -m spacy download en_core_web_sm
```

## 2. LLM Engine Setup (vLLM)
For edge verbalization and agentic workflow, we utilize local open-source models (e.g., Llama-3) via vLLM for maximum throughput.
```Bash 
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000
```

## 3. Run the Pipeline
### Step A: Offline Indexing (Run once per new database)
This parses the SQLite DB, verbalizes edges, trains/runs the GAT, and caches everything into FAISS.
```Bash 
python run_offline.py --db_path ./data/raw/bird_dev.sqlite
```

### Step B: Online Inference (Real-time query resolution)
This takes a natural language query, retrieves seed nodes, routes via PCST, and triggers the adaptive agents.
```Bash 
python run_online.py --query "List the names of employees in the IT department who have a salary greater than 50000."
```

# 📊 Evaluation & Metrics
Our system is evaluated on two primary axes:

**1. Execution Accuracy (EX) & Valid Efficiency Score (VES)**: Standard Text-to-SQL metrics.

**2. Rejection Accuracy & Over-Rejection Rate**: Measured via our custom Confusion Matrix implementation in utils/metrics.py to prove the robustness of our Uncertainty-Aware Skeptic Agent against unanswerable queries.

# 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.