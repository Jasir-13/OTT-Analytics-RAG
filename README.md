📊 OTT Analytics RAG System

A Retrieval-Augmented Generation (RAG) system designed to analyze OTT platform business metrics including revenue, churn, subscribers, engagement, watch time, and platform performance trends (2016–2026).

This project demonstrates:

Vector-based semantic retrieval

LLM-powered analytics reasoning

Automated evaluation & accuracy scoring

Hallucination detection

Schema validation for business metrics

🚀 Project Overview

The OTT Analytics RAG system enables natural language queries such as:

"What is the average monthly revenue per user from 2016 to 2026?"

"Which platform had the highest churn rate in 2022?"

"Compare mobile vs desktop usage trends."

"Which subscription plan grew the most?"

The system retrieves structured OTT analytics data and generates grounded responses using an LLM while enforcing schema validation to prevent hallucinations.

🏗 Architecture
User Query
    ↓
Query Parser
    ↓
Schema Validator
    ↓
Vector Retrieval (FAISS)
    ↓
Context Builder
    ↓
LLM (Deterministic Mode)
    ↓
Evaluation & Scoring Engine

Core Components
Component	Description
vector.py	Builds embeddings and FAISS vector index
main.py	Handles query processing, retrieval, response generation, and evaluation
dataset.csv	OTT analytics dataset (2016–2026)
Evaluation Module	Computes accuracy, precision, hallucination rate
📂 Project Structure
ott-analytics-rag/
│
├── dataset.csv
├── vector.py
├── main.py
├── requirements.txt
└── README.md

📊 Dataset Schema

The system expects structured OTT analytics data with fields such as:

Column	Description
year	Year (2016–2026)
platform	OTT platform name
region	Geographic region
subscribers	Total subscribers
revenue	Total revenue
avg_watch_time	Average watch time per user
churn_rate	Churn percentage
subscription_plan	Basic / Standard / Premium
device_type	Mobile / Desktop / TV
engagement_score	User engagement metric

⚠️ If a query references a metric not present in the schema, the system returns:

"Requested metric not available in dataset."

🛠 Installation

1️⃣ Clone the repository
git clone https://github.com/Jasir-13/OTT-Analytics-RAG.git
cd ott-analytics-rag

2️⃣ Install dependencies
pip install -r requirements.txt

▶️ Running the System
Build Vector Store
python vector.py

Run Evaluation
python main.py

📈 Evaluation Framework

The system evaluates:

Metric	Description
Retrieval Precision	% relevant records retrieved
Numeric Accuracy	Correct calculation of metrics
Schema Validation Score	Correct metric mapping
Hallucination Rate	% fabricated metrics
Confidence Calibration	Semantic similarity confidence
Scoring Scale
Score	Interpretation
90–100%	Production Ready
75–89%	Minor Improvements Needed
50–74%	Prototype Stage
< 50%	Requires Major Fixes
🧠 Model Configuration

Embedding Model: text-embedding-3-small

LLM: gpt-4o-mini

Temperature: 0 (deterministic output)

Retrieval: FAISS (L2 similarity)

🔍 Known Issues & Limitations

Requires structured analytics dataset (not movie metadata)

Performance depends on schema quality

Does not auto-correct malformed queries

Needs guardrails to prevent cross-domain retrieval

🛡 Hallucination Prevention Strategy

The system prevents incorrect answers by:

Schema validation before response generation

Confidence threshold enforcement

Deterministic LLM configuration

Rejecting unsupported metrics

Context-restricted prompting

🧪 Example Evaluation Prompts

What is the average monthly revenue per user from 2016–2026?

Which platform had the highest subscribers in 2022?

How has churn changed from 2018–2024?

Which subscription plan grew the most?

Compare mobile vs desktop usage trends.

📊 Professional Use Cases

OTT Business Intelligence

Revenue forecasting dashboards

Subscriber analytics

Churn prediction analysis

Academic research in AI + Media Tech

RAG evaluation benchmarking

🔧 Future Improvements

Add structured query parsing (SQL layer)

Add automated confusion matrix reporting

Deploy via FastAPI

Add Streamlit dashboard

Add Pinecone cloud vector DB

Add multi-agent analytics reasoning

Add explainability layer for enterprise audit

📄 License

MIT License

👨‍💻 Author

OTT Analytics RAG Project
Built for advanced evaluation of retrieval-based analytics systems.

⭐ Final Note

This project demonstrates both:

RAG capability for business analytics

The importance of schema grounding to prevent hallucinations

It is designed as a benchmark framework for evaluating analytics-grade LLM systems.
