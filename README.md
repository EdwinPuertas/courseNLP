<div align="center">

# Natural Language Processing
### Faculty of Engineering — Computer Science Program

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)
![License](https://img.shields.io/github/license/EdwinPuertas/courseNLP)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![ACM Ethics](https://img.shields.io/badge/Ethics-ACM%20Code-red)

**Professor:** Edwin Puertas, Ph.D. · `epuerta@utb.edu.co`
**Institution:** Universidad Tecnológica de Bolívar
**Repository:** [github.com/EdwinPuertas/courseNLP](https://github.com/EdwinPuertas/courseNLP)

</div>

---

## Table of Contents

1. [Course Overview](#course-overview)
2. [Learning Outcomes](#learning-outcomes)
3. [Course Structure](#course-structure)
4. [Repository Organization](#repository-organization)
5. [Modules](#modules)
   - [Unit 1 — Structural Foundations of Language](#-unit-1--structural-foundations-of-language)
   - [Unit 2 — Statistical Patterns in Language](#-unit-2--statistical-patterns-in-language)
   - [Unit 3 — Distributed Meaning Representations](#-unit-3--distributed-meaning-representations)
   - [Unit 4 — Sequential Understanding Models](#-unit-4--sequential-understanding-models)
   - [Unit 5 — Attention-Driven Language Intelligence](#-unit-5--attention-driven-language-intelligence)
   - [Unit 6 — Final Project](#-unit-6--final-project)
   - [Supplementary — Prompt Engineering](#-supplementary--prompt-engineering)
   - [Supplementary — Advanced Prompting Techniques](#-supplementary--advanced-prompting-techniques)
6. [Setup & Installation](#setup--installation)
7. [Academic Policy](#academic-policy)
8. [Bibliography & Resources](#bibliography--resources)
9. [Contact](#contact)

---

## Course Overview

This course provides a rigorous introduction to the theory and practice of **Natural Language Processing (NLP)**, with emphasis on deep learning approaches to language understanding. Students explore both classical NLP methods and state-of-the-art neural architectures — including word embeddings, recurrent networks, attention mechanisms, and Transformers — learning how intelligent systems process text, extract meaning, and generate predictions.

The course follows a **theoretical-practical format**:

| Session type | Hours / week |
|---|---|
| Theoretical instruction | 1 h |
| Supervised laboratory | 2 h |
| Independent study (minimum) | **6 h** |

Lectures introduce core concepts and current research directions, while laboratory sessions guide students through hands-on implementation of NLP models using modern deep learning tools. The professor is available during office hours for individual consultation. Periodic unannounced assessments — reading quizzes, workshops, and project reviews — are conducted throughout the semester.

Beyond technical competence, the course aims to develop **creativity**, **teamwork**, and **problem-solving** skills, preparing students to design, evaluate, and responsibly deploy intelligent language systems.

> **Foundational Reference:** This course is aligned with and complemented by the publicly available lecture materials of **Stanford CS124: From Languages to Information** (Prof. Dan Jurafsky), available at [web.stanford.edu/class/cs124](https://web.stanford.edu/class/cs124/).

---

## Learning Outcomes

Upon successful completion of this course, students will be able to:

| # | Outcome |
|---|---|
| 1 | **Understand** the core concepts and techniques of NLP — including language modeling, sentiment analysis, vector semantics, semantic representation, and neural network fundamentals applied to language |
| 2 | **Apply** NLP methods to real-world tasks such as text classification, sequence labeling, machine translation, chatbot development, and other practical language-based applications across organizational and research contexts |
| 3 | **Develop, evaluate, and optimize** supervised and unsupervised NLP models using Python, NLTK, TensorFlow, or equivalent frameworks — including metric selection, performance analysis, and model refinement |
| 4 | **Integrate** ethical, research, and collaborative competencies into NLP projects — demonstrating the ability to identify and solve complex NLP problems, review academic literature, design experiments, communicate results, and apply ethical principles |

---

## Course Structure

The curriculum is organized into six thematic units that progress from linguistic foundations to state-of-the-art transformer models and applied projects:

```
Unit 1  │  Structural Foundations of Language
         │  History of NLP · Linguistic levels · Development environment
         │  ─── Stanford CS124 Reference: Week 1–2 (Tokenization & Regex)

Unit 2  │  Statistical Patterns in Language
         │  N-gram models · Naive Bayes · Text classification · TF-IDF
         │  ─── Stanford CS124 Reference: Week 2–3 (LM, Logistic Regression)

Unit 3  │  Distributed Meaning Representations
         │  Vector semantics · Word embeddings · Edit distance · Neural LMs
         │  ─── Stanford CS124 Reference: Week 4–5 (IR, Embeddings)

Unit 4  │  Sequential Understanding Models
         │  LLMs · Transformer architecture · Pre-training & fine-tuning
         │  ─── Stanford CS124 Reference: Week 6 (Neural Networks, LLMs, Transformers)

Unit 5  │  Attention-Driven Language Intelligence
         │  BERT · NER · Sequence labeling · Business NLP applications
         │  ─── Stanford CS124 Reference: Week 7 (Transformers & Attention)

Unit 6  │  Final Project
         │  End-to-end NLP system design, evaluation, and presentation
```

> **Support Material:** *Tractatus Logico-Philosophicus* — Ludwig Wittgenstein (1922).
> A philosophical inquiry into the relationship between language, meaning, and the limits of expression — providing foundational context for the computational treatment of language.

---

## Repository Organization

```
courseNLP/
│
├── examples/                        # Core NLP implementations (Units 1–5)
│   ├── data/
│   ├── logic/
│   ├── word_tokenization.ipynb
│   ├── bow_model.ipynb
│   ├── tfidf_model.ipynb
│   ├── tfidf_naive_bayes_imdb.ipynb
│   ├── tass_bow_softmax_workshop.ipynb
│   ├── pycaret_text_classification.ipynb
│   ├── classification_evaluation.ipynb
│   ├── ner_introduction.ipynb
│   ├── ner_business.ipynb
│   ├── training_ner.ipynb
│   ├── custom_ner_bert.ipynb
│   ├── bert_classifier.ipynb
│   ├── spanish_review_classifier.ipynb
│   ├── customer_service_chatbot.ipynb
│   ├── metrics_llm_business.ipynb
│   └── text_generation_business.ipynb
│
├── prompt_engineering/              # Supplementary – Prompt design for LLMs
│   ├── 00_intro.ipynb
│   ├── 01_prompt_structure.ipynb
│   ├── 02_clear_and_direct.ipynb
│   ├── 03_assigning_roles.ipynb
│   ├── 04_separating_data_instructions.ipynb
│   ├── 05_formatting_output_and_speaking_claude.ipynb
│   ├── 06_precognition_thinking.ipynb
│   ├── 07_fewshot_prompting.ipynb
│   ├── 08_avoiding_hallucinations.ipynb
│   └── 09_complex_prompts_from_scratch.ipynb
│
├── prompting_techniques/            # Supplementary – Advanced prompting strategies
│   ├── chain_of_thought.ipynb
│   ├── few_shot_prompting.ipynb
│   ├── image_segmentation.ipynb
│   ├── program_aided_language_pal.ipynb
│   ├── generated_knowledge_prompting.ipynb
│   ├── react_prompting.ipynb
│   ├── retrieval_augmented_generation.ipynb
│   └── zero_shot_cot_prompting.ipynb
│
├── LICENSE
└── README.md
```

---

## Modules

### 📖 Unit 1 — Structural Foundations of Language

> *[`examples/`](./examples/)*

**Objectives**
- Understand the historical evolution of NLP and its relationship to linguistics and AI
- Identify the main NLP tasks and the computational processing pipeline
- Set up a reproducible Python development environment

**Stanford CS124 Alignment:** Week 1–2 — *Tokenization, Regular Expressions & Edit Distance*
([Lecture Slides PDF](https://www.stanford.edu/class/cs124/lec/tokens_jan26.pdf) · [J&M Ch. 2: Words and Tokens](https://web.stanford.edu/~jurafsky/slp3/))

| Notebook | Topic |
|---|---|
| [`word_tokenization.ipynb`](./examples/word_tokenization.ipynb) | Text Tokenization & Preprocessing Pipeline |
| [`bow_model.ipynb`](./examples/bow_model.ipynb) | Bag-of-Words Text Representation |

---

### 📊 Unit 2 — Statistical Patterns in Language

> *[`examples/`](./examples/)*

**Objectives**
- Model language using statistical and probabilistic approaches
- Build and evaluate text classifiers using Naive Bayes and logistic regression
- Apply TF-IDF weighting for document representation and information retrieval

**Stanford CS124 Alignment:** Week 2–3 — *N-gram Language Modeling & Text Classification*
([N-gram Slides PDF](https://www.stanford.edu/class/cs124/lec/lm_jan25.pdf) · [Classification Slides PDF](https://www.stanford.edu/class/cs124/lec/logreg26jan.pdf) · [J&M Ch. 3](https://web.stanford.edu/~jurafsky/slp3/) · [J&M Ch. 4](https://web.stanford.edu/~jurafsky/slp3/))

| Notebook | Topic |
|---|---|
| [`tfidf_model.ipynb`](./examples/tfidf_model.ipynb) | TF-IDF Vectorization & Document Similarity |
| [`tfidf_naive_bayes_imdb.ipynb`](./examples/tfidf_naive_bayes_imdb.ipynb) | Naive Bayes Text Classification — IMDB |
| [`tass_bow_softmax_workshop.ipynb`](./examples/tass_bow_softmax_workshop.ipynb) | Sentiment Analysis — BoW + Softmax (TASS) |
| [`pycaret_text_classification.ipynb`](./examples/pycaret_text_classification.ipynb) | AutoML Text Classification with PyCaret |
| [`classification_evaluation.ipynb`](./examples/classification_evaluation.ipynb) | Classification Evaluation Metrics |

---

### 🔢 Unit 3 — Distributed Meaning Representations

> *[`examples/`](./examples/)*

**Objectives**
- Understand vector semantics and the distributional hypothesis
- Implement and interpret word embeddings and dense representations
- Apply similarity and distance metrics to NLP tasks

**Stanford CS124 Alignment:** Week 4–5 — *Information Retrieval & Vector Semantics*
([IR Slides PDF](https://www.stanford.edu/class/cs124/lec/ir_jan17_2026.pdf) · [Embeddings Slides PDF](https://www.stanford.edu/class/cs124/lec/vector26jan.pdf) · [J&M Ch. 5](https://web.stanford.edu/~jurafsky/slp3/) · [J&M Ch. 11](https://web.stanford.edu/~jurafsky/slp3/))

| Notebook | Topic |
|---|---|
| [`bow_model.ipynb`](./examples/bow_model.ipynb) | Co-occurrence Vectors & Sparse Representations |
| [`tfidf_model.ipynb`](./examples/tfidf_model.ipynb) | Dense vs. Sparse Retrieval Models |

---

### 🔁 Unit 4 — Sequential Understanding Models

> *[`examples/`](./examples/)*

**Objectives**
- Understand the architecture and pre-training objectives of Large Language Models
- Apply Transformer-based models to NLP downstream tasks
- Design and evaluate LLM-powered systems for real-world applications

**Stanford CS124 Alignment:** Week 6 — *Neural Networks, LLMs & Transformers*
([LLM Slides PDF](https://www.stanford.edu/class/cs124/lec/llm_cs124_26.pdf) · [Transformer Slides PDF](https://www.stanford.edu/class/cs124/lec/transformer_cs124_26.pdf) · [J&M Ch. 6](https://web.stanford.edu/~jurafsky/slp3/) · [J&M Ch. 7](https://web.stanford.edu/~jurafsky/slp3/) · [J&M Ch. 8](https://web.stanford.edu/~jurafsky/slp3/))

| Notebook | Topic |
|---|---|
| [`text_generation_business.ipynb`](./examples/text_generation_business.ipynb) | Text Generation with LLMs — Business Cases |
| [`metrics_llm_business.ipynb`](./examples/metrics_llm_business.ipynb) | LLM Evaluation Metrics in Applied Contexts |
| [`customer_service_chatbot.ipynb`](./examples/customer_service_chatbot.ipynb) | Conversational Agent Design with LLMs |

---

### 🤖 Unit 5 — Attention-Driven Language Intelligence

> *[`examples/`](./examples/)*

**Objectives**
- Fine-tune BERT-based models for sequence classification and token labeling
- Build Named Entity Recognition (NER) systems using transformer architectures
- Evaluate and compare model performance across NLP benchmarks

**Stanford CS124 Alignment:** Week 7 — *Transformers & Attention Mechanisms*
([Slides PDF](https://www.stanford.edu/class/cs124/lec/transformer_cs124_26.pdf) · [J&M Ch. 8](https://web.stanford.edu/~jurafsky/slp3/))

| Notebook | Topic |
|---|---|
| [`ner_introduction.ipynb`](./examples/ner_introduction.ipynb) | Introduction to Named Entity Recognition |
| [`ner_business.ipynb`](./examples/ner_business.ipynb) | NER for Business Information Extraction |
| [`training_ner.ipynb`](./examples/training_ner.ipynb) | Training Custom NER Models |
| [`custom_ner_bert.ipynb`](./examples/custom_ner_bert.ipynb) | BERT-based Custom NER Pipeline |
| [`bert_classifier.ipynb`](./examples/bert_classifier.ipynb) | Text Classification with BERT |
| [`spanish_review_classifier.ipynb`](./examples/spanish_review_classifier.ipynb) | Spanish Review Classifier — Fine-tuned BERT |

---

### 🎓 Unit 6 — Final Project

Students design, implement, and evaluate an end-to-end NLP system addressing a real-world problem. Projects integrate techniques from across all course units and are assessed on technical rigor, innovation, reproducibility, documentation quality, and oral presentation.

**Deliverables**
- Technical report (IEEE format)
- Reproducible codebase (Jupyter Notebook + GitHub repository)
- Oral presentation and live system demonstration

**Stanford CS124 Reference:** PA7 — Agent Project
([Lab Guidelines](https://github.com/cs124/labs/blob/main/Lab4_Agent.md))

---

### 🗣️ Supplementary — Prompt Engineering

> *[`prompt_engineering/`](./prompt_engineering/)*

A structured sequence on systematic prompt design for large language models — a critical applied skill for modern NLP practitioners.

| Notebook | Topic |
|---|---|
| [`00_intro.ipynb`](./prompt_engineering/00_intro.ipynb) | Introduction to Prompt Engineering |
| [`01_prompt_structure.ipynb`](./prompt_engineering/01_prompt_structure.ipynb) | Structural Design of Prompts |
| [`02_clear_and_direct.ipynb`](./prompt_engineering/02_clear_and_direct.ipynb) | Clarity and Directness |
| [`03_assigning_roles.ipynb`](./prompt_engineering/03_assigning_roles.ipynb) | Role Assignment and Persona Design |
| [`04_separating_data_instructions.ipynb`](./prompt_engineering/04_separating_data_instructions.ipynb) | Separating Data from Instructions |
| [`05_formatting_output_&_speaking_claude.ipynb`](./prompt_engineering/05_formatting_output_&_speaking_claude.ipynb) | Output Formatting Strategies |
| [`06_precognition_thinking.ipynb`](./prompt_engineering/06_precognition_thinking.ipynb) | Extended Reasoning & Chain-of-Thought |
| [`07_fewshot_prompting.ipynb`](./prompt_engineering/07_fewshot_prompting.ipynb) | Few-Shot Prompting |
| [`08_avoiding_hallucinations.ipynb`](./prompt_engineering/08_avoiding_hallucinations.ipynb) | Hallucination Mitigation Techniques |
| [`09_Complex_Prompts_from_Scratch.ipynb`](./prompt_engineering/09_Complex_Prompts_from_Scratch.ipynb) | Building Complex Prompt Pipelines |

---

### ⚡ Supplementary — Advanced Prompting Techniques

> *[`prompting_techniques/`](./prompting_techniques/)*

Advanced reasoning and retrieval strategies that extend LLM capabilities beyond standard prompting.

| Notebook | Topic |
|---|---|
| [`chain_of_thought.ipynb`](./prompting_techniques/chain_of_thought.ipynb) | Chain-of-Thought Prompting |
| [`zero_shot_cot_prompting.ipynb`](./prompting_techniques/zero_shot_cot_prompting.ipynb) | Zero-Shot Chain-of-Thought |
| [`few_shot_prompting.ipynb`](./prompting_techniques/few_shot_prompting.ipynb) | Few-Shot Prompting Strategies |
| [`program_aided_language_pal.ipynb`](./prompting_techniques/program_aided_language_pal.ipynb) | Program-Aided Language (PAL) |
| [`react_prompting.ipynb`](./prompting_techniques/react_prompting.ipynb) | ReAct: Reasoning + Acting |
| [`retrieval_augmented_generation.ipynb`](./prompting_techniques/retrieval_augmented_generation.ipynb) | Retrieval-Augmented Generation (RAG) |
| [`generated_knowledge_prompting.ipynb`](./prompting_techniques/generated_knowledge_prompting.ipynb) | Generated Knowledge Prompting |
| [`image_segmentation.ipynb`](./prompting_techniques/image_segmentation.ipynb) | Multimodal Prompting — Image Segmentation |

---

## Setup & Installation

### Prerequisites

- **Python** 3.8 or higher
- **pip** or **conda** package manager
- **Jupyter Notebook** or **JupyterLab**

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/EdwinPuertas/courseNLP.git
cd courseNLP

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

### Core Dependencies

| Category | Packages |
|---|---|
| Data Science & ML | `numpy`, `scipy`, `pandas`, `scikit-learn` |
| Deep Learning & Transformers | `torch`, `transformers`, `tensorflow` / `keras` |
| NLP Tools | `nltk`, `spacy`, `gensim` |
| AutoML | `pycaret` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Data Utilities | `datasets`, `tqdm`, `requests` |
| AI APIs | `anthropic` |

---

## Academic Policy

### Attendance & Participation

Punctual attendance is mandatory for all sessions. Students are expected to complete assigned readings and workshop submissions on schedule. The professor will conduct **periodic, unannounced assessments** — including reading quizzes, workshop reviews, and project evaluations — throughout the semester.

### Independent Study Requirement

A minimum of **six (6) hours per week** of independent study is required, consistent with the standard of two study hours per contact hour. This time should be devoted to reinforcing class material, completing programming assignments, and advancing the course project.

### Academic Integrity

All work submitted must be the student's original contribution. Any use of external sources, libraries, models, or generated code must be explicitly acknowledged with proper attribution. This course adheres to the **ACM Code of Ethics and Professional Conduct**:

> [https://www.acm.org/code-of-ethics](https://www.acm.org/code-of-ethics)

Violations of academic integrity will be addressed in accordance with institutional regulations and may result in course failure or formal disciplinary proceedings.

---

## Bibliography & Resources

### Primary Textbooks

1. **Jurafsky, D., & Martin, J. H.** (2025). *Speech and Language Processing* (3rd ed., draft). Stanford University.
   [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/) *(Open access)*

2. **Beysolow II, T.** (2018). *Applied Natural Language Processing with Python: Implementing Machine Learning and Deep Learning Algorithms for Natural Language Processing*. Apress.

3. **Vajjala, S., Majumder, B., Gupta, A., & Surana, H.** (2020). *Practical Natural Language Processing: A Comprehensive Guide to Building Real-World NLP Systems*. O'Reilly Media.

4. **Srinivasa-Desikan, B.** (2018). *Natural Language Processing and Computational Linguistics: A Practical Guide to Text Analysis with Python, Gensim, spaCy, and Keras*. Packt Publishing.

### Supplementary Lecture Resources

5. **Jurafsky, D.** (2026). *CS124: From Languages to Information — Course Materials*. Stanford University.
   [https://web.stanford.edu/class/cs124/](https://web.stanford.edu/class/cs124/)

   Selected slide decks cited in this course:

   | Topic | Slides |
   |---|---|
   | Tokenization & Edit Distance | [PDF](https://www.stanford.edu/class/cs124/lec/tokens_jan26.pdf) |
   | N-gram Language Modeling | [PDF](https://www.stanford.edu/class/cs124/lec/lm_jan25.pdf) |
   | Logistic Regression & Text Classification | [PDF](https://www.stanford.edu/class/cs124/lec/logreg26jan.pdf) |
   | Information Retrieval & RAG | [PDF](https://www.stanford.edu/class/cs124/lec/ir_jan17_2026.pdf) |
   | Vector Semantics & Embeddings | [PDF](https://www.stanford.edu/class/cs124/lec/vector26jan.pdf) |
   | Large Language Models | [PDF](https://www.stanford.edu/class/cs124/lec/llm_cs124_26.pdf) |
   | Transformer Architecture | [PDF](https://www.stanford.edu/class/cs124/lec/transformer_cs124_26.pdf) |

### Philosophical Foundation

6. **Wittgenstein, L.** (1922). *Tractatus Logico-Philosophicus*. Kegan Paul, Trench, Trübner & Co.
   *(Foundational inquiry into the nature of language, meaning, and the limits of representation)*

### Professional Standards

7. Association for Computing Machinery. (2018). *ACM Code of Ethics and Professional Conduct*.
   [https://www.acm.org/code-of-ethics](https://www.acm.org/code-of-ethics)

---

## Contact

| | |
|---|---|
| **Professor** | Edwin Puertas, Ph.D. |
| **Email** | epuerta@utb.edu.co |
| **Institution** | Universidad Tecnológica de Bolívar |
| **Office Hours** | Available by appointment — contact via institutional channels |

---

<div align="center">

*This repository is maintained for educational purposes. All materials are intended solely for enrolled students of the Natural Language Processing course.*

[![GitHub](https://img.shields.io/badge/GitHub-EdwinPuertas-181717?logo=github&logoColor=white)](https://github.com/EdwinPuertas)
[![UTB](https://img.shields.io/badge/Universidad-Tecnológica%20de%20Bolívar-003DA5)](https://www.utb.edu.co)
[![Stanford CS124](https://img.shields.io/badge/Reference-Stanford%20CS124-8C1515)](https://web.stanford.edu/class/cs124/)

</div>
