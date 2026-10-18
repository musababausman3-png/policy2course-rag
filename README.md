# Policy2Course RAG
### A Retrieval-Augmented Generation System for Policy-Grounded Employee Training

## Overview

Policy2Course RAG is an AI application designed to transform company policy documents into structured employee training materials using Retrieval-Augmented Generation (RAG).

Instead of relying entirely on a language model's general knowledge, the system retrieves relevant information from a supplied policy document and uses that context to generate training content. The goal is to make policy-based learning more accessible, structured, and grounded in source material.

This project explores how retrieval systems and large language models can support practical knowledge transformation and workplace learning.

## Problem Statement

Organisations often communicate important rules, procedures, and compliance requirements through lengthy policy documents. Employees may find it difficult to identify the most relevant information and translate it into practical learning activities.

Policy2Course RAG explores an AI-assisted approach to this problem by converting policy content into structured training materials while using retrieved source information to guide generation.

## Objectives

- Develop a document-grounded AI pipeline for generating training materials.
- Retrieve relevant policy content before generating responses.
- Explore the use of embeddings and vector-based retrieval in a practical RAG application.
- Produce structured training outputs from source documents.
- Investigate ways to reduce unsupported or irrelevant generated content by grounding generation in retrieved evidence.

## How It Works

The system follows a retrieval-augmented generation workflow:

1. **Document input:** A policy document is supplied as source material.
2. **Text processing:** The document is processed into content suitable for retrieval.
3. **Embedding and retrieval:** Relevant text is represented and retrieved using the configured retrieval components.
4. **Context construction:** Retrieved passages are assembled into context for the language model.
5. **Training-content generation:** The model generates training material based on the retrieved context.
6. **Output:** The generated training content can be inspected in the application's output.

## Technology Stack

The project uses Python and a retrieval-augmented generation architecture. Its specific dependencies and model configuration are defined in `requirements.txt` and the application source code.

Core concepts include:

- Retrieval-Augmented Generation (RAG)
- Text embeddings and semantic retrieval
- Large language models (LLMs)
- Document processing and prompt construction
- Structured AI-generated outputs

## Repository Structure

```text
policy2course-rag/
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── sample_policy.txt
├── generated_course.json
├── .env.example
├── .gitignore
└── README.md
```

- `app.py` — application entry point.
- `rag_pipeline.py` — retrieval and generation pipeline.
- `requirements.txt` — Python dependencies.
- `sample_policy.txt` — example policy input.
- `generated_course.json` — example generated training output.
- `.env.example` — example environment configuration.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/musababausman3-png/policy2course-rag.git
cd policy2course-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux or macOS**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the environment

Review `.env.example` and configure the environment variables required by the application. Supply any required API credentials through environment variables or a local environment file that is excluded from version control.

Never commit API keys, passwords, or other secrets to the repository.

### 5. Run the application

Check the instructions and entry point in `app.py` to confirm the correct command for your current implementation.

## Example Input and Output

The repository includes `sample_policy.txt` as an example source document and `generated_course.json` as an example output.

The intended workflow is to use policy information as grounding context for the generation of structured employee training material. The example files should be inspected together to assess how well the output reflects the source policy.

## Evaluation and Limitations

RAG systems can still produce incomplete, irrelevant, or unsupported statements. Retrieval quality, document coverage, prompt design, and language-model behaviour all affect the output.

A rigorous evaluation should examine:

- Whether generated statements are supported by the source document.
- Whether important policy requirements are omitted.
- Whether retrieved passages are relevant to the generated content.
- Whether the generated training material is clear and logically structured.
- Whether results remain consistent across different policy documents.

Any reported evaluation results should specify the test cases, evaluation method, and limitations. This README does not claim independent validation beyond the tests documented in the repository.

## Potential Applications

- Employee onboarding and workplace training.
- Policy awareness and internal learning.
- Compliance education.
- Converting lengthy policy documents into structured learning materials.

The system is an experimental AI-assisted tool. Generated content should be reviewed by a qualified person before being used for formal compliance training.

## Future Improvements

Potential future work includes:

- Adding document upload and support for multiple file formats.
- Providing source citations for generated training sections.
- Developing a test set for measuring retrieval relevance and factual grounding.
- Comparing retrieval and generation configurations.
- Adding automated evaluation and human review.
- Improving error handling, reproducibility, and deployment documentation.

These are proposed improvements, not claims that the features are already implemented.

## Research and Learning Goals

This project provides a practical setting for exploring retrieval-augmented generation, semantic search, prompt design, and the evaluation of AI-generated content. It reflects an interest in developing AI systems that connect language-model capabilities with supplied evidence and practical user needs.

## Author

**Musa Baba Usman**

GitHub: https://github.com/musababausman3-png

## Disclaimer

This project is intended for learning and experimentation. 
