# Learning plan — first week

## Architecture direction

We will grow the application in small layers:

```text
CLI or web UI
    ↓
application service
    ↓
document processing and retrieval
    ↓
model provider
```

The future document flow will be:

```text
PDF upload → text extraction → chunks → embeddings/index → retrieval
    → grounded prompt → answer + source citations
```

The first milestone contains only the model-provider boundary. Keeping that boundary small will make it easier to test and later replace direct question answering with retrieval-augmented generation (RAG).

## First-week plan

1. Python foundations: functions, modules, exceptions, type hints, virtual environments, and reading tracebacks.
2. API foundations: HTTP request/response concepts, environment variables, the OpenAI SDK, and prompt/input design.
3. Testing: assertions, fakes/mocks, testing failure cases, and why tests should not depend on a live network service.
4. Git: make a checkpoint, inspect a diff, and write a useful commit message.
5. Project milestone: explain the data flow from the CLI to `ask_question()` and the model response.
6. Review: run tests and linting, write a short reflection, and explain the test back from memory.

## Book study

Use *The Developer's Guide to AI* as the main practical guide. Look for sections with topics such as:

- AI application architecture and model/API calls.
- Prompting and message roles.
- Structured outputs, when we add a typed response contract.
- Embeddings, vector databases, and RAG, before PDF search.
- Tool calling, later when the assistant needs external actions.

Use Géron's *Hands-On Machine Learning with Scikit-Learn and PyTorch* selectively. For this week, postpone most model-training chapters. Study only the introductory material that explains what a model does, inputs/outputs, and the difference between using a hosted model API and training a model yourself. Neural networks, attention, and transformers become more useful when we need to understand RAG limitations and model behavior; they are not prerequisites for this first API call.

## Questions for the learner

- What is the difference between the question passed into `ask_question()` and the answer returned from it?
- Why does the test pass a fake client?
- Where does the API key come from, and why should it not be hard-coded?
- What part of the future PDF workflow is not implemented yet?
