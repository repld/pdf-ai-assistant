# Deploying the Streamlit UI

Streamlit Community Cloud deploys from a published GitHub repository. The local project currently has no GitHub remote, so deployment cannot start until the repository is created and pushed.

## 1. Create and publish the GitHub repository

Create a new empty repository on GitHub. Do not upload `.env`, PDFs, or local database files.

From this project folder, run:

```powershell
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin master
```

Replace the placeholder URL with your repository URL. If GitHub asks for authentication, use GitHub’s browser login, GitHub CLI, or a personal access method; never put an access token into the remote URL.

## 2. Create the Streamlit app

In Streamlit Community Cloud:

1. Choose **Create app**.
2. Select the GitHub repository.
3. Select the `master` branch.
4. Set the main file to `src/ai_document_assistant/ui.py`.
5. Use Python 3.12 if the settings page asks for a version.
6. Deploy.

The repository’s `requirements.txt` installs the package and the Streamlit UI dependency.

## 3. Add the OpenRouter secret

In the app’s **Secrets** settings, add:

```toml
AI_PROVIDER = "openrouter"
OPENROUTER_API_KEY = "paste-your-key-here"
OPENROUTER_MODEL = "deepseek/deepseek-v3.2"
OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"
```

Never commit this content to GitHub with a real key. Streamlit’s Secrets feature makes root-level values available as environment variables, which is why the existing configuration loader can read them.

## 4. Important deployment limitation

The UI uses an in-memory document index for each browser session. Uploaded documents are not permanent, and restarting or redeploying the app clears them. This is suitable for a portfolio demo. Persistent multi-user storage would require authentication, a database, and a hosted vector store.
