# Prompt Dojo

A prompt engineering trainer that helps you learn to write better AI prompts through interactive exercises.

## Features

- **Compare Prompts**: Learn by identifying which of two prompts is better crafted (subtle differences, not just length)
- **Test Your Skills**: Write your own prompts and get detailed AI feedback with scores across 4 metrics
- **AI-Powered**: Questions and feedback are dynamically generated
- **Demo Mode**: Try without an API key using sample content

## Requirements

- Python 3.8 or higher
- OpenAI API key OR Google Gemini API key

## Installation

### Mac

1. **Open Terminal** (press Cmd + Space, type "Terminal", press Enter)

2. **Navigate to the folder**:
   ```bash
   cd /path/to/Prompt\ Dojo
   ```

3. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

   Or double-click `Launch Prompt Dojo.command` in Finder.

   > **Note**: First time running the .command file, you may need to right-click and select "Open" to bypass security warnings.

### Windows

1. **Install Python** (if not already installed):
   - Download from https://www.python.org/downloads/
   - **Important**: Check "Add Python to PATH" during installation

2. **Open Command Prompt** (press Win + R, type "cmd", press Enter)

3. **Navigate to the folder**:
   ```cmd
   cd C:\path\to\Prompt Dojo
   ```

4. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

5. **Run the app**:
   ```cmd
   streamlit run app.py
   ```

   Or double-click `Launch Prompt Dojo.bat` - it will automatically install dependencies if missing.

## Quick Start

1. Launch the app (a browser window will open automatically)
2. Either:
   - Enter your OpenAI or Gemini API key and click "Test Connection"
   - Or check "Demo Mode" to try with sample content
3. Choose a training mode:
   - **Compare Prompts** - Good for beginners
   - **Test Your Skills** - Practice writing prompts

## Getting an API Key

### OpenAI
1. Go to https://platform.openai.com/api-keys
2. Create an account or sign in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)

### Google Gemini
1. Go to https://aistudio.google.com/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key

## File Structure

```
Prompt Dojo/
├── app.py                      # Main application
├── prompts.json                # Legacy static prompts (optional)
├── requirements.txt            # Python dependencies
├── Launch Prompt Dojo.command  # Mac launcher (double-click)
├── Launch Prompt Dojo.bat      # Windows launcher (double-click)
└── README.md                   # This file
```

## Troubleshooting

**"streamlit: command not found"**
- Make sure you ran `pip install -r requirements.txt`
- On Mac, try `pip3` instead of `pip`

**"Python is not recognized"** (Windows)
- Reinstall Python and check "Add Python to PATH"

**API connection fails**
- Double-check your API key is correct
- Ensure you have billing set up (OpenAI) or the API enabled (Gemini)

**App won't open in browser**
- Manually go to http://localhost:8501 in your browser

## Privacy

Your API key is stored in memory only during the session. It is never saved to disk or transmitted anywhere except directly to OpenAI/Google.
