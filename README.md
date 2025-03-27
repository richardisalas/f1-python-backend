# F1 AI Assistant

An AI-powered assistant for Formula 1 data analysis and strategy optimization with real-time web search capabilities.

## Features

- Natural language interface for asking F1-related questions
- Real-time web search for up-to-date racing information
- Tracks conversation context for follow-up questions
- Clean, citation-free responses with detailed information
- Command-line interface for easy interaction
- Optional web search capability (can be disabled)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/richardisalas/f1-python-backend.git
   cd f1-python-backend
   ```

2. Install dependencies:
   ```
   pip install openai python-dotenv
   ```

3. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the assistant in interactive mode:
```
python main.py
```

Run without web search capability (uses GPT-4o instead of GPT-4o-search-preview):
```
python main.py --no-web-search
```

Example queries:
- "Who won the last F1 race?"
- "What are Max Verstappen's championship stats?"
- "What tire strategies work best at Monaco?"
- "Tell me about the upcoming race weekend"

Use batch mode with an input file:
```
python main.py -b -i input.json -o results.json
```

## Configuration

You can customize the assistant with a JSON configuration file:
```
python main.py -c config.json
```

## Requirements

- Python 3.8+
- OpenAI API key with access to GPT-4o-search-preview model

## License

MIT 