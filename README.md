# Supply Chain Intelligence Demo

A demonstration application showcasing **OpenAI GPT-OSS-20B** for intelligent supply chain analysis and decision-making.

## 🎯 Features

- **🤖 OpenAI GPT-OSS-20B Integration**: Uses the latest 20B parameter open-source model from OpenAI
- **🚚 Supply Chain Analytics**: Inventory management, delivery tracking, and supplier analysis
- **📊 Interactive Visualizations**: Real-time charts and metrics with Plotly
- **⚡ Local AI Inference**: Runs entirely on your machine via Ollama
- **🔄 Streaming Responses**: Progressive text generation for better user experience
- **📋 Actionable Insights**: AI-powered recommendations with fallback rule-based analysis

## 🛠 Technology Stack

- **AI Model**: OpenAI GPT-OSS-20B (20.9B parameters, MXFP4 quantized)
- **Runtime**: Ollama (local inference engine)
- **Frontend**: Streamlit (interactive web interface)
- **Data**: Pandas (supply chain data processing)
- **Visualizations**: Plotly (interactive charts)
- **Language**: Python 3.12+

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama** (for running GPT-OSS locally):
   ```bash
   brew install ollama
   brew services start ollama
   ```

2. **Pull the GPT-OSS-20B model**:
   ```bash
   ollama pull gpt-oss:20b
   ```
   *Note: This will download ~13GB and may take some time*

### Setup & Launch

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd oss-toy
   chmod +x run.sh
   ./run.sh
   ```

2. **Or manual setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

3. **Open your browser**: http://localhost:8501

## 📖 Usage

1. **Load the AI Model**: Click "🚀 Load OpenAI GPT-OSS-20B" in the sidebar
2. **Ask Questions**: Enter supply chain queries like:
   - "Which items need restocking?"
   - "What are the delivery delays?"
   - "Analyze supplier performance"
   - "Optimize warehouse utilization"
3. **Get AI Insights**: View streaming responses with actionable recommendations
4. **Apply Actions**: Use suggested actions to modify data (simulation)

## 🔧 Model Performance

- **Model Size**: 20.9B parameters (13GB download)
- **Quantization**: Native MXFP4 for efficiency
- **Memory**: Runs in ~16GB RAM
- **Speed**: ~5-15 seconds per response (CPU dependent)
- **Quality**: Advanced reasoning and context understanding

## 🏗 Architecture

```
User Query → Streamlit UI → Supply Chain Data → GPT-OSS-20B (Ollama) → AI Analysis → Actionable Insights
```

## 📁 Project Structure

```
oss-toy/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── run.sh             # Quick launch script
├── test_*.py          # Test scripts
└── data/              # Mock supply chain data
```

## 🔍 Example Queries

- **Inventory**: "Show me items below reorder point"
- **Deliveries**: "Which deliveries are delayed?"
- **Suppliers**: "Compare supplier performance metrics"
- **Optimization**: "How can we improve warehouse efficiency?"
- **Forecasting**: "Predict next month's inventory needs"

## 🛠 Development

- **Add new data sources**: Modify `load_supply_chain_data()`
- **Customize prompts**: Update `generate_prompt()`
- **Add actions**: Extend the action system
- **Model tuning**: Adjust Ollama generation parameters

## 📊 Demo Data

The app includes realistic mock data:
- 8 items across electronics/accessories categories
- 3 warehouses (NY, LA, Chicago)
- Current inventory levels and reorder points
- 5 active deliveries with various statuses

## 🎉 What Makes This Special

- **Latest AI**: Uses OpenAI's newest open-source model
- **Local & Private**: No data sent to external APIs
- **Real-time**: Streaming AI responses
- **Production Ready**: Error handling, fallbacks, progress indicators
- **Extensible**: Easy to add new features and data sources