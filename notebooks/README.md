# Azure LLM Toolkit - Interactive Tutorials

This directory contains comprehensive Jupyter notebooks to help you learn and master the Azure LLM Toolkit.

## 📚 Notebooks

### 1. [Getting Started](01_getting_started.ipynb)
**Difficulty**: Beginner  
**Time**: 30-45 minutes

Learn the fundamentals:
- Installation and setup
- Basic configuration
- Simple chat completion
- Streaming responses
- Function calling basics
- Cost tracking
- Embeddings
- Caching
- Health checks

**Prerequisites**: Azure OpenAI API access

---

### 2. [Rate Limiting Strategies](02_rate_limiting_strategies.ipynb)
**Difficulty**: Intermediate  
**Time**: 45-60 minutes

Master rate limiting:
- Understanding Azure OpenAI rate limits
- Token bucket algorithm
- Custom rate limit configuration
- Handling concurrent requests
- Adaptive rate limiting
- Dealing with 429 errors
- Monitoring rate limit usage
- Batch processing with rate limits
- Best practices for high-throughput

**Prerequisites**: Completed notebook 1

---

### 3. [Cost Optimization](03_cost_optimization.ipynb)
**Difficulty**: Intermediate  
**Time**: 30-45 minutes

Minimize your API costs:
- Token usage optimization
- Caching strategies
- Model selection for cost efficiency
- Batch processing savings
- Cost tracking and analytics
- Budget alerts
- Best practices summary

**Prerequisites**: Basic understanding of Azure OpenAI pricing

---

### 4. [RAG Implementation](04_rag_implementation.ipynb)
**Difficulty**: Advanced  
**Time**: 60-90 minutes

Build a Retrieval-Augmented Generation system:
- Document processing and chunking
- Embedding generation
- Vector storage and retrieval
- Similarity search
- Query augmentation
- Answer generation
- Reranking for better results
- End-to-end RAG pipeline

**Prerequisites**: Completed notebooks 1-2, understanding of embeddings

---

### 5. [Agent Patterns](05_agent_patterns.ipynb)
**Difficulty**: Advanced  
**Time**: 60-90 minutes

Create intelligent agents:
- Defining and using tools
- Maintaining conversation history
- Decision making and actions
- Multi-step workflows
- ReAct pattern implementation
- Multi-agent collaboration
- Agent orchestration

**Prerequisites**: Completed notebooks 1-3, understanding of function calling

---

### 6. [Production Deployment](06_production_deployment.ipynb)
**Difficulty**: Advanced  
**Time**: 45-60 minutes

Deploy to production:
- Configuration management
- Health checks and monitoring
- Error handling and logging
- Performance optimization
- Security best practices
- OpenTelemetry integration
- FastAPI integration
- Kubernetes deployment
- Production readiness checklist

**Prerequisites**: Completed notebooks 1-5, familiarity with deployment concepts

---

## 🚀 Getting Started

### Installation

1. **Install Jupyter**:
   ```bash
   pip install jupyter notebook
   # or
   pip install jupyterlab
   ```

2. **Install Azure LLM Toolkit**:
   ```bash
   pip install azure-llm-toolkit
   # or for development
   pip install -e .
   ```

3. **Install notebook dependencies**:
   ```bash
   pip install tiktoken numpy
   ```

### Running Notebooks

**Jupyter Notebook**:
```bash
cd notebooks
jupyter notebook
```

**JupyterLab**:
```bash
cd notebooks
jupyter lab
```

**VS Code**:
1. Install the Jupyter extension
2. Open a `.ipynb` file
3. Select your Python kernel
4. Run cells with Shift+Enter

---

## ⚙️ Configuration

Before running the notebooks, configure your Azure OpenAI credentials:

### Option 1: Environment Variables (Recommended)

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="gpt-4"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

### Option 2: `.env` File

Create a `.env` file in the notebooks directory:

```env
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

Then load it in notebooks:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Option 3: Direct Configuration

Set credentials directly in the notebook:
```python
import os
os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com"
```

⚠️ **Security Warning**: Never commit credentials to version control!

---

## 📋 Learning Path

### Beginner Path
1. 01_getting_started.ipynb
2. 03_cost_optimization.ipynb
3. 02_rate_limiting_strategies.ipynb

### Intermediate Path
1. Complete Beginner Path
2. 04_rag_implementation.ipynb
3. 05_agent_patterns.ipynb

### Advanced Path
1. Complete Intermediate Path
2. 06_production_deployment.ipynb
3. Explore the examples/ directory
4. Read the benchmarks/ code

---

## 💡 Tips

### For Learning
- Run cells sequentially (don't skip)
- Modify examples and experiment
- Read inline comments carefully
- Check the output of each cell
- Try different parameters

### For Development
- Use the notebooks as templates
- Copy patterns to your own code
- Refer back when stuck
- Bookmark frequently used sections

### For Production
- Start with notebook 6 (Production Deployment)
- Implement health checks (from notebook 1)
- Set up monitoring (from notebook 6)
- Use rate limiting (from notebook 2)
- Track costs (from notebook 3)

---

## 🛠️ Troubleshooting

### "Module not found" errors
```bash
pip install azure-llm-toolkit jupyter
```

### "API key not configured"
- Check environment variables: `echo $AZURE_OPENAI_API_KEY`
- Verify .env file exists and is loaded
- Ensure credentials are set before creating client

### "Rate limit exceeded"
- Check your Azure deployment limits
- Reduce concurrency or requests per minute
- Implement proper rate limiting (see notebook 2)

### "Out of memory"
- Restart the kernel: Kernel → Restart
- Clear output: Cell → All Output → Clear
- Process data in smaller batches

### Notebook won't run
- Ensure correct Python version (3.9+)
- Check Jupyter installation: `jupyter --version`
- Try: `jupyter notebook --generate-config`

---

## 📚 Additional Resources

### Documentation
- [Main README](../README.md)
- [API Documentation](../docs/)
- [Examples](../examples/)
- [Tests](../tests/)

### Azure Resources
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure OpenAI Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
- [Rate Limits](https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits)

### Community
- [GitHub Issues](https://github.com/tsoernes/azure-llm-toolkit/issues)
- [Discussions](https://github.com/tsoernes/azure-llm-toolkit/discussions)
- [Contributing Guide](../CONTRIBUTING.md)

---

## 🤝 Contributing

Found an issue or want to improve a notebook?

1. Open an issue describing the problem/enhancement
2. Fork the repository
3. Create a branch: `git checkout -b fix/notebook-issue`
4. Make your changes
5. Test thoroughly
6. Submit a pull request

---

## 📝 License

Same as the main project - see [LICENSE](../LICENSE)

---

## 🙏 Acknowledgments

- Built with [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- Inspired by the community's needs
- Thanks to all contributors!

---

**Happy Learning!** 🚀

If you find these tutorials helpful, please ⭐ star the repository!