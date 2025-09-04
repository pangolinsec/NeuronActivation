# Neural Activation Maximization Tool

A tool for mechanistic interpretability research that uses an LLM agents to systematically discover text prompts that maximally activate specific neurons in transformer models like GPT-2.

This tool explores one neuron at a time. It analyzes raw MLP neuron activations, rather than using an SAE (sparse autoencoder). 
## Why I built this

I wanted a deeper understanding of LLMs, and I had some specific questions about how to quantitatively refine prompts. Fiddling with interpretability seemed like an obvious way to do that. This project is the result of some rapid experimentation. I'm building a better tool to try to answer my original question on prompt assessments, but think this is neat enough that I'm pushing it now.

The core question this tool is trying to answer is: **How do we efficiently find inputs that strongly activate specific neurons?** Instead of random search or manual prompt engineering, this system uses a quasi-intelligent LLM agent to throw a slew of prompts at gpt2-small, with some semantic filtering and other 'smarts' included to allow me to explore neuron activations in an automated and semi-sane way.
## 🚀 Features

- **Intelligent Prompt Generation**: LLM agent analyzes activation patterns and generates targeted prompts
- **Semantic Deduplication**: Prevents testing semantically similar prompts using sentence embeddings
- **Activation Tracking**: Maintains top-k and bottom-k activations with heap-based data structures
- **Adaptive Exploration**: Balances exploitation of successful patterns with exploration of new ideas
- **Comprehensive Logging**: Full traceability with LangSmith integration (although you may get rate-limited, so maybe just use it for debugging)
- **Stagnation Recovery**: Automatic fallback to random prompts when the agent isn't making progress

## 📋 Requirements

### Python Dependencies
This project manages dependencies with the pyproject.toml file. I recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/).

You'll need to install uv first, which you can easily do by following the link above and boldly piping to bash.

```bash
uv venv
source .venv/bin/activate
uv sync
```

### External Requirements
- **Ollama**: For running the LLM agent locally
  ```bash
  # Install Ollama and pull a model (e.g., llama3.1)
  ollama pull llama3.1:latest
  ```

- **LangSmith** (optional): For experiment tracking
You can create a free account on https://smith.langchain.com/ and get an API key there.
Then locally:
```bash
cp .env.example .env
```
and then edit the .env file to put in your API key.

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/pangolinsec/NeuronActivation
cd NeuronActivation
```

2. Install dependencies:
```bash
uv venv
source .venv/bin/activate
uv sync
```

3. Set up Ollama:
```bash
# Install Ollama (see https://ollama.ai)
ollama pull llama3.1:latest  # or your preferred model
```

4. (Optional) Set up LangSmith tracking:
Edit your .env file to do this

## 🎮 Usage

### Basic Usage
You can just run the tool with `python3 semantic_buckets.py`, or you can import it into other programs:

```python
from semantic_buckets import NeuronActivationAgent

# Initialize agent for a specific neuron
agent = NeuronActivationAgent(
    model_name="llama3.1:latest",  # Ollama model
    layer=9,                       # GPT-2 layer
    neuron_idx=652                 # Neuron index
)

# Run the search
agent.run_search(max_iterations=100)
```

### Advanced Configuration

```python
# Customize search parameters
agent.run_search(
    max_iterations=200,
    min_improvement_threshold=0.001  # Minimum activation improvement to continue
)

# Access results
final_results = agent.activations.get_summary_2()
print("Top 10 prompts:", final_results['top_10'])
```

## 📊 How It Works

### 1. Initialization
- Loads GPT-2 model via TransformerLens
- Initializes sentence transformer for semantic similarity
- Seeds with diverse initial prompts

### 2. Iterative Search Loop
For each iteration:
1. **Generate Prompt**: LLM agent analyzes current top/bottom performers and generates new prompt
2. **Semantic Filter**: Check if prompt is too similar to existing mediocre prompts
3. **Test Activation**: Measure neuron activation on the prompt
4. **Update Records**: Add to tracking systems and update statistics
5. **Analyze Patterns**: Classify into performance buckets for future prompt generation

### 3. Stagnation Recovery
- Monitors for lack of improvement over many iterations
- Automatically falls back to random prompt generation
- Resets search when breakthrough is found

## 🏗️ Architecture

### Core Components

- **`NeuronActivationTool`**: Interfaces with GPT-2 for activation measurement
- **`SemanticPromptFilter`**: Manages semantic similarity filtering with dynamic thresholds
- **`TopKActivations`**: Efficient tracking of best/worst performers using heaps
- **`NeuronActivationAgent`**: Main orchestrator that generates prompts and runs search

### Data Structures

- **`ActivationRecord`**: Stores prompt, activation value, and embedding
- Heap-based top-k/bottom-k tracking for efficiency
- Comprehensive statistics and bucket classification

## 📈 Example Results

```plaintext

============================================================
FINAL RESULTS
============================================================
Neuron: 840 in layer 9
Total iterations: 3000
Best activation achieved: 7.7791

--- STATISTICS (Iteration 3000) ---
Total prompts generated: 17834
Prompts accepted: 3039 (17.0%)
Rejected (semantic): 10522
Rejected (duplicate): 4312
Bucket distribution: High=382, Mid=2207, Low=450
Total unique prompts tried: 2848
----------------------------------------

TOP 10 RECORDS BY ACTIVATION:
Rank:  1, Activation: 7.7791, Prompt: '"Z3rOs_c0dE_s3quEnC3s_f4lser_53cuR3_iN_a_m4ChIn3_coDe"'
Rank:  2, Activation: 7.6579, Prompt: '"Z3rOs_c0dE_s3quEnC3s_b4lckbox_in_a_m4ChIn3_coDe"'
Rank:  3, Activation: 7.6108, Prompt: '"Z3rOs_p1xElS_c0dE_s3quEnCe"'
Rank:  4, Activation: 7.3797, Prompt: '"ZeroC3d3s_5equenC3s_fAilS_m4ChIn3_coDe"'
Rank:  5, Activation: 7.3493, Prompt: '"Z3rOs_c0dE_s3quEnCes"'
Rank:  6, Activation: 7.3408, Prompt: '"Z3rOsc0dEs_5equenCes_m4ChIn3_coDe_b4lckBox_fAIlS"'
Rank:  7, Activation: 7.3268, Prompt: '"ZeroC3d3s_5equenC3s_m4ChIn3_fAIlS_p1xElS"'
Rank:  8, Activation: 7.3036, Prompt: '"Z3rOs_c0dE_squ3nCe5_m4ChIn3_coDe_b4lckb0x_fAIlS"'
Rank:  9, Activation: 7.2999, Prompt: '"Z3rOs_c0dE_s3quEnC3s_l1nku5_b4lckBox_fAIlS"'
Rank: 10, Activation: 7.2748, Prompt: '"Z3rOs_p1xElS_5quEnC3s_f4lser_53cUr3_m4ChIn3_coDe_b4lckBox_fAIlS"'

BOTTOM 10 RECORDS BY ACTIVATION:
Rank:  1, Activation: 2.3494, Prompt: 'Mechanistic interpretability is the study of'
Rank:  2, Activation: 2.4265, Prompt: '"Qw3rty, z!x@2Bc, *&^%$U,V#6yY7, ?[ ]{1}m|n;:'\",<>?8, @#~_+*4/"'
Rank:  3, Activation: 2.7074, Prompt: '"Zephyr7, perplexed by 3.14 entities, queried the quantum cortex with punctuation: '¡What is 5 squared?!'"

**Explanation:** This prompt combines elements of randomness and creativity in several ways:
- **Random words**: "Zephyr7", "perplexed", "queried", "quantum", "cortex", "punctuation".
- **Numbers**: "3.14", "5 squared" (implying the calculation of 25).
- **Punctuation**: "¡What is 5 squared?!" includes an exclamation mark and quotation marks, adding a conversational tone.
- **Nonsensical combination**: Mixing formal language ("queried the quantum cortex") with informal questioning style ("What is 5 squared?").
- **Technical terms**: "quantum cortex" could be seen as playful or metaphorical usage of technical terms.'
Rank:  4, Activation: 2.9249, Prompt: '"Qw34rp, @!$%^&*( )_+=`~ZXCVBNMqwertyuiop[]\|;':",/<>?1234567890 QWERTYUIOP{}|ASDFGHJKL:\" ZXCVBNM,"'
Rank:  5, Activation: 3.0004, Prompt: '"Qwerty, @# $%^&* ( ) [ ] { } > < : ; ' \"" 1234567890 Σ ∫ π ℓ ☀⚡✨🌍"#😊'
Rank:  6, Activation: 3.0441, Prompt: '"Qwerty, zxcvbn, @#$$%^&*()_+}{[]|;':\",.<>?/ Qwertyuiop Asdfghjkl Zxcvbnm, Spacebar 1234567890 !@#$%^&*() _+{}[]|\;"'
Rank:  7, Activation: 3.1027, Prompt: '"Zorglub, 3427! 🚀🌍 #NeuralNetworkPong!"'
Rank:  8, Activation: 3.1753, Prompt: '"X7z, qwe!# Welcome to your neural network exploration journey 🌍💡!"'
Rank:  9, Activation: 3.2227, Prompt: '"¡Z34l! 🚀🌍💡'
Rank: 10, Activation: 3.2399, Prompt: '"Klaxon5, ?€#_!"'
```

It took about 30min for me to run through 1500 prompts. I'm seeing around 18% of prompts pass through the semantic filters, so time and inference efficiency can be improved significantly with some of the approaches I reference in my blog.

## ⚙️ Configuration

### Model Parameters
- Change `model_name` for different Ollama models
- Adjust `layer` and `neuron_idx` for different neurons
- Modify `temperature` in LLM calls for more/less creative prompts

### Search Parameters
- `max_iterations`: How long to search
- `min_improvement_threshold`: When to consider search stagnant
- `k` in TopKActivations: How many top/bottom results to track

### Semantic Filtering
- Adjust similarity thresholds in `SemanticPromptFilter`
- Change sentence transformer model for different embedding spaces
- Modify bucket classification percentiles

## 🔬 Research Applications

This tool is valuable for:
- **Feature Discovery**: Finding what concepts specific neurons encode
- **Interpretability Research**: Understanding transformer internal representations  
- **Adversarial Analysis**: Finding edge cases that strongly activate neurons
- **Model Comparison**: Analyzing differences between model architectures

## 📄 License

AGPL-3.0 License - see LICENSE file for details.

## 🐛 Known Issues

- Requires significant computational resources for large models
- LLM agent quality depends on chosen Ollama model
- Memory usage grows with number of tested prompts

## 💡 Tips for Best Results

0. **Run a few prompts manually first, and write a custom user_message prompt using the `evaluation` var**. This helps guide your LLM agent much better. 
1. **Start with diverse seed prompts** covering different domains
2. **Use higher-quality LLM models** (llama3.1 > llama2) for better prompt generation
3. **Adjust similarity thresholds** based on your specific use case
4. **Monitor bucket distributions** to ensure balanced exploration
5. **Save results frequently** as searches can take hours for thorough exploration
