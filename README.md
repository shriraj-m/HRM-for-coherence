# 🧠 Extending the Hierarchical Reasoning Model with Attention Feedback for NLP

# 📚 Overview

This project, developed by the NLP Foreigners team for CAI4304, explores how the Hierarchical Reasoning Model (HRM). A neuroscience-inspired architecture, can be adapted to Natural Language Processing (NLP) tasks such as question answering (QA) and coherence reasoning.
Our main goal is to extend the HRM architecture with an attention feedback mechanism to enable multi-hop reasoning and context refinement within a two-module framework.


# 👥 Team Members
Shriraj Mandulapalli: Project Lead
Designed project scope, oversaw architecture development, implemented attention feedback between modules.

Matthieu Drouin: Data Specialist
Conducted dataset curation and literature review on HRM and NLP datasets.

Inés Alonso: Low-Level Module Researcher
Led literature review and design for token-level module.

Alec Brenes: High-Level Module Researcher	
Focused on architecture and reasoning flow in the high-level module.


# 🎯 Problem Statement

Current transformer-based NLP models excel at pattern matching but struggle with deep reasoning, compositional generalization, and multi-hop question answering.
The Hierarchical Reasoning Model (HRM) introduces a recurrent two-module structure inspired by neuroscience, but it has never been applied to NLP.

Our project aims to:

-Adapt HRM for NLP reasoning and coherence tasks.

-Introduce attention feedback between high-level and low-level modules.

-Demonstrate hierarchical multi-hop reasoning without massive datasets.


# 📊 Data Sets

We will be using existing NLP datasets that are completely open-source. These datasets will focus on reasoning-heavy tasks and will be used to evaluate our model. The datasets that we aim to use are HotpotQA, SCAN, and CFQ. Here are the links, we will be using HuggingFace.

HotpotQA: https://github.com/hotpotqa/hotpot
SCAN: https://github.com/brendenlake/SCAN
CFQ: https://github.com/google-research/google-research/tree/master/cfq


# 📁 Project Structure

TBD


# Changes
We've made major modifications to the core architecture of the HRM. Here is a semi-detailed overview of these changes:

## Modified
### **`models/layers.py`** - Added Cross-Attention Support
- Modified the `Attention` class to support cross-attention using the  `key_value_states` parameter
- When `key_value_states=None`: performs self-attention (original HRM behavior)
- When `key_value_states` is provided: performs cross-attention

```python
def forward(self, cos_sin, hidden_states, key_value_states=None):
    if key_value_states is not None:
        # CROSS-ATTENTION: Q from hidden_states, K/V from key_value_states
        # Enabling H ↔ L communication
    else:
        # SELF-ATTENTION: Q, K, V all from hidden_states
```

### **`models/hrm/hrm_act_v2.py`** - Complete NLP Adaptation

#### Config Updates (`HierarchicalReasoningModel_ACTV2Config`)

**New Parameters:**
```python
max_sentences: int = 20           # Max sentences per document (for H module)
use_cross_attention: bool = True   # Enable H ↔ L cross-attention
sentence_pooling: str = "mean"     # Method to pool tokens → sentences
```

#### New Architecture Module: `CrossAttentionSegment`
Its **Purpose** is to allow Bidirectional cross-attention between H (sentences) and L (tokens) modules.
```
H queries L: "What token details support my sentence reasoning?"
   Input:  z_H [batch, num_sentences, hidden]
   Attend: z_L [batch, num_tokens, hidden]
   Output: H_feedback [batch, num_sentences, hidden]

L queries H: "What sentence context guides my token understanding?"
   Input:  z_L [batch, num_tokens, hidden]
   Attend: z_H [batch, num_sentences, hidden]
   Output: L_feedback [batch, num_tokens, hidden]
```

#### Sentence Pooling Method
**Added:** `_sentence_pooling()` method, which converts token embeddings to sentence embeddings using binary masks

**New Inputs:**
- `token_embeddings`: [batch, seq_len, hidden] - every word
- `sentence_masks`: [batch, max_sentences, seq_len] - which tokens belong to which sentence
- `num_sentences`: [batch ] - actual count per sample

**New Output:**
- `sentence_embeddings`: [batch, max_sentences, hidden] - semantic chunks

**Different Pooling Strategies:**
1. **Mean pooling** (default): Average all tokens in a sentence
2. **Max pooling**: Take maximum value across tokens
3. **First token**: Use first token (like BERT's [CLS])

```python
masked_tokens = token_embeddings * sentence_masks
sentence_sums = masked_tokens.sum(over_tokens)
sentence_embeddings = sentence_sums / token_counts
```

#### Dual Positional Encodings
This was **changed** to maintain TWO separate RoPE instances

```python
# For L-level (tokens)
self.rotary_emb = RotaryEmbedding(
    max_position_embeddings=seq_len + puzzle_emb_len
    # puzzle_emb_len is 1 since it's not used.
)

# For H-level (sentences)
self.rotary_emb_sentences = RotaryEmbedding(
    max_position_embeddings=max_sentences
)
```

#### Updated the Carry States

The carry states were **changed** to provide H and L DIFFERENT dimensions.

```python
# ORIGINAL (v1): Both same shape
z_H: [batch, seq_len, hidden]
z_L: [batch, seq_len, hidden]

# NEW (v2): Different shapes
z_H: [batch, max_sentences, hidden]  # Sentence-level
z_L: [batch, seq_len, hidden]        # Token-level
```

#### New Rewritten Forward Pass

**Separate Positional Encodings:**
```python
seq_info_L = dict(cos_sin=self.rotary_emb())           # Token positions
seq_info_H = dict(cos_sin=self.rotary_emb_sentences()) # Sentence positions
```

**Token Embeddings (L-level input):**
```python
input_embeddings = self._input_embeddings(batch["inputs"], ...)
# Shape: [batch, seq_len, hidden]
```

**Sentence Embeddings (H-level input):**
```python
sentence_embeddings = self._sentence_pooling(
    input_embeddings, 
    batch["sentence_masks"],
    batch["num_sentences"]
)
# Shape: [batch, max_sentences, hidden]
```

**Hierarchical Reasoning with Cross-Attention:**
```python
for H_cycle in range(H_cycles):
    for L_cycle in range(L_cycles):
        # L queries H for context
        _, L_feedback = cross_attention(z_H, z_L, ...)
        z_L = L_level(z_L, L_feedback + input_embeddings)
    
    # H queries L for details
    H_feedback, _ = cross_attention(z_H, z_L, ...)
    z_H = H_level(z_H, H_feedback + sentence_embeddings)
```

**Output from Both Levels:**
```python
output = lm_head(z_L)      # Token predictions from L
q_logits = q_head(z_H[:, 0]) # Reasoning decision from H
```


## How the Information Flows NOW

### ORIGINAL HRM Architecture (v1):
```
Token Embeddings
      ↓
   [H-level] ← additive injection from L
      ↓
   [L-level] ← additive injection from H
      ↓
Token Predictions
```


### NEW Architecture (v2):
```
Token Embeddings → [L-level: token understanding]
      ↓                    ↕ cross-attention
Sentence Pooling → [H-level: sentence reasoning]
      ↓                    ↕ cross-attention
      └────────────→ [L-level: refined tokens]
                            ↓
                    Token Predictions
                    
[H-level] → Halting Decision
```

## Everything New in Our Version

### 1. **Two-Level Representations**
- **L-level:** Word-level understanding (tokens)
- **H-level:** Sentence-level reasoning (semantic chunks)

### 2. **Cross-Attention Feedback**
- Replaces additive injection
- H queries L: "Which words support this sentence?"
- L queries H: "Which sentence context guides this word?"

### 3. **Sentence Pooling**
- Converts tokens → sentences using masks
- Flexible pooling strategies (mean/max/first)
- Handles variable-length sentences

### 4. **Separate Positional Encodings**
- Token positions for L-level
- Sentence positions for H-level
- Proper handling in cross-attention

### 5. **Hierarchical Output**
- Token predictions from L (language modeling)
- Reasoning decisions from H (halting/QA)
- 



# 🧾 References

Wang et al., Hierarchical Reasoning Model, (2025).

Bounsi et al., Transformers Meet Neural Algorithmic Reasoners, (2024).

Wei & Tay et al., Chain-of-Thought Prompting Elicits Reasoning in LLMs, (2022).

Gong & Bowman, Ruminating Reader: Reasoning with Gated Multi-Hop Attention, ACL (2018).

Guo & Chen, Decoupling Knowledge and Reasoning in Transformers, (2025).
