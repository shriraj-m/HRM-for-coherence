from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV2InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV2Carry:
    inner_carry: HierarchicalReasoningModel_ACTV2InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV2Config(BaseModel):
    batch_size: int
    seq_len: int # this is the max # of tokens in the document
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int # number of high-level reasoning cycles
    L_cycles: int # of low-level processing cycles PER H cycle

    H_layers: int # number of layers in the high-level reasoning module
    L_layers: int # number of layers in the low-level processing module

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # NLP stuff
    max_sentences: int = 20 # default of 20, this helps with sequence length 
    use_cross_attention: bool = True # whether to enable cross-attention between H and L
    # when True: H queries L for token details, L queries H for context
    sentence_pooling: str = "mean" # how we pool tokens into sentences


class CrossAttentionSegment(nn.Module):
    """
    Bi-Directional Cross Attention between the High and Low Levels
    - H queries L: "What token details support my sentence-level reasoning?"
    - L queries H: "What sentence-level context should guide my token understanding?"
    
    This mimics human reading: your brain (H) guides where your eyes (L) look,
    and what your eyes see refines your understanding.
    """
    def __init__(self, config: HierarchicalReasoningModel_ACTV2Config):
        super().__init__()

        # H queries L for more supporting details
        # H will be a query [batch, # of sentences, hidden]
        # L is context [batch, # of tokens, hidden]
        # output will be redefined H
        self.H_query_L = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            casual=False # allows for bidirectional
        )

        # now L queries H for context
        # L will be a query [batch, # of tokens, hidden]
        # H is context [batch, # of sentences, hidden]
        # output will be redefined L
        self.L_query_H = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            casual=False # allows for bidirectional
        )

        self.norm_eps = config.rms_norm_eps


    def forward(
        self,
        z_H: torch.Tensor, # high level state
        z_L: torch.Tensor, # low level state
        cos_sin_H: CosSin, # embedding for sentences
        cos_sin_L: CosSin, # embedding for tokens
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs bidirectional cross-attention between H and L.
        
        Args:
            z_H: Sentence-level representations [batch, num_sentences, hidden_size]
            z_L: Token-level representations [batch, num_tokens, hidden_size]
            cos_sin_H: RoPE encodings for sentence positions (or None)
            cos_sin_L: RoPE encodings for token positions (or None)
            
        Returns:
            H_feedback: Information from L to inject into H [batch, num_sentences, hidden_size]
            L_feedback: Information from H to inject into L [batch, num_tokens, hidden_size]
        """
        H_feedback = self.H_query_L(
            cos_sin=cos_sin_H, # sentence position for queries
            hidden_states=z_H # queries com from high level
            # key_value_states=z_L # key/values of low level
        )
        L_feedback = self.L_query_H(
            cos_sin=cos_sin_L, # token position for queries
            hidden_states=z_L, # queries come from low level
            # key_value_states=z_H # key/values of high level
        )

        return H_feedback, L_feedback


class HierarchicalReasoningModel_ACTV2Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV2Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV2ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV2Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV2_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV2Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        # got to modify rope as well
        # (L) uses RoPE for tokens 
        # (H) needs a separate RoPE for sentences (diff seq length)
        if self.config.pos_encodings == "rope":
            # create separate RoPE for sentene pos
            self.rotary_emb_sentences = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.max_sentences,
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos_sentences = CastedEmbedding(
                self.config.max_sentences,  # number of sentence positions
                self.config.hidden_size, 
                init_std=embed_init_std, 
                cast_to=self.forward_dtype
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV2ReasoningModule(layers=[HierarchicalReasoningModel_ACTV2Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HierarchicalReasoningModel_ACTV2ReasoningModule(layers=[HierarchicalReasoningModel_ACTV2Block(self.config) for _i in range(self.config.L_layers)])
        
        # cross attention for H <--> L feedback
        if self.config.use_cross_attention:
            self.cross_attention = CrossAttentionSegment(self.config)
            # This module enables:
            # - H to query L: "Which tokens support my sentence understanding?"
            # - L to query H: "Which sentence context guides this token?"
        
        
        # changed initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(self.config.max_sentences, self.config.hidden_size, dtype=self.forward_dtype), 
                std=1
            ), 
            persistent=True
        )
        # L's initial state stays token-level
        self.L_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype), 
                std=1
            ), 
            persistent=True
        )
        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    
    # new required function to pool token embeddings into sentence embeddings 
    def _sentence_pooling(
        self,
        token_embeddings: torch.Tensor,
        sentence_masks: torch.Tensor,
        num_sentences: torch.Tensor
        ) -> torch.Tensor:
        """
        pools the tokens into sentence embeddings.
        this lets us go from token level (L) to sentence level (H)
        token embeddings are [batch, # of tokens, hidden] (eveyr word)
        sentence positions are [batch, # of tokens] (semantic chunks)

        itll return sentence embeddings

        Args:
            token_embeddings: word representation
            sentence_masks: binary marks showing which tokens belong to a sentence
            num_sentences: number of sentences the sample has.
        Returns:
            sentence_embeddings: sentence representation
        """
        
        batch_size, max_sentences, sequence_len = sentence_masks.shape
        _, _, hidden_size = token_embeddings.shape

        # firstly, we need to multiply masks with embeddings, however they have diff shapes
        expanded_tokens = token_embeddings.unsqueeze(1) # turns hidden into hudden_size and adds a 1
        expanded_masks = sentence_masks.unsqueeze(-1) # adds a 1 to the dim

        # now we apply the masks
        # this will zero out tokens that dont belong to a sentence
        masked_tokens = expanded_tokens * expanded_masks

        # now get sum over squence length dim to get total for each sentennce
        sentence_sums = masked_tokens.sum(dim=2) # sum across sequence_length and [batch, max_sentences, hidden_size]

        # get the token count in the sentence to get the average
        token_counts = expanded_masks.sum(dim=2) # sum across mask values and [batch, max_sentences, 1]

        token_counts = torch.clamp(token_counts, min=1) # avoid division by zero. clamp assures no value is less than 1

        # now we divide sum by count to get sentence_pooling type in config
        if self.config.sentence_pooling == "mean":
            sentence_embeddings = sentence_sums / token_counts
        elif self.config.sentence_pooling == "max":
            # max pooling: take maximum value across tokens
            masked_tokens_for_max = torch.where(
                expanded_masks.bool(),
                expanded_tokens.expand_as(masked_tokens),
                torch.tensor(float('-inf'), device=token_embeddings.device)
            )
            sentence_embeddings = masked_tokens_for_max.max(dim=2)[0]
        elif self.config.sentence_pooling == "first":
            # use first token of each sentence (like BERT's [CLS] token approach)
            # get first token index for each sentence
            first_token_indices = sentence_masks.argmax(dim=2)  # [batch, max_sentences]
            sentence_embeddings = torch.gather(
                token_embeddings.unsqueeze(1).expand(-1, max_sentences, -1, -1),
                dim=2,
                index=first_token_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, hidden_size)
            ).squeeze(2)
        else:
            raise ValueError(f"error unknown pooling method: {self.config.sentence_pooling}")

        # scale input embeddings for consistent magnitude
        sentence_embeddings = self.embed_scale * sentence_embeddings
        return sentence_embeddings



    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV2InnerCarry(
            z_H=torch.empty(batch_size, self.config.max_sentences, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV2InnerCarry):
        return HierarchicalReasoningModel_ACTV2InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV2InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV2InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV2InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV2(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV2Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV2_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV2Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV2Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV2Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return HierarchicalReasoningModel_ACTV2Carry(new_inner_carry, new_steps, halted, new_current_data), outputs