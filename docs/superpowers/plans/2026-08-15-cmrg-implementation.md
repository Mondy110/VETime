# CMRG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add factorized Cross-Modal Relational Guidance to VETime without changing its existing visual fusion path.

**Architecture:** A new CMRG module distills frozen MAE patches into 16 Distilled Relation Tokens, produces unnormalized temporal-to-visual relation logits, and passes cached relation factors to the custom temporal attention. Attention layers materialize the Corr term only while forming their existing score tensor.

**Tech Stack:** Python, PyTorch, pytest, existing VETime/TimeRCD custom RoPE attention.

**Spec:** `docs/superpowers/specs/2026-08-15-cmrg-design.md`

## Global Constraints

- Keep `unfold_image -> PTA -> AWCL/TMF` behavior unchanged.
- Use `K_R=16`, guide dimension 512, eight heads, identity-initialized `W_R`, and zero-initialized scalar per-layer gates.
- Never apply Softmax to Guider relation logits.
- Do not persist a full Corr tensor across layers.
- Preserve legacy checkpoint and LoRA compatibility.

---

### Task 1: CMRG relation components

**Files:**
- Create: `model/CMRG.py`
- Test: `tests/test_cmrg.py`

**Interfaces:**
- Produces `RelationDistiller(vision_dim, guide_dim, num_relation_tokens, num_heads)`.
- Produces `CrossModalRelationGuider(temporal_dim, guide_dim, num_heads, num_relation_tokens)` returning `(relation_logits, relation_factor)`.

- [ ] **Step 1: Write failing shape and factorization tests**

```python
def test_guider_returns_unnormalized_relation_factors():
    guider = CrossModalRelationGuider(8, 8, 2, 3)
    logits, factor = guider(torch.randn(2, 5, 8), torch.randn(2, 3, 8), torch.ones(2, 5, dtype=torch.bool))
    assert logits.shape == (2, 2, 5, 3)
    assert factor.shape == (2, 2, 5, 3)
    assert not torch.allclose(logits.sum(-1), torch.ones(2, 2, 5))
```

- [ ] **Step 2: Run the new test and verify import failure**

Run: `pytest tests/test_cmrg.py -q`

- [ ] **Step 3: Implement minimal CMRG classes**

```python
relation_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
relation_logits = relation_logits * valid_mask
relation_factor = torch.matmul(relation_logits, self.relation_metric)
```

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `pytest tests/test_cmrg.py -q`

### Task 2: Temporal encoder context propagation

**Files:**
- Modify: `model/TS_encoder/ts_encoder.py`
- Modify: `model/TS_encoder/encoding_utils.py`
- Test: `tests/test_cmrg.py`

**Interfaces:**
- `TimeSeriesEncoder.prepare_inputs(time_series, mask)` returns patch embeddings and metadata.
- `TimeSeriesEncoder.encode_prepared(prepared, cmrg_context=None)` returns temporal patch embeddings.
- Existing `forward(time_series, mask, cmrg_context=None)` preserves its return tuple.

- [ ] **Step 1: Write failing zero-gate equivalence test**

```python
def test_zero_gate_context_preserves_attention_output():
    baseline = attention(query, key, value, freqs, query_id, query_id, mask)
    guided = attention(query, key, value, freqs, query_id, query_id, mask, cmrg_context=context, cmrg_alpha=torch.tensor(0.0))
    torch.testing.assert_close(guided, baseline, rtol=0, atol=0)
```

- [ ] **Step 2: Run the new test and verify signature failure**

Run: `pytest tests/test_cmrg.py::test_zero_gate_context_preserves_attention_output -q`

- [ ] **Step 3: Add optional context propagation and score correction**

```python
corr = torch.matmul(context.relation_factor, context.relation_logits.transpose(-2, -1))
scores = scores + cmrg_alpha * corr / math.sqrt(self.head_dim)
```

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `pytest tests/test_cmrg.py -q`

### Task 3: VETime integration and configuration

**Files:**
- Modify: `model/VETime.py`
- Modify: `model/TS_encoder/ts_encoder.py`
- Modify: `train.py`
- Test: `tests/test_cmrg.py`

**Interfaces:**
- `VETIME` accepts CMRG configuration from `config_t`.
- MAE executes once; CMRG consumes raw tokens and the original path consumes unfolded tokens.

- [ ] **Step 1: Write failing integration test for raw visual token and context path**

```python
def test_vetime_cmrg_uses_raw_mae_tokens_without_replacing_fusion_path():
    model = make_tiny_vetime(cmrg_enabled=True)
    model(images, series, mask, image_size)
    assert model.cmrg_guider.last_relation_shape[-1] == 16
```

- [ ] **Step 2: Run the integration test and verify it fails**

Run: `pytest tests/test_cmrg.py::test_vetime_cmrg_uses_raw_mae_tokens_without_replacing_fusion_path -q`

- [ ] **Step 3: Integrate preparation, MAE raw tokens, and context creation**

```python
prepared = self.ts_encoder.ts_encoder.prepare_inputs(time_series, att_mask)
raw_visual_tokens, _ = self.vit_encoder(hidden_states)
relation_tokens = self.cmrg_distiller(raw_visual_tokens)
context = self.cmrg_guider(prepared.embedded_patches, relation_tokens, prepared.full_mask)
patch_embeddings = self.ts_encoder.ts_encoder.encode_prepared(prepared, context)
```

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `pytest tests/test_cmrg.py -q`

### Task 4: Training controls, monitoring, and regression coverage

**Files:**
- Modify: `train.py`
- Test: `tests/test_cmrg.py`

**Interfaces:**
- CMRG remains trainable in freeze mode outside classifier warmup.
- Training logger receives `alpha_l` and factorized `rho_l` values.

- [ ] **Step 1: Write failing tests for gate gradient staging and factorized Corr equality**

```python
def test_zero_gate_updates_gate_before_guider_parameters():
    loss = guided_attention(...).square().mean()
    loss.backward()
    assert attention.cmrg_alpha.grad is not None
    assert guider.temporal_proj.weight.grad is None
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_cmrg.py -q`

- [ ] **Step 3: Add logging helpers and freeze-mode exclusions**

```python
rho = cmrg_context.relative_strength(query_states, key_states, cmrg_alpha)
metrics[f"cmrg/alpha_{layer_idx}"] = cmrg_alpha.detach()
metrics[f"cmrg/rho_{layer_idx}"] = rho.detach()
```

- [ ] **Step 4: Run full regression suite**

Run: `pytest tests -q`
