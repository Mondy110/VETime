# Cross-Modal Relational Guidance Design

## Goal

Add Cross-Modal Relational Guidance (CMRG) to VETime so frozen MAE visual semantics guide temporal self-attention through relation-logit corrections, while preserving the existing visual fusion path.

## Constraints

- CMRG is a VETime module; it does not redefine or modify TimeRCD's pretrained Q/K/V weights.
- The existing `unfold_image -> PTA -> AWCL/TMF` path remains unchanged.
- CMRG uses a fixed 16-token Relation Distiller, not a Prototype Bank.
- Guider relations are unnormalized logits; do not apply Softmax before Corr construction.
- The full learned metric is `W_R: (16, 16)`, initialized to identity and shared across heads and layers.
- Corr is computed once from initial temporal patch embeddings and shared across all encoder layers.
- Each layer has a scalar gate initialized to zero. The main mode injects every layer; last-layer-only is an ablation mode.
- Store relation factors, not a persistent `(B, H, N, N)` Corr tensor.
- CMRG initially supports only the existing RoPE custom attention implementation.

## Data Flow

1. `TimeSeriesEncoder` creates `Z_T^0: (B, N, 512)`, RoPE frequencies, feature IDs, and a patch mask before temporal encoding.
2. The frozen MAE returns raw patch tokens `Z_V^MAE: (B, K_MAE, 768)` before `unfold_image`.
3. `RelationDistiller` projects visual tokens to 512 and uses 16 learnable relation queries with one cross-attention layer to produce `Z_V^R: (B, 16, 512)`.
4. `CrossModalRelationGuider` computes `R_TV: (B, 8, N, 16)` using projected, normalized temporal and relational tokens, without Softmax. Padded temporal rows are zeroed.
5. It computes `M = R_TV W_R: (B, 8, N, 16)` and passes `R_TV`, `M`, and the patch mask to the temporal encoder.
6. Every temporal attention layer forms `C = M R_TV^T` transiently and adds `alpha_l * C / sqrt(head_dim)` to the existing attention scores.
7. MAE raw tokens separately proceed through the unchanged `unfold_image` fusion path.

## Formulae

`R_TV,h = Q_T,h^g (K_V,h^g)^T / sqrt(64)`.

`C_V,h = R_TV,h W_R R_TV,h^T`.

`S_l,h = Q_l,h K_l,h^T / sqrt(d_h) + alpha_l C_V,h / sqrt(d_h) + B_binary + B_mask`.

`W_R` is a shared `(16, 16)` parameter initialized to identity. `C_V` is head-specific. `alpha_l` is one scalar per temporal layer, shared across heads, initialized to `0.0`.

## Configuration

`cmrg_enabled=false`, `cmrg_num_relation_tokens=16`, `cmrg_guide_dim=512`, `cmrg_num_heads=8`, `cmrg_metric_init=identity`, `cmrg_gate_init=0.0`, `cmrg_injection_mode=all_layers`, `cmrg_factorized=true`, and `cmrg_log_interval=100`.

## Compatibility and Monitoring

Old checkpoints load with `strict=False`; only CMRG weights are missing. Q/K/V, RoPE, BinaryAttentionBias, mask logic, and LoRA wrappers remain intact. Freeze mode must leave CMRG trainable outside classification-only warmup.

Log every layer's `alpha_l` and `rho_l = ||alpha_l C_V||_F / (||Q_l K_l^T||_F + eps)` from factorized Gram products so logging does not materialize Corr.

## Verification

Test Relation Distiller and Guider shapes, unnormalized relations, factorized/direct Corr equality, padding behavior, zero-gate encoder equivalence, gate gradient staging, legacy checkpoint compatibility, LoRA/freeze backward smoke tests, and all configured ablation modes.
