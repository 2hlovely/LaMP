import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import time
import datetime
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerBase
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.modeling import build_backbone
from dassl.evaluation import build_evaluator
import clip


class DeepTextEncoder(nn.Module):
    """CLIP text encoder with optional branch-specific deep prompts."""
    def __init__(
        self,
        clip_model,
        prompt_depth,
        n_ctx,
        ctx_dim,
        device,
        learnable_prompts=True,
        use_checkpoint=False,
    ):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.prompt_depth = prompt_depth
        self.n_ctx = n_ctx
        self.device = device
        self.use_checkpoint = use_checkpoint
        if learnable_prompts:
            self.deep_prompts = nn.Parameter(
                torch.empty(prompt_depth, n_ctx, ctx_dim, dtype=self.dtype)
            )
            if prompt_depth > 0:
                nn.init.normal_(self.deep_prompts, std=0.02)
        else:
            self.register_parameter("deep_prompts", None)

    def _resblock_forward(self, resblock, x):
        """Wrapper for checkpoint-friendly resblock forward."""
        out = resblock(x)
        return out[0] if isinstance(out, list) else out

    def forward(self, prompts, tokenized_prompts, deep_prompts=None):
        """
        Forward pass with deep prompts
        
        Args:
            prompts: (batch, 77, dim) - input prompts with shallow tokens replaced
            tokenized_prompts: (batch, 77) - tokenized prompts
            deep_prompts: optional branch-specific prompts, shape
                (prompt_depth, n_ctx, dim)
            
        Returns:
            text_features: (batch, dim) - text features
        """
        if deep_prompts is None:
            deep_prompts = self.deep_prompts

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Process through transformer layers with deep prompts
        for i, resblock in enumerate(self.transformer.resblocks):
            # If this layer needs deep prompts
            if i < self.prompt_depth:
                if deep_prompts is None:
                    raise RuntimeError(
                        "Deep prompts are required when prompt_depth > 0"
                    )
                # x shape: (77, batch, dim)
                # Keep the first token (SOS) and tokens after prompt positions
                prefix = x[:1, :, :]  # SOS token
                suffix = x[1 + self.n_ctx:, :, :]  # Remaining tokens
                
                # Get deep prompts for this layer
                # deep_prompts[i]: (n_ctx, dim)
                # Expand to match batch size
                deep_prompt = deep_prompts[i].unsqueeze(1).expand(-1, x.shape[1], -1)
                
                # Replace shallow prompts with deep prompts
                x = torch.cat([prefix, deep_prompt, suffix], dim=0)
            
            # Pass through the residual block with optional gradient checkpointing
            if self.use_checkpoint and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(
                    self._resblock_forward, resblock, x,
                    use_reentrant=False,
                )
            else:
                x = resblock(x)[0] if isinstance(x, list) else resblock(x)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # Take features from the eot embedding
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x


class LaMPModel(nn.Module):
    """Three-branch LaMP model.

    Branches:
        class: learns category semantics, one prototype per class.
        style: learns domain/style semantics, one prototype per domain.
        composition: learns class-style semantics, one prototype per domain-class.
    """
    def __init__(self, cfg, classnames, domain_names, clip_model, device, llm_descriptions_path):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.classnames = classnames
        self.domain_names = domain_names
        self.n_cls = len(classnames)
        self.n_domain = len(domain_names)

        pro_cfg = cfg.TRAINER.LAMP

        self.n_ctx = pro_cfg.N_CTX
        self.prompt_depth = pro_cfg.PROMPT_DEPTH_TEXT
        self.lambda_mse = pro_cfg.LAMBDA_MSE
        self.lambda_class = pro_cfg.LAMBDA_CLASS
        self.lambda_style = pro_cfg.LAMBDA_STYLE
        self.lambda_comp = pro_cfg.LAMBDA_COMP
        self.logit_class_weight = pro_cfg.LOGIT_CLASS_WEIGHT
        self.logit_comp_weight = pro_cfg.LOGIT_COMP_WEIGHT
        self.style_temperature = pro_cfg.STYLE_TEMPERATURE
        self.inference_mode = pro_cfg.INFERENCE_MODE.lower()
        self.class_prompt_template = pro_cfg.CLASS_PROMPT_TEMPLATE
        self.style_prompt_template = pro_cfg.STYLE_PROMPT_TEMPLATE
        self.comp_prompt_template = pro_cfg.COMP_PROMPT_TEMPLATE
        self.precision = pro_cfg.PREC
        self.use_checkpoint = pro_cfg.USE_CHECKPOINT

        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        # Learnable params always FP32: autocast handles FP16 conversion in forward,
        # and GradScaler requires FP32 gradients for unscaling
        param_dtype = torch.float32

        print("LaMP: Learning three prompt branches")
        print(f"Number of classes: {self.n_cls}")
        print(f"Number of domains: {self.n_domain}")
        print(f"Context tokens per branch: {self.n_ctx}")
        print(f"Prompt depth per branch: {self.prompt_depth} layers")
        print(f"Inference mode: {self.inference_mode}")

        self.clip_model = clip_model

        self.class_ctx = self._init_ctx(self.n_ctx, ctx_dim, param_dtype)
        self.style_ctx = self._init_ctx(self.n_ctx, ctx_dim, param_dtype)
        self.comp_ctx = self._init_ctx(self.n_ctx, ctx_dim, param_dtype)

        self.class_deep_prompts = self._init_deep_prompts(
            self.prompt_depth, self.n_ctx, ctx_dim, param_dtype
        )
        self.style_deep_prompts = self._init_deep_prompts(
            self.prompt_depth, self.n_ctx, ctx_dim, param_dtype
        )
        self.comp_deep_prompts = self._init_deep_prompts(
            self.prompt_depth, self.n_ctx, ctx_dim, param_dtype
        )

        self.text_encoder = DeepTextEncoder(
            clip_model,
            self.prompt_depth,
            self.n_ctx,
            ctx_dim,
            device,
            learnable_prompts=False,
            use_checkpoint=self.use_checkpoint,
        )

        # UOT hyperparameters for target construction.
        # USE_UOT_TARGETS is kept as a global switch for compatibility.
        self.use_uot_targets = bool(getattr(pro_cfg, "USE_UOT_TARGETS", True))
        self.use_desc_uot_targets = bool(
            getattr(pro_cfg, "USE_DESC_UOT_TARGETS", self.use_uot_targets)
        )
        # By default, after description-level UOT, class/style prototypes use mean pooling
        # over the denoised T[d, c]. Set USE_PROTO_UOT_TARGETS=True only if you want
        # to stack another UOT pooling over domain/class prototypes.
        self.use_proto_uot_targets = bool(
            getattr(pro_cfg, "USE_PROTO_UOT_TARGETS", False)
        )
        self.uot_epsilon = float(getattr(pro_cfg, "UOT_EPSILON", 0.10))
        self.uot_tau = float(getattr(pro_cfg, "UOT_TAU", 1.0))
        self.uot_iters = int(getattr(pro_cfg, "UOT_ITERS", 50))

        print(f"Use UOT targets: {self.use_uot_targets}")
        print(f"Use description-level UOT: {self.use_desc_uot_targets}")
        print(f"Use prototype-level UOT: {self.use_proto_uot_targets}")
        if self.use_uot_targets:
            print(
                f"UOT params: epsilon={self.uot_epsilon}, "
                f"tau={self.uot_tau}, iters={self.uot_iters}"
            )

        # T[d][c]: frozen CLIP text features from LLM domain-class descriptions.
        # If USE_DESC_UOT_TARGETS=True, each T[d, c] is obtained by UOT pooling
        # over its raw LLM description features. This is where noisy descriptions
        # are suppressed before constructing class/style/composition targets.
        self.target_features = self._compute_domain_class_targets(llm_descriptions_path)

        # Build class/style targets from the denoised composition targets T[d, c].
        # Default after description-level UOT:
        #   T_cls[c]   = mean_d T[d, c]
        #   T_style[d] = mean_c T[d, c]
        # Optional if USE_PROTO_UOT_TARGETS=True:
        #   T_cls[c]   = UOTPool_d T[d, c]
        #   T_style[d] = UOTPool_c T[d, c]
        self.target_features_cls = self._build_class_level_targets()
        self.target_features_style = self._build_style_level_targets()

    def _init_ctx(self, n_ctx, ctx_dim, dtype):
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        return nn.Parameter(ctx_vectors)

    def _init_deep_prompts(self, prompt_depth, n_ctx, ctx_dim, dtype):
        prompts = torch.empty(prompt_depth, n_ctx, ctx_dim, dtype=dtype)
        if prompt_depth > 0:
            nn.init.normal_(prompts, std=0.02)
        return nn.Parameter(prompts)

    @staticmethod
    def _dedupe(items):
        output = []
        for item in items:
            if item not in output:
                output.append(item)
        return output

    def _name_candidates(self, name):
        name = str(name)
        space_name = name.replace("_", " ")
        underscore_name = name.replace(" ", "_")
        candidates = [
            name,
            name.lower(),
            name.upper(),
            space_name,
            space_name.lower(),
            space_name.title(),
            underscore_name,
            underscore_name.lower(),
            underscore_name.upper(),
        ]

        alias_map = {
            "adversarial": ["imagenet-a", "imagenet_a", "imageneta", "ImageNet-A"],
            "rendition": ["imagenet-r", "imagenet_r", "imagenetr", "ImageNet-R"],
            "sketch": ["imagenet-s", "imagenet_s", "imagenets", "ImageNet-S"],
            "v2": ["imagenet-v2", "imagenet_v2", "imagenetv2", "ImageNet-V2"],
        }
        norm_name = name.replace("_", "-").lower()
        for alias_key, aliases in alias_map.items():
            if norm_name == alias_key or norm_name in [a.lower() for a in aliases]:
                for alias in aliases:
                    alias_space = alias.replace("_", " ")
                    alias_under = alias.replace(" ", "_")
                    candidates.extend([
                        alias,
                        alias.lower(),
                        alias.upper(),
                        alias_space,
                        alias_space.lower(),
                        alias_space.title(),
                        alias_under,
                        alias_under.lower(),
                        alias_under.upper(),
                    ])

        return self._dedupe(candidates)

    def _display_class(self, cls_name):
        return str(cls_name).replace("_", " ").lower()

    def _display_domain(self, domain_name):
        return str(domain_name).replace("_", " ").lower()

    def _unwrap_llm_data(self, llm_data):
        known_keys = [
            "pacs_domain_class_descriptions",
            "vlcs_domain_class_descriptions",
            "officehome_domain_class_descriptions",
            "office_home_domain_class_descriptions",
            "domainnet_domain_class_descriptions",
        ]
        for key in known_keys:
            if key in llm_data and isinstance(llm_data[key], dict):
                return llm_data[key]

        for key, value in llm_data.items():
            if key.endswith("domain_class_descriptions") and isinstance(value, dict):
                return value

        if len(llm_data) == 1:
            only_value = next(iter(llm_data.values()))
            if isinstance(only_value, dict):
                return only_value

        return llm_data

    def _lookup_descriptions(self, llm_data, domain, cls):
        domain_data = None
        for domain_key in self._name_candidates(domain):
            if domain_key in llm_data:
                domain_data = llm_data[domain_key]
                break

        if isinstance(domain_data, dict):
            for cls_key in self._name_candidates(cls):
                if cls_key in domain_data:
                    descriptions = domain_data[cls_key]
                    if isinstance(descriptions, str):
                        return [descriptions]
                    if isinstance(descriptions, list) and len(descriptions) > 0:
                        return descriptions

        # Fallback: flat structure where llm_data is directly class -> descriptions
        # (e.g., imagenet_prompts_full.json with no domain nesting)
        for cls_key in self._name_candidates(cls):
            if cls_key in llm_data:
                descriptions = llm_data[cls_key]
                if isinstance(descriptions, list) and len(descriptions) > 0:
                    return descriptions
                if isinstance(descriptions, str):
                    return [descriptions]

        return None

    def _fallback_descriptions(self, domain, cls):
        cls_text = self._display_class(cls)
        domain_text = self._display_domain(domain)
        return [
            f"an image of a {cls_text} in {domain_text} style",
            f"a {domain_text} domain image containing a {cls_text}",
        ]

    def _description_anchor_texts(self, domain, cls):
        """
        Build neutral composition anchors for description-level UOT.

        These anchors are used only to estimate the reliability of each raw LLM
        description for the current domain-class pair. Multiple anchors make the
        UOT weights less sensitive to one specific template.
        """
        cls_text = self._display_class(cls)
        domain_text = self._display_domain(domain)

        anchors = [
            self._format_prompt(
                self.comp_prompt_template,
                cls_name=cls,
                domain_name=domain,
            ),
            f"an image of a {cls_text} in {domain_text} style",
            f"a {domain_text} style image of a {cls_text}",
        ]
        return self._dedupe(anchors)

    def _compute_domain_class_targets(self, llm_descriptions_path):
        """
        Compute composition targets T[d, c] from LLM descriptions.

        Mean version:
            T[d, c] = mean_m T(desc_{d,c,m})

        Description-level UOT version:
            T[d, c] = UOTPool_m T(desc_{d,c,m})

        This moves UOT to the raw LLM-description level, where the noise usually
        appears, instead of applying UOT only after descriptions have already
        been averaged into one domain-class prototype.
        """
        with open(llm_descriptions_path, "r", encoding="utf-8") as f:
            llm_data = json.load(f)
        llm_data = self._unwrap_llm_data(llm_data)

        target_features = {}
        missing_entries = []

        for domain in self.domain_names:
            target_features[domain] = {}
            for cls in self.classnames:
                descriptions = self._lookup_descriptions(llm_data, domain, cls)
                if descriptions is None:
                    descriptions = self._fallback_descriptions(domain, cls)
                    missing_entries.append((domain, cls))

                # Encode all raw LLM descriptions for this domain-class pair.
                desc_features = self._encode_frozen_texts(descriptions)

                if self.use_desc_uot_targets and len(descriptions) > 1:
                    # Use composition anchors to softly select reliable descriptions.
                    anchor_texts = self._description_anchor_texts(domain, cls)
                    anchor_features = self._encode_frozen_texts(anchor_texts)
                    target_feat = self._transport_pool(desc_features, anchor_features)
                else:
                    # Fallback to the original mean pooling.
                    target_feat = desc_features.mean(dim=0)
                    target_feat = F.normalize(target_feat.float(), dim=0)

                target_features[domain][cls] = target_feat

        if missing_entries:
            preview = ", ".join(
                f"{domain}/{cls}" for domain, cls in missing_entries[:5]
            )
            print(
                f"Warning: {len(missing_entries)} LLM entries were missing; "
                f"used fallback prompts. Examples: {preview}"
            )

        return target_features

    @torch.no_grad()
    def _encode_frozen_texts(self, texts, chunk_size=64):
        """
        Encode plain text prompts using the frozen CLIP text encoder.

        Args:
            texts: list[str]
            chunk_size: int

        Returns:
            Tensor with shape (len(texts), dim), L2-normalized.
        """
        all_features = []

        for start in range(0, len(texts), chunk_size):
            end = min(start + chunk_size, len(texts))
            tokenized = torch.cat(
                [clip.tokenize(text) for text in texts[start:end]]
            ).to(self.device)

            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=(torch.device(self.device).type == "cuda")
            ):
                feats = self.clip_model.forward_text_ori(tokenized)
                feats = F.normalize(feats.float(), dim=-1)

            all_features.append(feats)

        return torch.cat(all_features, dim=0)

    @torch.no_grad()
    def _uot_weights(self, cost, epsilon=None, tau=None, n_iters=None):
        """
        Compute UOT transport weights with generalized Sinkhorn iterations.

        Args:
            cost: Tensor with shape (n_source, n_target).
                  Smaller cost indicates stronger matching.
            epsilon: entropy regularization strength.
            tau: KL marginal relaxation strength.
            n_iters: number of generalized Sinkhorn iterations.

        Returns:
            Tensor with shape (n_source,), normalized source weights.
        """
        epsilon = self.uot_epsilon if epsilon is None else float(epsilon)
        tau = self.uot_tau if tau is None else float(tau)
        n_iters = self.uot_iters if n_iters is None else int(n_iters)

        cost = cost.float()
        n_source, n_target = cost.shape
        device = cost.device

        a = torch.full((n_source,), 1.0 / n_source, device=device)
        b = torch.full((n_target,), 1.0 / n_target, device=device)

        # UOT relaxation coefficient.
        # tau -> infinity approximates balanced OT; smaller tau allows mass relaxation.
        rho = tau / (tau + epsilon)

        # Gibbs kernel. Clamp avoids numerical underflow producing exact zeros.
        K = torch.exp(-cost / epsilon).clamp_min(1e-12)

        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(n_iters):
            Kv = torch.matmul(K, v).clamp_min(1e-12)
            u = (a / Kv).pow(rho)

            KTu = torch.matmul(K.t(), u).clamp_min(1e-12)
            v = (b / KTu).pow(rho)

        pi = u[:, None] * K * v[None, :]
        source_weights = pi.sum(dim=1)
        source_weights = source_weights / source_weights.sum().clamp_min(1e-12)

        return source_weights

    @torch.no_grad()
    def _transport_pool(self, source_features, anchor_features):
        """
        UOT-based pooling from source features to anchor features.

        Args:
            source_features: Tensor with shape (n_source, dim).
            anchor_features: Tensor with shape (n_target, dim).

        Returns:
            Tensor with shape (dim,), L2-normalized pooled feature.
        """
        source_features = F.normalize(source_features.float(), dim=-1)
        anchor_features = F.normalize(anchor_features.float(), dim=-1)

        # Cosine distance as transport cost.
        cost = 1.0 - torch.matmul(source_features, anchor_features.t())

        weights = self._uot_weights(cost)
        pooled = torch.sum(weights[:, None] * source_features, dim=0)
        pooled = F.normalize(pooled, dim=0)

        return pooled

    def _build_class_level_targets(self):
        """
        Build class-level targets.

        Mean pooling:
            T_cls[c] = mean_d T[d, c]

        UOT pooling:
            T_cls[c] = UOTPool_d T[d, c]

        The UOT anchor is the frozen CLIP feature of:
            "a photo of a {class}"
        """
        target_features_cls = {}

        if not self.use_proto_uot_targets:
            for cls in self.classnames:
                feats = [
                    self.target_features[domain][cls]
                    for domain in self.domain_names
                ]
                feat = torch.stack(feats, dim=0).mean(dim=0)
                feat = F.normalize(feat.float(), dim=0)
                target_features_cls[cls] = feat
            return target_features_cls

        class_anchor_texts = [
            f"an image of a {self._display_class(cls)}"
            for cls in self.classnames
        ]
        class_anchors = self._encode_frozen_texts(class_anchor_texts)

        for cls_idx, cls in enumerate(self.classnames):
            source_feats = torch.stack(
                [
                    self.target_features[domain][cls]
                    for domain in self.domain_names
                ],
                dim=0,
            ).to(self.device)

            anchor_feat = class_anchors[cls_idx:cls_idx + 1]
            feat = self._transport_pool(source_feats, anchor_feat)
            target_features_cls[cls] = feat

        return target_features_cls

    def _build_style_level_targets(self):
        """
        Build style-level targets.

        Mean pooling:
            T_style[d] = mean_c T[d, c]

        UOT pooling:
            T_style[d] = UOTPool_c T[d, c]

        The UOT anchor is the frozen CLIP feature of:
            "an image in {domain} style"
        """
        target_features_style = {}

        if not self.use_proto_uot_targets:
            for domain in self.domain_names:
                feats = [
                    self.target_features[domain][cls]
                    for cls in self.classnames
                ]
                feat = torch.stack(feats, dim=0).mean(dim=0)
                feat = F.normalize(feat.float(), dim=0)
                target_features_style[domain] = feat
            return target_features_style

        style_anchor_texts = [
            self._format_prompt(
                self.style_prompt_template,
                domain_name=domain,
            )
            for domain in self.domain_names
        ]
        style_anchors = self._encode_frozen_texts(style_anchor_texts)

        for domain_idx, domain in enumerate(self.domain_names):
            source_feats = torch.stack(
                [
                    self.target_features[domain][cls]
                    for cls in self.classnames
                ],
                dim=0,
            ).to(self.device)

            anchor_feat = style_anchors[domain_idx:domain_idx + 1]
            feat = self._transport_pool(source_feats, anchor_feat)
            target_features_style[domain] = feat

        return target_features_style

    def _format_prompt(self, template, cls_name=None, domain_name=None):
        cls_text = self._display_class(cls_name) if cls_name is not None else ""
        domain_text = self._display_domain(domain_name) if domain_name is not None else ""
        values = {
            "classname": cls_text,
            "class_name": cls_text,
            "class": cls_text,
            "domain": domain_text,
            "style": domain_text,
        }
        return template.format(**values)

    def _add_ctx_tokens(self, text):
        return f"{'X ' * self.n_ctx}{text}"

    def _encode_prompt_texts(self, prompt_texts, ctx, deep_prompts, chunk_size=64):
        prompts_list = []
        tokenized_prompts_list = []

        for prompt_text in prompt_texts:
            prompt = self._add_ctx_tokens(prompt_text)
            tokenized_prompt = clip.tokenize(prompt).to(self.device)

            with torch.no_grad():
                embedding = self.clip_model.token_embedding(tokenized_prompt)

            embedding[0, 1:1 + self.n_ctx, :] = ctx
            prompts_list.append(embedding)
            tokenized_prompts_list.append(tokenized_prompt)

        prompts = torch.cat(prompts_list, dim=0)
        tokenized_prompts = torch.cat(tokenized_prompts_list, dim=0)

        # Chunked forward to save memory for large-scale datasets (e.g., DomainNet: 344)
        all_features = []
        for start in range(0, len(prompts), chunk_size):
            end = min(start + chunk_size, len(prompts))
            chunk_features = self.text_encoder(
                prompts[start:end],
                tokenized_prompts[start:end],
                deep_prompts,
            )
            all_features.append(chunk_features)

        features = torch.cat(all_features, dim=0)
        features = F.normalize(features, dim=1)
        return features

    def encode_class_features(self, class_indices=None):
        if class_indices is None:
            class_indices = list(range(self.n_cls))

        prompt_texts = [
            self._format_prompt(
                self.class_prompt_template,
                cls_name=self.classnames[cls_idx],
            )
            for cls_idx in class_indices
        ]
        return self._encode_prompt_texts(
            prompt_texts,
            self.class_ctx,
            self.class_deep_prompts,
        )

    def encode_style_features(self, domain_indices=None):
        if domain_indices is None:
            domain_indices = list(range(self.n_domain))

        prompt_texts = [
            self._format_prompt(
                self.style_prompt_template,
                domain_name=self.domain_names[domain_idx],
            )
            for domain_idx in domain_indices
        ]
        return self._encode_prompt_texts(
            prompt_texts,
            self.style_ctx,
            self.style_deep_prompts,
        )

    def encode_composition_features(self, domain_indices=None, class_indices=None):
        if domain_indices is None:
            domain_indices = list(range(self.n_domain))
        if class_indices is None:
            class_indices = list(range(self.n_cls))

        # Batch per domain to avoid OOM on large datasets (e.g., DomainNet: 6*345=2070)
        all_features = []
        for domain_idx in domain_indices:
            domain_name = self.domain_names[domain_idx]
            prompt_texts = []
            for cls_idx in class_indices:
                cls_name = self.classnames[cls_idx]
                prompt_texts.append(
                    self._format_prompt(
                        self.comp_prompt_template,
                        cls_name=cls_name,
                        domain_name=domain_name,
                    )
                )
            features = self._encode_prompt_texts(
                prompt_texts,
                self.comp_ctx,
                self.comp_deep_prompts,
            )
            all_features.append(features)

        features = torch.cat(all_features, dim=0)
        return features.reshape(len(domain_indices), len(class_indices), -1)

    def _class_targets_tensor(self, class_indices):
        feats = [
            self.target_features_cls[self.classnames[cls_idx]]
            for cls_idx in class_indices
        ]
        return torch.stack(feats, dim=0).to(self.device)

    def _style_targets_tensor(self, domain_indices):
        feats = [
            self.target_features_style[self.domain_names[domain_idx]]
            for domain_idx in domain_indices
        ]
        return torch.stack(feats, dim=0).to(self.device)

    def _composition_targets_tensor(self, domain_indices, class_indices):
        feats = []
        for domain_idx in domain_indices:
            domain_name = self.domain_names[domain_idx]
            for cls_idx in class_indices:
                cls_name = self.classnames[cls_idx]
                feats.append(self.target_features[domain_name][cls_name])
        return torch.stack(feats, dim=0).to(self.device)

    def forward(self, class_indices=None, domain_indices=None):
        """
        Text-only training forward pass.

        Returns:
            loss_dict: class/style/composition MSE losses and total loss.
        """
        if class_indices is None:
            class_indices = list(range(self.n_cls))
        if domain_indices is None:
            domain_indices = list(range(self.n_domain))

        class_features = self.encode_class_features(class_indices)
        style_features = self.encode_style_features(domain_indices)
        comp_features = self.encode_composition_features(domain_indices, class_indices)
        comp_features = comp_features.reshape(-1, comp_features.shape[-1])

        class_targets = self._class_targets_tensor(class_indices)
        style_targets = self._style_targets_tensor(domain_indices)
        comp_targets = self._composition_targets_tensor(domain_indices, class_indices)

        loss_class = F.mse_loss(class_features, class_targets)
        loss_style = F.mse_loss(style_features, style_targets)
        loss_comp = F.mse_loss(comp_features, comp_targets)

        loss_mse = (
            self.lambda_class * loss_class
            + self.lambda_style * loss_style
            + self.lambda_comp * loss_comp
        )
        loss_total = self.lambda_mse * loss_mse

        return {
            "loss_total": loss_total,
            "loss_mse": loss_mse,
            "loss_class_mse": loss_class,
            "loss_style_mse": loss_style,
            "loss_comp_mse": loss_comp,
        }

    def encode_all_text_features(self):
        class_features = self.encode_class_features()
        style_features = self.encode_style_features()
        comp_features = self.encode_composition_features()
        return {
            "class": class_features,
            "style": style_features,
            "composition": comp_features,
        }

    def compute_logits_from_cache(self, image_features, text_cache, domain_idx=None):
        logit_scale = self.clip_model.model.logit_scale.exp()
        class_logits = logit_scale * image_features @ text_cache["class"].t()

        mode = self.inference_mode
        if mode == "class_only":
            return class_logits

        comp_features = text_cache["composition"]
        comp_domain_logits = logit_scale * torch.einsum(
            "bf,dcf->bdc", image_features, comp_features
        )

        if mode == "oracle_domain":
            if domain_idx is None:
                raise ValueError("domain_idx is required for oracle_domain inference")
            comp_logits = comp_domain_logits[:, domain_idx, :]
        elif mode == "composition_mean":
            comp_logits = comp_domain_logits.mean(dim=1)
        elif mode == "composition_max":
            comp_logits = comp_domain_logits.max(dim=1).values
        elif mode == "composition_only":
            comp_logits = comp_domain_logits.mean(dim=1)
            return comp_logits
        elif mode == "style_gated":
            style_logits = logit_scale * image_features @ text_cache["style"].t()
            temperature = max(float(self.style_temperature), 1e-6)
            style_weights = F.softmax(style_logits / temperature, dim=1)
            comp_logits = torch.sum(
                style_weights.unsqueeze(-1) * comp_domain_logits,
                dim=1,
            )
        else:
            raise ValueError(f"Unsupported inference mode: {self.inference_mode}")

        logits = (
            self.logit_class_weight * class_logits
            + self.logit_comp_weight * comp_logits
        )
        return logits

    @torch.no_grad()
    def inference(self, domain_idx, image_features):
        with torch.cuda.amp.autocast(enabled=self.precision == "amp"):
            text_cache = self.encode_all_text_features()
            return self.compute_logits_from_cache(image_features, text_cache, domain_idx)


@TRAINER_REGISTRY.register()
class LaMP(TrainerBase):
    """LaMP trainer"""
    
    def __init__(self, cfg):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.cfg = cfg
        
        # Initialize data and build model
        self.init_train_data()
        self.build_model()

    def init_train_data(self):
        """Load class names and domain information"""
        txts_dir_path = self.cfg.TXTS_PATH
        txt_path = os.path.join(txts_dir_path, self.cfg.DATASET.NAME + '.txt')
        
        with open(txt_path, 'r') as f:
            lines = f.read().splitlines()
        self.classnames = list(lines)
        self.num_classes = len(self.classnames)
        
        # Domain names for different datasets
        if self.cfg.DATASET.NAME == 'PACS_SF':
            self.domain_names = ['art_painting', 'cartoon', 'photo', 'sketch']
        elif self.cfg.DATASET.NAME == 'OfficeHomeDG_SF':
            self.domain_names = ['art', 'clipart', 'product', 'real_world']
        elif self.cfg.DATASET.NAME == 'VLCS_SF':
            self.domain_names = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
        elif self.cfg.DATASET.NAME == 'DomainNet_SF':
            self.domain_names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        elif self.cfg.DATASET.NAME == 'ImageNetR_SF':
            self.domain_names = ['rendition']
        elif self.cfg.DATASET.NAME == 'ImageNetS_SF':
            self.domain_names = ['sketch']
        elif self.cfg.DATASET.NAME == 'ImageNetA_SF':
            self.domain_names = ['adversarial']
        elif self.cfg.DATASET.NAME == 'ImageNetV2_SF':
            self.domain_names = ['v2']
        else:
            raise ValueError(f"Unsupported dataset: {self.cfg.DATASET.NAME}")
        
        # Path to LLM descriptions
        self.llm_descriptions_path = self.cfg.TRAINER.LAMP.LLM_DESC_PATH
        assert os.path.exists(self.llm_descriptions_path), \
            f"LLM descriptions file not found: {self.llm_descriptions_path}"
    
    def build_data_loader(self):
        """Build data loader for testing"""
        # For LaMP, directly use the dataset classes
        from dassl.data.datasets import DATASET_REGISTRY
        from dassl.data.data_manager_sf import build_data_loader
        from dassl.data.transforms import build_transform
        
        cfg = self.cfg
        
        # Create minimal train_data dict (only for dataset initialization)
        # We don't actually use this for training, just to satisfy the dataset constructor
        train_data = {
            "classnames": self.classnames,
            "n_cls": len(self.classnames),
            "n_style": 80,  # Default style count
        }
        
        # Build dataset using the registered dataset class
        dataset = DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg, train_data)
        
        # Get test datasets
        test_datasets = dataset.test
        
        # Build transform for test
        tfm_test = build_transform(cfg, is_train=False)
        
        # Build data loaders for each domain
        batch_size = cfg.DATALOADER.TEST.BATCH_SIZE
        num_workers = cfg.DATALOADER.NUM_WORKERS
        self.test_loader = []
        
        for dataset_item in test_datasets:
            loader = build_data_loader(
                cfg,
                data_source=dataset_item,
                batch_size=batch_size,
                is_train=False,
                tfm=tfm_test,
                train_data=None,
                num_workers=num_workers
            )
            self.test_loader.append(loader)
        
        self.lab2cname = dataset.lab2cname
        self.num_classes = dataset.num_classes

    def parse_batch_test(self, batch):
        """Parse test batch"""
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def build_model(self):
        """Build the model"""
        cfg = self.cfg
        print("Building LaMP model")
        
        # Load CLIP backbone
        self.clip_model = build_backbone(
            cfg.MODEL.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            device=self.device,
        )
        self.clip_model.to(self.device)
        
        # Build data loader
        self.build_data_loader()
        
        # Build LaMP model
        self.model = LaMPModel(
            cfg, 
            self.classnames, 
            self.domain_names,
            self.clip_model, 
            self.device,
            self.llm_descriptions_path
        )
        self.model.to(self.device)
        
        print(f"# params: {count_num_param(self.model):,}")
        
        # Only optimize prompt learner parameters
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("LaMP", self.model, self.optim, self.sched)
        
        # Setup precision and AMP scaler
        self.precision = cfg.TRAINER.LAMP.PREC
        if self.precision == "fp32":
            self.clip_model.model.float()
            self.model.float()
            self.clip_model.dtype = self.clip_model.model.dtype
            self.model.text_encoder.dtype = torch.float32
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.precision == "amp"))
        print(f"Training precision: {self.precision}")
        
        # Build evaluator
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = ([0] * len(self.domain_names), 0)

    def train(self):
        """Training loop - Pure text-based training"""
        start_time = time.time()
        self.model.train()
        
        print("Starting LaMP training...")
        print(f"Training for {self.max_epoch} epochs")
        print(f"Classes: {self.classnames}")
        
        # Training loop: one forward pass per epoch (no domain iteration needed)
        for epoch in range(self.max_epoch):
            self.optim.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.precision == "amp"):
                # Forward pass over class, style, and composition branches
                loss_dict = self.model()
                loss_total = loss_dict["loss_total"]
            
            # Backward pass with AMP scaler
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            self.sched.step()
            
            # Print training info
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = self.optim.param_groups[0]["lr"]
                loss_mse = loss_dict["loss_mse"].item()
                loss_class = loss_dict["loss_class_mse"].item()
                loss_style = loss_dict["loss_style_mse"].item()
                loss_comp = loss_dict["loss_comp_mse"].item()
                loss_mse_weighted = self.model.lambda_mse * loss_mse
                
                info = []
                info += [f"epoch [{epoch + 1}/{self.max_epoch}]"]
                info += [f"total_loss {loss_total.item():.4f}"]
                info += [f"mse {loss_mse:.6f}"]
                info += [f"class {loss_class:.6f}"]
                info += [f"style {loss_style:.6f}"]
                info += [f"comp {loss_comp:.6f}"]
                info += [f"mse_w {loss_mse_weighted:.4f}"]
                info += [f"lr {current_lr:.4e}"]
                print(" ".join(info))
        
        # Save model
        self.save_model(epoch, self.output_dir)
        
        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"\nTraining finished!")
        print(f"Elapsed: {elapsed}")

    @torch.no_grad()
    def test(self, split=None):
        """Test the model on target domains"""
        print("=" * 80)
        print("Testing LaMP model")
        print("=" * 80)
        
        self.set_model_mode("eval")
        
        # Pre-compute all branch text features once (optimization!)
        print("\nPre-computing text features...")

        with torch.cuda.amp.autocast(enabled=self.precision == "amp"):
            text_cache = self.model.encode_all_text_features()
        print(f"  Class text features shape = {text_cache['class'].shape}")
        print(f"  Style text features shape = {text_cache['style'].shape}")
        print(f"  Composition text features shape = {text_cache['composition'].shape}")
        print("Text features pre-computed!\n")
        
        # Test on each target domain
        results = []
        domain_names = self.domain_names
        
        for domain_idx, domain_name in enumerate(domain_names):
            print(f"\nTesting on domain: {domain_name}")
            
            # Get test data loader for this domain
            data_loader = self.test_loader[domain_idx]
            
            self.evaluator.reset()
            
            # Performance profiling
            start_time = time.time()
            batch_count = 0
            
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                # Parse batch
                input, label = self.parse_batch_test(batch)
                
                with torch.cuda.amp.autocast(enabled=self.precision == "amp"):
                    # Extract image features using CLIP visual encoder
                    image_features = self.clip_model.forward_image(input)
                    image_features = F.normalize(image_features, dim=1)
                    
                    # Compute logits using pre-computed three-branch text features
                    logits = self.model.compute_logits_from_cache(
                        image_features,
                        text_cache,
                        domain_idx=domain_idx,
                    )
                
                # Evaluate incrementally to avoid GPU memory buildup
                self.evaluator.process(logits, label)
                
                batch_count += 1
            
            # Evaluate
            domain_results = self.evaluator.evaluate()
            results.append(list(domain_results.values())[0])
            
            for k, v in domain_results.items():
                print(f"  {domain_name}/{k}: {v:.2f}")
        
        # Compute mean accuracy
        mean_acc = sum(results) / len(results)
        print(f"\nMean accuracy: {mean_acc:.2f}")
        print(f"Per-domain results: {results}")
        
        # Update best result
        is_best = False
        if self.best_result[-1] < mean_acc:
            self.best_result = (results, mean_acc)
            is_best = True
            print(f"New best result!")
        
        return results, is_best

    def save_model(self, epoch, directory, **kwargs):
        """Save model weights"""
        names = self.get_model_names()
        model_file = f"model.pth.tar-{epoch + 1}"
        
        for name in names:
            model_dir = os.path.join(directory, name)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, model_file)
            
            state = {
                "epoch": epoch + 1,
                "state_dict": self._models[name].state_dict(),
                "val_result": 0.0,  # Placeholder for compatibility with dassl framework
            }
            torch.save(state, model_path)
            print(f"Saved model to {model_path}")
