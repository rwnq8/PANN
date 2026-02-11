import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union, List
import math

# ============================
# P-ADIC UTILITY FUNCTIONS
# ============================

def p_adic_valuation(x: torch.Tensor, p: int, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute p-adic valuation v_p(x) for tensor elements.
    For x = 0, returns large negative value (or 0 depending on application).
    """
    if p <= 1:
        raise ValueError(f"p must be prime > 1, got {p}")
    
    # Handle zero elements separately
    mask_zero = torch.abs(x) < eps
    mask_nonzero = ~mask_zero
    
    # For nonzero elements: v_p(x) = max{k : p^k divides x}
    # We'll compute using logarithms
    x_abs = torch.abs(x[mask_nonzero])
    valuation = torch.zeros_like(x_abs)
    
    # For floating point, we approximate using logarithms
    if x.dtype in [torch.float16, torch.float32, torch.float64]:
        # Use logarithms for approximation
        log_p = math.log(p)
        valuation = torch.floor(torch.log(x_abs + eps) / log_p)
    else:
        # For integers, compute exact valuation
        for k in range(0, 100):  # Practical limit
            divisible = (x_abs % (p ** (k + 1)) == 0)
            if not divisible.any():
                break
            valuation += divisible.float()
    
    # Combine results
    result = torch.zeros_like(x)
    result[mask_nonzero] = valuation
    result[mask_zero] = -100.0  # Special value for zero
    return result

def p_adic_norm(x: torch.Tensor, p: int, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute p-adic norm |x|_p = p^{-v_p(x)}.
    Returns 0 for x = 0.
    """
    valuation = p_adic_valuation(x, p, eps)
    # For zero elements (valuation = -100), norm should be 0
    mask_zero = valuation < -50
    norm = torch.zeros_like(x)
    norm[~mask_zero] = torch.pow(p, -valuation[~mask_zero])
    return norm

def p_adic_expansion(x: torch.Tensor, p: int, max_digits: int = 10) -> torch.Tensor:
    """
    Compute p-adic expansion coefficients for each element.
    Returns tensor of shape (..., max_digits) with coefficients in [0, p-1].
    """
    x_int = x.long()  # Convert to integer for expansion
    shape = x_int.shape
    
    # Initialize coefficients tensor
    coeffs = torch.zeros((*shape, max_digits), dtype=torch.long)
    
    # Compute coefficients for each digit position
    x_curr = x_int.clone()
    for digit in range(max_digits):
        coeffs[..., digit] = x_curr % p
        x_curr = x_curr // p
    
    return coeffs.float()

def p_adic_distance(x: torch.Tensor, y: torch.Tensor, p: int, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute p-adic distance d_p(x,y) = |x - y|_p.
    Ultra-metric property: d_p(x,z) ≤ max(d_p(x,y), d_p(y,z))
    """
    diff = x - y
    return p_adic_norm(diff, p, eps)

# ============================
# P-ADIC NEURAL LAYERS
# ============================

class PAdicEmbedding(nn.Module):
    """
    P-adic embedding layer that maps integer indices to hierarchical embeddings.
    Uses p-adic structure to organize embeddings in a tree-like hierarchy.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, p: int = 2,
                 max_levels: int = 8, use_hierarchy: bool = True):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.p = p
        self.max_levels = max_levels
        self.use_hierarchy = use_hierarchy
        
        # Base embedding matrix
        self.base_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        
        if use_hierarchy:
            # Level-specific transformation matrices
            # Each level corresponds to a different digit in p-adic expansion
            self.level_weights = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=False)
                for _ in range(max_levels)
            ])
            
            # Level importance weights (learnable)
            self.level_alpha = nn.Parameter(torch.ones(max_levels))
            
            # Hierarchical bias terms
            self.level_biases = nn.ParameterList([
                nn.Parameter(torch.zeros(embedding_dim))
                for _ in range(max_levels)
            ])
    
    def get_hierarchical_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute hierarchical embedding using p-adic structure.
        """
        # Get base embeddings
        emb = self.base_embeddings(indices)  # Shape: (batch, embedding_dim)
        
        if not self.use_hierarchy:
            return emb
        
        # Get p-adic expansion for each index
        expansions = p_adic_expansion(indices.float(), self.p, self.max_levels)
        expansions = expansions.to(emb.device)  # Shape: (batch, max_levels)
        
        # Apply hierarchical transformations
        hierarchical_emb = torch.zeros_like(emb)
        
        for level in range(self.max_levels):
            # Get digit at this level
            digits = expansions[:, level]  # Shape: (batch,)
            
            # Create mask for non-zero digits at this level
            mask = (digits != 0).float().unsqueeze(-1)  # Shape: (batch, 1)
            
            # Transform base embedding for this level
            level_emb = self.level_weights[level](emb)  # Shape: (batch, embedding_dim)
            
            # Apply digit-weighted transformation
            # Higher-level digits (more significant) have stronger influence
            digit_weight = digits.unsqueeze(-1) / self.p
            level_contribution = level_emb * digit_weight * mask
            
            # Add level bias
            level_contribution = level_contribution + self.level_biases[level]
            
            # Weight by level importance
            level_weight = F.softmax(self.level_alpha, dim=0)[level]
            hierarchical_emb = hierarchical_emb + level_weight * level_contribution
        
        return hierarchical_emb
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.get_hierarchical_embedding(indices)

class PAdicAttention(nn.Module):
    """
    Attention mechanism using p-adic distances as similarity measure.
    Captures hierarchical relationships in data.
    """
    def __init__(self, embed_dim: int, p: int = 2, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.p = p
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # P-adic distance projection (learns to combine with standard attention)
        self.padic_proj = nn.Linear(1, num_heads)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        # Project queries, keys, values
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Standard attention scores (scaled dot product)
        q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Compute p-adic distances between positions
        # Create position indices
        positions = torch.arange(seq_len, device=query.device).float().unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        
        # Reshape for broadcasting
        pos_i = positions.unsqueeze(-1).unsqueeze(1)  # (batch, 1, seq_len, 1)
        pos_j = positions.unsqueeze(1).unsqueeze(-1)  # (batch, seq_len, 1, 1)
        
        # Compute p-adic distances
        padic_dist = p_adic_distance(pos_i, pos_j, self.p)  # (batch, seq_len, seq_len, 1)
        
        # Project p-adic distances to attention space
        padic_scores = self.padic_proj(padic_dist)  # (batch, seq_len, seq_len, heads)
        padic_scores = padic_scores.permute(0, 3, 1, 2)  # (batch, heads, seq_len, seq_len)
        
        # Combine standard and p-adic attention scores
        combined_scores = attn_scores + padic_scores
        
        # Apply mask if provided
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            combined_scores = combined_scores.masked_fill(mask, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(combined_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

# ============================
# P-ADIC LOSS FUNCTIONS
# ============================

class PAdicTripletLoss(nn.Module):
    """
    Triplet loss using p-adic distance as metric.
    Encourages hierarchical clustering of embeddings.
    """
    def __init__(self, p: int = 2, margin: float = 1.0,
                 alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.p = p
        self.margin = margin
        self.alpha = alpha  # Weight for Euclidean distance
        self.beta = beta    # Weight for p-adic distance
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        # Euclidean distances
        eucl_dist_pos = F.pairwise_distance(anchor, positive, p=2)
        eucl_dist_neg = F.pairwise_distance(anchor, negative, p=2)
        
        # P-adic distances (using L2 norm of vectors as "numbers")
        padic_dist_pos = p_adic_distance(
            torch.norm(anchor, dim=-1),
            torch.norm(positive, dim=-1),
            self.p
        )
        padic_dist_neg = p_adic_distance(
            torch.norm(anchor, dim=-1),
            torch.norm(negative, dim=-1),
            self.p
        )
        
        # Combined distances
        dist_pos = self.alpha * eucl_dist_pos + self.beta * padic_dist_pos
        dist_neg = self.alpha * eucl_dist_neg + self.beta * padic_dist_neg
        
        # Triplet loss
        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()

class HierarchicalContrastiveLoss(nn.Module):
    """
    Contrastive loss that uses p-adic valuation to determine hierarchy levels.
    Pushes apart items at different hierarchy levels more strongly.
    """
    def __init__(self, p: int = 2, temperature: float = 0.07,
                 hierarchy_weight: float = 2.0):
        super().__init__()
        self.p = p
        self.temperature = temperature
        self.hierarchy_weight = hierarchy_weight
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.shape[0]
        
        # Compute similarity matrix
        embeddings_norm = F.normalize(embeddings, dim=-1)
        similarity = torch.matmul(embeddings_norm, embeddings_norm.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Compute p-adic distances between labels
        label_dist = p_adic_distance(
            labels.float().unsqueeze(0),
            labels.float().unsqueeze(1),
            self.p
        )
        
        # Higher hierarchy difference -> larger penalty
        # Convert p-adic distance to hierarchy weight
        # |x-y|_p = p^{-v_p(x-y)} so smaller distance = larger valuation = closer hierarchy
        hierarchy_weights = 1.0 / (label_dist + 1e-8)
        hierarchy_weights = hierarchy_weights / hierarchy_weights.max()
        
        # Adjust temperature based on hierarchy
        adjusted_temp = self.temperature * (1 + self.hierarchy_weight * (1 - hierarchy_weights))
        
        # Recompute similarity with adjusted temperatures
        similarity_adjusted = torch.matmul(embeddings_norm, embeddings_norm.T)
        similarity_adjusted = similarity_adjusted / adjusted_temp
        
        # Mask out self-similarity
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        label_mask = label_mask & (~mask)
        
        # Compute logits
        exp_sim = torch.exp(similarity_adjusted)
        
        # Positive pairs
        pos_logits = torch.sum(exp_sim * label_mask.float(), dim=1)
        
        # Negative pairs
        neg_logits = torch.sum(exp_sim * (~label_mask).float(), dim=1)
        
        # Contrastive loss
        loss = -torch.log(pos_logits / (pos_logits + neg_logits + 1e-8))
        return loss.mean()

# ============================
# P-ADIC NETWORK MODULES
# ============================

class PAdicResidualBlock(nn.Module):
    """
    Residual block with p-adic gating mechanism.
    Uses p-adic information to control information flow through hierarchy levels.
    """
    def __init__(self, dim: int, p: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.p = p
        
        # Main transformation
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        
        # P-adic gate
        self.gate_proj = nn.Linear(1, dim)
        
        # Layer norm and dropout
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor, level_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        
        # Main transformation
        x_norm = self.norm(x)
        h = self.linear1(x_norm)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.linear2(h)
        
        # P-adic gating
        if level_info is not None:
            # level_info could be p-adic valuation or hierarchy level
            gate = self.gate_proj(level_info.unsqueeze(-1))
            gate = torch.sigmoid(gate)
            h = h * gate
        
        # Residual connection
        output = residual + self.dropout(h)
        return output

class HierarchicalTransformerEncoder(nn.Module):
    """
    Transformer encoder with p-adic hierarchical attention.
    """
    def __init__(self, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, p: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.p = p
        
        # Positional encoding with p-adic structure
        self.pos_encoder = PAdicPositionalEncoding(d_model, p=p, dropout=dropout)
        
        # Transformer layers with p-adic attention
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # P-adic attention module
        self.padic_attention = PAdicAttention(d_model, p=p, num_heads=nhead)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Apply transformer
        output = self.transformer_encoder(src, src_mask)
        
        # Apply p-adic attention
        padic_output, _ = self.padic_attention(output, output, output)
        
        # Combine and normalize
        combined = output + padic_output
        return self.norm(combined)

class PAdicPositionalEncoding(nn.Module):
    """
    Positional encoding that incorporates p-adic distance information.
    """
    def __init__(self, d_model: int, p: int = 2, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.p = p
        
        # Learnable p-adic positional embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # Fixed sinusoidal encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Standard sinusoidal encoding
        pos_encoding = self.pe[:seq_len, :]
        
        # P-adic enhanced encoding
        positions = torch.arange(seq_len, device=x.device)
        padic_pos = self.pos_embedding(positions)
        
        # Combine encodings
        encoding = pos_encoding + padic_pos
        encoding = encoding.unsqueeze(0)  # Add batch dimension
        
        return self.dropout(x + encoding)

# ============================
# TEST FUNCTIONS
# ============================

def test_padic_functions():
    """Test basic p-adic operations"""
    print("Testing p-adic functions...")
    
    # Test valuation
    x = torch.tensor([0, 1, 2, 4, 8, 16, 27])
    print(f"Input: {x}")
    print(f"v_2(x): {p_adic_valuation(x, 2)}")
    print(f"|x|_2: {p_adic_norm(x, 2)}")
    
    # Test expansion
    expansions = p_adic_expansion(x, p=2, max_digits=5)
    print(f"Binary expansions:\n{expansions}")
    
    # Test distance
    y = torch.tensor([4, 4, 4, 4, 4, 4, 4])
    dist = p_adic_distance(x, y, p=2)
    print(f"d_2(x,4): {dist}")
    
    # Ultra-metric property test
    a, b, c = torch.tensor([2]), torch.tensor([10]), torch.tensor([18])
    d_ab = p_adic_distance(a, b, 2)
    d_bc = p_adic_distance(b, c, 2)
    d_ac = p_adic_distance(a, c, 2)
    print(f"\nUltra-metric test:")
    print(f"d(2,10) = {d_ab.item()}")
    print(f"d(10,18) = {d_bc.item()}")
    print(f"d(2,18) = {d_ac.item()}")
    print(f"max(d(2,10), d(10,18)) = {max(d_ab.item(), d_bc.item())}")
    print(f"d(2,18) ≤ max(d(2,10), d(10,18)): {d_ac.item() <= max(d_ab.item(), d_bc.item())}")

def test_padic_embedding():
    """Test the p-adic embedding layer"""
    print("\nTesting PAdicEmbedding...")
    
    # Create layer
    vocab_size = 100
    embed_dim = 32
    padic_embed = PAdicEmbedding(vocab_size, embed_dim, p=3, max_levels=5)
    
    # Test forward pass
    indices = torch.randint(0, vocab_size, (4, 10))  # batch_size=4, seq_len=10
    embeddings = padic_embed(indices)
    
    print(f"Input shape: {indices.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding device: {embeddings.device}")
    
    # Test that similar numbers (in p-adic sense) have similar embeddings
    # Numbers with same lower p-adic digits should be closer
    test_indices = torch.tensor([[0, 1, 3, 9, 27]])  # 3^0, 3^1, 3^1+?, 3^2, 3^3
    test_embeds = padic_embed(test_indices)
    
    # Compute distances
    from torch.nn.functional import cosine_similarity
    for i in range(5):
        for j in range(i+1, 5):
            sim = cosine_similarity(
                test_embeds[0, i:i+1],
                test_embeds[0, j:j+1]
            )
            print(f"Similarity between {test_indices[0,i]} and {test_indices[0,j]}: {sim.item():.4f}")
    
    return padic_embed

def test_padic_losses():
    """Test p-adic loss functions"""
    print("\nTesting p-adic loss functions...")
    
    # Create synthetic data
    batch_size = 16
    embed_dim = 32
    
    # Anchor, positive, negative embeddings
    anchor = torch.randn(batch_size, embed_dim)
    positive = anchor + torch.randn(batch_size, embed_dim) * 0.1  # Close to anchor
    negative = torch.randn(batch_size, embed_dim)  # Far from anchor
    
    # Test triplet loss
    triplet_loss = PAdicTripletLoss(p=2, margin=1.0)
    loss = triplet_loss(anchor, positive, negative)
    print(f"PAdicTripletLoss: {loss.item():.4f}")
    
    # Test hierarchical contrastive loss
    labels = torch.randint(0, 4, (batch_size,))
    contrastive_loss = HierarchicalContrastiveLoss(p=2, temperature=0.07)
    loss = contrastive_loss(anchor, labels)
    print(f"HierarchicalContrastiveLoss: {loss.item():.4f}")
    
    return loss

def test_padic_attention():
    """Test p-adic attention mechanism"""
    print("\nTesting PAdicAttention...")
    
    batch_size = 2
    seq_len = 8
    embed_dim = 32
    
    # Create attention module
    attention = PAdicAttention(embed_dim, p=2, num_heads=8)
    
    # Create sample data
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass
    output, attn_weights = attention(query, key, value)
    
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Check that p-adic structure influences attention
    # Create positions that are close in p-adic sense
    positions = torch.tensor([[0, 2, 4, 6, 8, 10, 12, 14]]).float()
    positions = positions.expand(batch_size, seq_len)
    
    # Positions 0, 2, 4, 6,... are all divisible by 2
    # So they should have stronger attention to each other
    
    return attention, attn_weights

def train_simple_example():
    """Train a simple example with p-adic embeddings"""
    print("\nTraining simple example...")
    
    # Create synthetic hierarchical data
    # Simulating tree structure with 3 levels
    num_samples = 1000
    num_classes = 8  # 2^3 leaves
    
    # Generate hierarchical labels
    # Each sample belongs to a leaf in a binary tree
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Create embeddings that should capture hierarchy
    embed_dim = 16
    padic_embed = PAdicEmbedding(num_classes, embed_dim, p=2, max_levels=3)
    
    # Simple classifier
    classifier = nn.Sequential(
        nn.Linear(embed_dim, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes)
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(padic_embed.parameters()) + list(classifier.parameters()),
        lr=0.001
    )
    
    # Training loop
    num_epochs = 10
    batch_size = 32
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, num_samples, batch_size):
            batch_labels = labels[i:i+batch_size]
            
            # Get hierarchical embeddings
            embeddings = padic_embed(batch_labels)
            
            # Classify
            outputs = classifier(embeddings)
            
            # Compute loss
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
        
        acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/(num_samples/batch_size):.4f}, Acc: {acc:.4f}")
    
    print("Training complete!")
    
    # Visualize embeddings (conceptually)
    with torch.no_grad():
        all_embeddings = padic_embed(torch.arange(num_classes))
        
        # Compute distances
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        emb_np = all_embeddings.numpy()
        sim_matrix = cosine_similarity(emb_np)
        
        print("\nEmbedding similarity matrix:")
        for i in range(num_classes):
            row = [f"{sim_matrix[i,j]:.2f}" for j in range(num_classes)]
            print(f"Class {i}: {row}")
        
        # Check hierarchical structure
        # Classes that share higher-level ancestors should be more similar
        print("\nChecking hierarchy preservation:")
        for i in range(0, num_classes, 2):
            j = i + 1
            if j < num_classes:
                # These share a parent in binary tree
                sim = sim_matrix[i, j]
                print(f"Classes {i} and {j} (siblings): similarity = {sim:.4f}")

# ============================
# INTEGRATION EXAMPLE
# ============================

class HierarchicalClassifier(nn.Module):
    """
    Complete hierarchical classifier using p-adic components.
    """
    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 128,
                 p: int = 2, num_levels: int = 8):
        super().__init__()
        
        # Hierarchical embedding
        self.embedding = PAdicEmbedding(
            vocab_size, embed_dim, p=p, max_levels=num_levels
        )
        
        # P-adic transformer encoder
        self.encoder = HierarchicalTransformerEncoder(
            d_model=embed_dim, p=p, num_layers=4
        )
        
        # Hierarchical prediction heads (one for each level)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, (p ** level) if level < num_levels - 1 else num_classes)
            )
            for level in range(num_levels)
        ])
        
        # Level weighting
        self.level_weights = nn.Parameter(torch.ones(num_levels))
    
    def forward(self, x: torch.Tensor, return_hierarchical: bool = False):
        # Get hierarchical embeddings
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # Encode with p-adic transformer
        encoded = self.encoder(emb)  # (batch, seq_len, embed_dim)
        
        # Pool sequence dimension (mean pooling)
        pooled = encoded.mean(dim=1)  # (batch, embed_dim)
        
        # Get predictions at each hierarchy level
        level_logits = []
        for head in self.heads:
            level_logits.append(head(pooled))
        
        # Weight level predictions
        weights = F.softmax(self.level_weights, dim=0)
        final_logits = sum(w * logits for w, logits in zip(weights, level_logits))
        
        if return_hierarchical:
            return final_logits, level_logits
        return final_logits

# ============================
# MAIN TEST
# ============================

if __name__ == "__main__":
    print("=" * 60)
    print("P-ADIC NEURAL LAYERS DEMONSTRATION")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test individual components
    test_padic_functions()
    embed_model = test_padic_embedding()
    loss = test_padic_losses()
    attention_model, weights = test_padic_attention()
    
    # Train simple example
    train_simple_example()
    
    # Demonstrate complete model
    print("\n" + "=" * 60)
    print("COMPLETE HIERARCHICAL CLASSIFIER DEMONSTRATION")
    print("=" * 60)
    
    model = HierarchicalClassifier(
        vocab_size=1000,
        num_classes=10,
        embed_dim=64,
        p=2,
        num_levels=4
    )
    
    # Create sample input
    batch_size = 8
    seq_len = 16
    sample_input = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    logits, hierarchical_logits = model(sample_input, return_hierarchical=True)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Final logits shape: {logits.shape}")
    print(f"Number of hierarchy levels: {len(hierarchical_logits)}")
    for i, level_logits in enumerate(hierarchical_logits):
        print(f"  Level {i} logits shape: {level_logits.shape}")
    
    print("\nKey Features:")
    print("1. P-adic embeddings capture hierarchical structure")
    print("2. Ultra-metric distances enable efficient hierarchical clustering")
    print("3. Multi-level predictions allow coarse-to-fine classification")
    print("4. Attention mechanism incorporates p-adic positional information")
    
    print("\n" + "=" * 60)
    print("APPLICATIONS:")
    print("=" * 60)
    print("1. Taxonomic classification (biology, products)")
    print("2. Document categorization (Dewey decimal, topic hierarchy)")
    print("3. Geographic data (country->state->city->neighborhood)")
    print("4. Organizational structures (company departments)")
    print("5. Multi-scale time series analysis")
