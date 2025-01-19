"""
LunarCache™ - Advanced Memory System for Large Model
Implements a sophisticated caching system with Short-Term Visual Memory (STVM)
and basic pattern management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass
from torch.cuda.amp import autocast

@dataclass
class CacheConfig:
    """Configuration for LunarCache system."""
    stvm_size: int = 1024          # Size of short-term memory cache
    pattern_dim: int = 2048        # Dimension of pattern embeddings
    cache_threshold: float = 0.85   # Similarity threshold for cache hits
    max_patterns: int = 100        # Maximum patterns to return
    temperature: float = 0.1       # Temperature for similarity scoring
    device: str = 'cuda'           # Device to store cache
    enable_logging: bool = True    # Enable detailed logging
    priority_threshold: float = 0.9 # Threshold for high-priority patterns
    cleanup_frequency: int = 1000   # Frequency of cache cleanup

class STVMCache(nn.Module):
    """Short-Term Visual Memory Cache implementation."""
    
    def __init__(self, config: CacheConfig):
        super().__init__()
        self.config = config
        
        # Initialize cache tensors
        self.patterns = nn.Parameter(
            torch.zeros(config.stvm_size, config.pattern_dim),
            requires_grad=False
        )
        self.pattern_scores = nn.Parameter(
            torch.zeros(config.stvm_size),
            requires_grad=False
        )
        self.pattern_priorities = nn.Parameter(
            torch.zeros(config.stvm_size),
            requires_grad=False
        )
        self.usage_counts = nn.Parameter(
            torch.zeros(config.stvm_size),
            requires_grad=False
        )
        
        # Initialize metadata
        self.metadata = {
            'total_hits': 0,
            'total_updates': 0,
            'pattern_lifetimes': torch.zeros(config.stvm_size),
            'last_access': torch.zeros(config.stvm_size),
            'high_priority_hits': 0
        }
        
        self.register_buffer('empty_slots', torch.ones(config.stvm_size, dtype=torch.bool))
        self.current_step = 0
    
    @torch.no_grad()
    def update(self, pattern: torch.Tensor, score: float, priority: float = 0.0) -> None:
        """Update cache with new pattern."""
        self.current_step += 1
        
        # Memory cleanup if needed
        if self.current_step % self.config.cleanup_frequency == 0:
            self._cleanup_memory()
        
        # Find slot for new pattern
        if torch.any(self.empty_slots):
            slot_idx = torch.where(self.empty_slots)[0][0]
            self.empty_slots[slot_idx] = False
        else:
            # Replace lowest scoring non-priority pattern
            non_priority_mask = self.pattern_priorities < self.config.priority_threshold
            if torch.any(non_priority_mask):
                scores_temp = self.pattern_scores.clone()
                scores_temp[~non_priority_mask] = float('inf')
                slot_idx = torch.argmin(scores_temp)
            else:
                slot_idx = torch.argmin(self.pattern_scores)
        
        # Update pattern and metadata
        self.patterns[slot_idx] = pattern.detach()
        self.pattern_scores[slot_idx] = score
        self.pattern_priorities[slot_idx] = priority
        self.usage_counts[slot_idx] = 0
        self.metadata['last_access'][slot_idx] = self.current_step
        self.metadata['total_updates'] += 1
        
        if self.config.enable_logging:
            logging.debug(f"Updated cache slot {slot_idx} with score {score:.4f} and priority {priority:.4f}")
    
    def _cleanup_memory(self):
        """Perform memory cleanup by removing old, unused patterns."""
        if self.config.enable_logging:
            logging.info("Performing cache cleanup...")
        
        # Calculate age of patterns
        age = self.current_step - self.metadata['last_access']
        old_unused = (age > self.config.cleanup_frequency) & (self.usage_counts < 5) & (self.pattern_priorities < self.config.priority_threshold)
        
        # Reset old unused slots
        self.empty_slots[old_unused] = True
        if self.config.enable_logging:
            logging.info(f"Cleaned up {old_unused.sum().item()} unused patterns")
    
    @torch.no_grad()
    def query(self, query_pattern: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query cache for similar patterns."""
        # Compute similarities
        similarities = F.cosine_similarity(
            query_pattern.unsqueeze(0),
            self.patterns[~self.empty_slots],
            dim=1
        )
        
        # Apply temperature scaling
        similarities = similarities / self.config.temperature
        
        # Get top patterns
        if len(similarities) > 0:
            top_k = min(len(similarities), self.config.max_patterns)
            top_similarities, top_indices = torch.topk(similarities, k=top_k)
            
            # Filter by threshold
            mask = top_similarities >= self.config.cache_threshold
            top_similarities = top_similarities[mask]
            top_indices = top_indices[mask]
            
            if len(top_indices) > 0:
                # Update metadata
                self.metadata['total_hits'] += 1
                self.usage_counts[top_indices] += 1
                self.metadata['last_access'][top_indices] = self.current_step
                
                # Get patterns
                patterns = self.patterns[~self.empty_slots][top_indices]
                return patterns, top_similarities
        
        return torch.empty(0, device=query_pattern.device), torch.empty(0, device=query_pattern.device)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        stats = {
            'total_patterns': (~self.empty_slots).sum().item(),
            'total_hits': self.metadata['total_hits'],
            'total_updates': self.metadata['total_updates'],
            'avg_score': self.pattern_scores[~self.empty_slots].mean().item(),
            'avg_usage': self.usage_counts[~self.empty_slots].mean().item(),
            'cache_utilization': (~self.empty_slots).float().mean().item()
        }
        return stats

class LunarCache(nn.Module):
    """Main LunarCache system implementation."""
    
    def __init__(self, config: CacheConfig):
        super().__init__()
        self.config = config
        
        # Initialize STVM
        self.stvm = STVMCache(config)
        
        # Pattern scoring network (simplified)
        self.pattern_scorer = nn.Sequential(
            nn.Linear(config.pattern_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # Output score and priority
            nn.Sigmoid()
        )
        
        # Initialize stats
        self.stats = {
            'queries': 0, 
            'cache_hits': 0, 
            'updates': 0,
            'high_priority_hits': 0
        }
    
    def score_pattern(self, pattern: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score a pattern for cache importance and priority."""
        with autocast(enabled=True):
            scores = self.pattern_scorer(pattern)
            return scores[:, 0], scores[:, 1]  # score, priority
    
    def forward(self, query_patterns: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process patterns through cache system."""
        batch_size = query_patterns.size(0)
        self.stats['queries'] += batch_size
        
        # Query cache for each pattern
        enhanced_patterns = []
        cache_info = []
        
        for pattern in query_patterns:
            # Query cache
            cached_patterns, similarities = self.stvm.query(pattern)
            
            if len(cached_patterns) > 0:
                # Enhance pattern with cache hits
                self.stats['cache_hits'] += 1
                weights = F.softmax(similarities / self.config.temperature, dim=0)
                enhanced = torch.sum(cached_patterns * weights.unsqueeze(1), dim=0)
                enhanced_patterns.append(enhanced)
                
                cache_info.append({
                    'hit': True,
                    'num_patterns': len(cached_patterns),
                    'max_similarity': similarities.max().item()
                })
            else:
                enhanced_patterns.append(pattern)
                cache_info.append({'hit': False})
            
            # Update cache if pattern is important
            score, priority = self.score_pattern(pattern)
            if score > self.config.cache_threshold:
                self.stvm.update(pattern, score.item(), priority.item())
                self.stats['updates'] += 1
                if priority > self.config.priority_threshold:
                    self.stats['high_priority_hits'] += 1
        
        # Stack enhanced patterns
        enhanced_patterns = torch.stack(enhanced_patterns)
        
        # Return enhanced patterns and cache info
        return enhanced_patterns, {
            'cache_info': cache_info,
            'stats': self.stats,
            'stvm_stats': self.stvm.get_stats()
        }
    
    def save_state(self, path: str) -> None:
        """Save cache state to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'patterns': self.stvm.patterns.cpu(),
            'scores': self.stvm.pattern_scores.cpu(),
            'priorities': self.stvm.pattern_priorities.cpu(),
            'usage_counts': self.stvm.usage_counts.cpu(),
            'metadata': self.stvm.metadata,
            'stats': self.stats,
            'config': self.config.__dict__
        }
        
        torch.save(state, save_path)
        
        if self.config.enable_logging:
            logging.info(f"Saved cache state to {save_path}")
    
    def load_state(self, path: str) -> None:
        """Load cache state from disk."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"No cache state found at {load_path}")
        
        state = torch.load(load_path)
        
        # Restore state
        self.stvm.patterns.data = state['patterns'].to(self.config.device)
        self.stvm.pattern_scores.data = state['scores'].to(self.config.device)
        self.stvm.pattern_priorities.data = state['priorities'].to(self.config.device)
        self.stvm.usage_counts.data = state['usage_counts'].to(self.config.device)
        self.stvm.metadata = state['metadata']
        self.stats = state['stats']
        
        if self.config.enable_logging:
            logging.info(f"Loaded cache state from {load_path}")
            logging.info(f"Cache stats: {self.stats}") 