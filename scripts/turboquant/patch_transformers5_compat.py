from pathlib import Path

path = Path("/workspace/tools/turboquant-kv/turboquant/hf_cache.py")
text = path.read_text()

# Transformers 5.x requires get_max_length().
if "def get_max_length(self)" not in text:
    old = """    def get_max_cache_shape(self) -> int:
        return -1
"""
    new = """    def get_max_length(self) -> int:
        return -1

    def get_max_cache_shape(self) -> int:
        return -1
"""
    if old not in text:
        raise RuntimeError("get_max_cache_shape patch location not found")
    text = text.replace(old, new, 1)

# Transformers 5.15 passes integer query_length to get_mask_sizes().
old_mask = """    def get_mask_sizes(self, cache_position: torch.Tensor) -> Tuple[int, int]:
        query_length = cache_position.shape[0]
        kv_offset = 0
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset
"""

new_mask = """    def get_mask_sizes(self, query_length) -> Tuple[int, int]:
        if isinstance(query_length, torch.Tensor):
            if query_length.ndim == 0:
                query_length = int(query_length.item())
            else:
                query_length = int(query_length.shape[0])
        else:
            query_length = int(query_length)

        kv_offset = 0
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset
"""

if old_mask in text:
    text = text.replace(old_mask, new_mask, 1)

path.write_text(text)

print("TurboQuant Transformers 5.x compatibility patches applied.")
