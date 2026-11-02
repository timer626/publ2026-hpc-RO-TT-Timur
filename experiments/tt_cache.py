"""TT-compressed KV cache adapter for the Transformers Cache API."""

from __future__ import annotations

import time
from statistics import fmean
from typing import Any

import numpy as np
import torch
import tt
from transformers.cache_utils import Cache, CacheLayerMixin


class Int8TTBlock:
    """Symmetrically quantized TT cores with float32 scales."""

    def __init__(self, cores: list[np.ndarray], per_row: bool = False) -> None:
        self.cores: list[np.ndarray] = []
        self.scales: list[np.float32 | np.ndarray] = []
        for core in cores:
            if per_row:
                max_abs = np.max(np.abs(core), axis=(1, 2), keepdims=True)
                scale = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(
                    np.float32
                )
            else:
                max_abs = float(np.max(np.abs(core)))
                scale = np.float32(max_abs / 127.0 if max_abs else 1.0)
            self.cores.append(
                np.clip(np.rint(core / scale), -127, 127).astype(np.int8)
            )
            self.scales.append(scale)
        self.r = np.asarray(
            [self.cores[0].shape[0]]
            + [core.shape[-1] for core in self.cores],
            dtype=np.int64,
        )

    @property
    def storage_bytes(self) -> int:
        return sum(core.nbytes for core in self.cores) + sum(
            scale.nbytes for scale in self.scales
        )

    def full(self) -> np.ndarray:
        cores = [
            core.astype(np.float32) * scale
            for core, scale in zip(self.cores, self.scales)
        ]
        return tt.vector(cores).full()


class TTCompressedCacheLayer(CacheLayerMixin):
    """Store KV in TT blocks and return dense tensors to attention."""

    is_sliding = False
    is_croppable = False

    def __init__(
        self,
        epsilon: float,
        block_size: int,
        core_quantization: str = "none",
        prefill_block_size: int | None = None,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.block_size = block_size
        self.prefill_block_size = prefill_block_size or block_size
        self.core_quantization = core_quantization
        self.key_blocks: list[Any] = []
        self.value_blocks: list[Any] = []
        self.tail_keys: torch.Tensor | None = None
        self.tail_values: torch.Tensor | None = None
        self.sequence_length = 0
        self.tt_decomposition_seconds = 0.0
        self.tt_reconstruction_seconds = 0.0
        self.device_to_host_seconds = 0.0
        self.host_to_device_seconds = 0.0
        self.block_reconstruction_seconds = 0.0
        self.materialization_calls = 0
        self.relative_errors: list[float] = []
        self.dense_key_numel = 0

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        if key_states.ndim != 4 or key_states.shape != value_states.shape:
            raise ValueError(
                "Ожидались K/V тензоры одинаковой формы "
                "[batch, heads, seq, dim]"
            )
        if key_states.shape[-2] == 0:
            raise ValueError("KV cache должен содержать хотя бы один токен")
        if (
            key_states.device != value_states.device
            or key_states.dtype != value_states.dtype
        ):
            raise ValueError("K и V должны иметь одинаковые dtype и device")
        if key_states.shape[0] != 1:
            raise ValueError(
                "TT baseline поддерживает генерацию по одному примеру за вызов"
            )
        self.state_shape = (
            key_states.shape[0],
            key_states.shape[1],
            key_states.shape[3],
        )
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.batch_size = key_states.shape[0]
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        materialize: bool = True,
        **kwargs: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            reconstructed_keys = None
            reconstructed_values = None
            for start in range(0, key_states.shape[-2], self.prefill_block_size):
                key_chunk = key_states[
                    ..., start : start + self.prefill_block_size, :
                ]
                value_chunk = value_states[
                    ..., start : start + self.prefill_block_size, :
                ]
                key_reconstruction, key_error = self._compress_block(key_chunk)
                value_reconstruction, value_error = self._compress_block(value_chunk)
                self.key_blocks.append(key_reconstruction[0])
                self.value_blocks.append(value_reconstruction[0])
                reconstructed_keys = key_reconstruction[1]
                reconstructed_values = value_reconstruction[1]
                self.relative_errors.extend((key_error, value_error))
        else:
            if key_states.shape[0] != self.batch_size:
                raise ValueError("Размер batch не совпадает с инициализированным KV cache")
            if key_states.shape != value_states.shape:
                raise ValueError("Формы новых K и V состояний не совпадают")
            if (
                key_states.shape[1] != self.state_shape[1]
                or key_states.shape[3] != self.state_shape[2]
            ):
                raise ValueError(
                    "Число KV heads или head_dim изменились внутри генерации"
                )
            if key_states.dtype != self.dtype or key_states.device != self.device:
                raise ValueError("dtype или device новых KV состояний изменился")
            if value_states.device != self.device or value_states.dtype != self.dtype:
                raise ValueError("dtype или device новых V состояний изменился")

            self._append_to_dense_tail(key_states, value_states)
            reconstructed_keys = None
            reconstructed_values = None
            tail_keys = self.tail_keys
            tail_values = self.tail_values
            if tail_keys is None or tail_values is None:
                raise RuntimeError("Не удалось добавить новые состояния в KV cache")

            while tail_keys.shape[-2] >= self.block_size:
                key_chunk = tail_keys[..., : self.block_size, :]
                value_chunk = tail_values[..., : self.block_size, :]
                key_reconstruction, key_error = self._compress_block(key_chunk)
                value_reconstruction, value_error = self._compress_block(value_chunk)
                self.key_blocks.append(key_reconstruction[0])
                self.value_blocks.append(value_reconstruction[0])
                reconstructed_keys = key_reconstruction[1]
                reconstructed_values = value_reconstruction[1]
                self.relative_errors.extend((key_error, value_error))

                remaining_keys = tail_keys[..., self.block_size :, :]
                remaining_values = tail_values[..., self.block_size :, :]
                if remaining_keys.shape[-2] == 0:
                    self.tail_keys = None
                    self.tail_values = None
                else:
                    self.tail_keys = remaining_keys
                    self.tail_values = remaining_values
                tail_keys = self.tail_keys
                tail_values = self.tail_values
                if tail_keys is None or tail_values is None:
                    break

        self.sequence_length += key_states.shape[-2]
        self.dense_key_numel += key_states.numel()

        if not materialize:
            return None, None

        return (
            self._materialize(self.key_blocks, self.tail_keys, reconstructed_keys),
            self._materialize(self.value_blocks, self.tail_values, reconstructed_values),
        )

    def _append_to_dense_tail(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        cpu_keys = key_states.detach().to(device="cpu")
        cpu_values = value_states.detach().to(device="cpu")

        if self.tail_keys is None:
            self.tail_keys = cpu_keys
            self.tail_values = cpu_values
            return

        self.tail_keys = torch.cat((self.tail_keys, cpu_keys), dim=-2)
        self.tail_values = torch.cat((self.tail_values, cpu_values), dim=-2)

    def _compress_block(
        self,
        states: torch.Tensor,
    ) -> tuple[tuple[Any, np.ndarray], float]:
        transfer_started = time.perf_counter()
        states_array = (
            states.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
        )
        self.device_to_host_seconds += time.perf_counter() - transfer_started

        started_at = time.perf_counter()
        compressed = tt.vector(states_array, eps=self.epsilon)
        block = (
            compressed
            if self.core_quantization == "none"
            else Int8TTBlock(
                compressed.cores,
                per_row=self.core_quantization == "int8_row",
            )
        )
        self.tt_decomposition_seconds += time.perf_counter() - started_at

        started_at = time.perf_counter()
        reconstructed = block.full()
        reconstruction_elapsed = time.perf_counter() - started_at
        self.tt_reconstruction_seconds += reconstruction_elapsed
        self.block_reconstruction_seconds += reconstruction_elapsed

        original_norm = float(np.linalg.norm(states_array))
        reconstruction_error = float(
            np.linalg.norm(states_array - reconstructed)
            / max(original_norm, np.finfo(np.float32).tiny)
        )
        return (block, reconstructed), reconstruction_error

    def _materialize(
        self,
        blocks: list[Any],
        dense_tail: torch.Tensor | None,
        latest_reconstruction: np.ndarray | None,
    ) -> torch.Tensor:
        started_at = time.perf_counter()
        self.materialization_calls += 1
        dense_blocks = []
        last_block_index = len(blocks) - 1

        for block_index, block in enumerate(blocks):
            if block_index == last_block_index and latest_reconstruction is not None:
                dense_blocks.append(latest_reconstruction)
            else:
                dense_blocks.append(block.full())

        if dense_tail is not None:
            dense_blocks.append(
                dense_tail.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
            )

        dense_states = np.concatenate(dense_blocks, axis=-2)
        dense_tensor = torch.from_numpy(np.ascontiguousarray(dense_states))
        host_to_device_started = time.perf_counter()
        result = dense_tensor.to(device=self.device, dtype=self.dtype)
        self.host_to_device_seconds += time.perf_counter() - host_to_device_started
        self.tt_reconstruction_seconds += time.perf_counter() - started_at
        return result

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        return self.sequence_length

    def get_max_length(self) -> int:
        return -1

    def reset(self) -> None:
        self.key_blocks.clear()
        self.value_blocks.clear()
        self.tail_keys = None
        self.tail_values = None
        self.sequence_length = 0
        self.tt_decomposition_seconds = 0.0
        self.tt_reconstruction_seconds = 0.0
        self.device_to_host_seconds = 0.0
        self.host_to_device_seconds = 0.0
        self.block_reconstruction_seconds = 0.0
        self.materialization_calls = 0
        self.relative_errors.clear()
        self.dense_key_numel = 0
        self.is_initialized = False

    def crop(self, tokens_to_remove: int) -> None:
        if tokens_to_remove == 0:
            return
        raise NotImplementedError("TT cache не поддерживает crop для этой версии runner")

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if beam_idx.numel() == 1 and int(beam_idx[0]) == 0:
            return
        raise NotImplementedError("TT cache runner рассчитан на один beam")

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats == 1:
            return
        raise NotImplementedError("TT cache runner рассчитан на один пример")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if indices.numel() == 1 and int(indices[0]) == 0:
            return
        raise NotImplementedError("TT cache runner рассчитан на один пример")

    def get_statistics(self) -> dict[str, int | float]:
        tt_core_bytes = sum(
            int(
                block.storage_bytes
                if isinstance(block, Int8TTBlock)
                else sum(core.nbytes for core in block.cores)
            )
            for block in (*self.key_blocks, *self.value_blocks)
        )
        dense_tail_bytes = 0
        if self.tail_keys is not None:
            dense_tail_bytes = (
                self.tail_keys.numel() + self.tail_values.numel()
            ) * self.tail_keys.element_size()

        dense_equivalent_bytes = (
            self.dense_key_numel
            * 2
            * torch.empty((), dtype=self.dtype).element_size()
        )
        stored_bytes = tt_core_bytes + dense_tail_bytes
        tt_ranks = [
            int(np.max(block.r))
            for block in (*self.key_blocks, *self.value_blocks)
        ]

        return {
            "sequence_length": self.sequence_length,
            "tt_blocks": len(self.key_blocks),
            "tt_core_bytes": tt_core_bytes,
            "dense_tail_bytes": dense_tail_bytes,
            "dense_equivalent_bytes": dense_equivalent_bytes,
            "stored_bytes": stored_bytes,
            "compression_ratio": (
                dense_equivalent_bytes / stored_bytes if stored_bytes else 0.0
            ),
            "max_tt_rank": max(tt_ranks, default=0),
            "mean_relative_reconstruction_error": (
                fmean(self.relative_errors) if self.relative_errors else 0.0
            ),
            "max_relative_reconstruction_error": max(
                self.relative_errors,
                default=0.0,
            ),
            "tt_decomposition_seconds": self.tt_decomposition_seconds,
            "tt_reconstruction_seconds": self.tt_reconstruction_seconds,
            "device_to_host_seconds": self.device_to_host_seconds,
            "host_to_device_seconds": self.host_to_device_seconds,
            "block_reconstruction_seconds": self.block_reconstruction_seconds,
            "materialization_calls": self.materialization_calls,
        }


class TTCompressedCache(Cache):
    """Transformers cache made of TT-compressed KV layers."""

    def __init__(
        self,
        config: Any,
        epsilon: float,
        block_size: int,
        core_quantization: str = "none",
        prefill_block_size: int | None = None,
    ) -> None:
        decoder_config = config.get_text_config(decoder=True)
        num_layers = decoder_config.num_hidden_layers
        if num_layers <= 0:
            raise ValueError("Число слоев модели должно быть больше нуля")
        layer_types = getattr(decoder_config, "layer_types", None)
        if layer_types and any(
            layer_type != "full_attention" for layer_type in layer_types
        ):
            raise ValueError(
                "Эта версия TT cache поддерживает только модели с full attention"
            )
        if not 0.0 < epsilon < 1.0:
            raise ValueError("TT epsilon должен быть между 0 и 1")
        if block_size <= 0:
            raise ValueError("Размер TT-блока должен быть больше нуля")
        if prefill_block_size is not None and prefill_block_size <= 0:
            raise ValueError("Размер TT-блока prefill должен быть больше нуля")
        if core_quantization not in {"none", "int8", "int8_row"}:
            raise ValueError("Поддерживается none, int8 или int8_row для TT-ядер")

        layers = [
            TTCompressedCacheLayer(
                epsilon, block_size, core_quantization, prefill_block_size
            )
            for _ in range(num_layers)
        ]
        super().__init__(layers=layers)

    def get_statistics(self) -> dict[str, int | float]:
        layer_statistics = [
            layer.get_statistics()
            for layer in self.layers
            if layer.is_initialized
        ]
        if not layer_statistics:
            raise RuntimeError("KV cache не был заполнен во время генерации")

        dense_equivalent_bytes = sum(
            int(layer["dense_equivalent_bytes"])
            for layer in layer_statistics
        )
        tt_core_bytes = sum(
            int(layer["tt_core_bytes"]) for layer in layer_statistics
        )
        dense_tail_bytes = sum(
            int(layer["dense_tail_bytes"])
            for layer in layer_statistics
        )
        stored_bytes = tt_core_bytes + dense_tail_bytes
        reconstruction_errors = [
            float(layer["mean_relative_reconstruction_error"])
            for layer in layer_statistics
        ]

        return {
            "layers": len(layer_statistics),
            "sequence_length": max(
                int(layer["sequence_length"]) for layer in layer_statistics
            ),
            "tt_blocks": sum(
                int(layer["tt_blocks"]) for layer in layer_statistics
            ),
            "tt_core_bytes": tt_core_bytes,
            "dense_tail_bytes": dense_tail_bytes,
            "dense_equivalent_bytes": dense_equivalent_bytes,
            "stored_bytes": stored_bytes,
            "compression_ratio": (
                dense_equivalent_bytes / stored_bytes if stored_bytes else 0.0
            ),
            "max_tt_rank": max(
                int(layer["max_tt_rank"]) for layer in layer_statistics
            ),
            "mean_relative_reconstruction_error": fmean(reconstruction_errors),
            "max_relative_reconstruction_error": max(
                float(layer["max_relative_reconstruction_error"])
                for layer in layer_statistics
            ),
            "tt_decomposition_seconds": sum(
                float(layer["tt_decomposition_seconds"])
                for layer in layer_statistics
            ),
            "tt_reconstruction_seconds": sum(
                float(layer["tt_reconstruction_seconds"])
                for layer in layer_statistics
            ),
            **{
                name: sum(float(layer[name]) for layer in layer_statistics)
                for name in (
                    "device_to_host_seconds",
                    "host_to_device_seconds",
                    "block_reconstruction_seconds",
                    "materialization_calls",
                )
            },
        }
