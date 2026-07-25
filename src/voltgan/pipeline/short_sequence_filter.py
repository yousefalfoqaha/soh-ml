from pathlib import Path

import h5py

from voltgan.pipeline.base import PipelineHandler, SampleContext


class ShortSequenceFilterHandler(PipelineHandler):
    def __init__(self, min_length: int):
        self.min_length = min_length

    @property
    def order(self) -> int:
        return 5

    def handle(self, context: SampleContext) -> SampleContext:
        target_files = context.metadata.get("target_files")
        if not target_files:
            return context

        kept = []
        for file_path in target_files:
            if not file_path.exists():
                continue

            with h5py.File(file_path, "r") as f:
                group = f[file_path.name]
                assert isinstance(group, h5py.Group)
                n_rows = len(next(iter(group.values())))

            if n_rows < self.min_length:
                file_path.unlink()
                print(
                    f"  [filter] removed {file_path.name} "
                    f"({n_rows} < {self.min_length} timesteps)"
                )
            else:
                kept.append(file_path)

        context.metadata["target_files"] = kept
        return context