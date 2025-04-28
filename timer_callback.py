# timer_callback.py
import time, transformers as tr, pynvml, torch

class Timer(tr.TrainerCallback):
    def on_train_begin(self, *a, **k):
        self.t0 = time.perf_counter()
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(
            torch.cuda.current_device())
        self.peak = 0

    def on_step_end(self, *a, **k):
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 2**20  # MB
        self.peak = max(self.peak, mem)

    def on_epoch_end(self, *a, state, **k):
        print(f"epoch-{state.epoch:.0f}  "
              f"{time.perf_counter()-self.t0:.1f} s  "
              f"| peak {self.peak/1024:.1f} GB")

