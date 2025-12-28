import time

class ETAEstimator:
    def __init__(self, alpha=0.2, warmup=20):
        self.alpha = alpha
        self.warmup = warmup
        self.avg_step_time = None
        self.last_time = None
        self.steps = 0

    def tick(self):
        now = time.time()
        if self.last_time is None:
            self.last_time = now
            return

        dt = now - self.last_time
        self.last_time = now
        self.steps += 1

        if self.avg_step_time is None:
            self.avg_step_time = dt
        else:
            self.avg_step_time = self.alpha * dt + (1 - self.alpha) * self.avg_step_time

    def eta(self, steps_left):
        if self.avg_step_time is None or self.steps < self.warmup:
            return None
        return self.avg_step_time * steps_left


def format_eta(seconds):
    if seconds is None:
        return "estimating..."
    if seconds < 0.2:
        return "finishing..."
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    return f"{seconds/3600:.1f} h"

