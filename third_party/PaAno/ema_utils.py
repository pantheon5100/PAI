"""EMA teacher utilities for PaAno."""

import copy
import math

import torch


class EMATeacher:
    """Exponential moving average wrapper for a PaAno encoder."""

    def __init__(self, student, tau_start=0.996, tau_end=0.999, total_steps=200):
        self.teacher = copy.deepcopy(student)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.total_steps = max(1, int(total_steps))
        self.step_count = 0

    def _current_tau(self):
        t = min(self.step_count / self.total_steps, 1.0)
        return self.tau_end - (self.tau_end - self.tau_start) * (1 + math.cos(math.pi * t)) / 2

    @torch.no_grad()
    def update(self, student):
        self.step_count += 1
        tau = self._current_tau()
        for student_param, teacher_param in zip(student.parameters(), self.teacher.parameters()):
            teacher_param.data.mul_(tau).add_(student_param.data, alpha=1.0 - tau)
        for student_buffer, teacher_buffer in zip(student.buffers(), self.teacher.buffers()):
            teacher_buffer.copy_(student_buffer)

    @torch.no_grad()
    def embedding(self, x):
        self.teacher.eval()
        return self.teacher.embedding(x)

    @torch.no_grad()
    def projection(self, h):
        self.teacher.eval()
        return self.teacher.projection(h)
