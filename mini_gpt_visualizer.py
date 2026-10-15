"""Interactive, single-file visual walkthrough of a small decoder-only Transformer.

Install:  uv pip install torch   (or use your existing PyTorch environment)
Run:      python mini_gpt_visualizer.py

Tkinter is included with typical python.org Windows installs.
This is an educational step-by-step *trace*, not a debugger; highlighted source
lines are the mathematical operation currently being explained in the UI.
"""
import math
import tkinter as tk
from tkinter import ttk, messagebox

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. TRAINING DATA
SEED = 123
torch.manual_seed(SEED)
SENTENCES = [
    "the cat chased the mouse",
    "the dog chased the ball",
    "the cat ate the fish",
    "the dog ate the food",
    "the fish ate the blargh",
]
WORDS = sorted({word for sentence in SENTENCES for word in sentence.split()})
stoi = {word: i for i, word in enumerate(WORDS)}
itos = {i: word for word, i in stoi.items()}
VOCAB_SIZE = len(WORDS)
X = torch.tensor([[stoi[w] for w in s.split()[:-1]] for s in SENTENCES])
Y = torch.tensor([[stoi[w] for w in s.split()[1:]] for s in SENTENCES])


# 2. CAUSAL SELF-ATTENTION
class CausalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.trace = {}

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores_raw = Q @ K.transpose(-2, -1)
        scores = scores_raw / math.sqrt(Q.shape[-1])
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        masked_scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(masked_scores, dim=-1)
        output = weights @ V
        self.trace = {name: val.detach().clone() for name, val in {
            "Q": Q, "K": K, "V": V, "QK^T": scores_raw,
            "scaled QK^T": scores, "mask (1=blocked)": mask,
            "masked scores": masked_scores, "attention weights": weights,
            "attention output": output,
        }.items()}
        return output


# 3. MINI LANGUAGE MODEL
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, dim=32, max_len=16):
        super().__init__()
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_len, dim)
        self.attention = CausalAttention(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.output = nn.Linear(dim, vocab_size)
        self.trace = {}

    def forward(self, tokens):
        B, T = tokens.shape
        if T > self.max_len:
            raise ValueError(f"Sequence has {T} tokens; max_len is {self.max_len}")
        positions = torch.arange(T, device=tokens.device)
        token_vectors = self.token_embedding(tokens)
        position_vectors = self.position_embedding(positions)
        x = token_vectors + position_vectors
        embedded = x
        normalized1 = self.norm1(x)
        attention_result = self.attention(normalized1)
        x = x + attention_result
        post_attention = x
        normalized2 = self.norm2(x)
        mlp_result = self.mlp(normalized2)
        x = x + mlp_result
        logits = self.output(x)
        self.trace = {name: val.detach().clone() for name, val in {
            "tokens": tokens, "token embeddings": token_vectors,
            "position embeddings": position_vectors, "combined embeddings": embedded,
            "norm1 output": normalized1, "post-attention + residual": post_attention,
            "norm2 output": normalized2, "MLP output": mlp_result,
            "final representations": x, "logits": logits,
            "probabilities": F.softmax(logits, dim=-1),
        }.items()}
        self.trace.update(self.attention.trace)
        return logits


# A source listing for highlighting, taken from the real functions above.
# Inspect the unmodified real implementation above, not an artificial simulation.
SOURCE = '''# TOKENIZATION + TRAINING PAIRS
WORDS = sorted({word for sentence in SENTENCES for word in sentence.split()})
stoi = {word: i for i, word in enumerate(WORDS)}
X = tensor([[stoi[w] for w in s.split()[:-1]] for s in SENTENCES])
Y = tensor([[stoi[w] for w in s.split()[1:]] for s in SENTENCES])

# EMBEDDINGS: MiniGPT.forward
positions = torch.arange(T, device=tokens.device)
token_vectors = self.token_embedding(tokens)
position_vectors = self.position_embedding(positions)
x = token_vectors + position_vectors

# ATTENTION: CausalAttention.forward
Q = self.query(x)
K = self.key(x)
V = self.value(x)
scores_raw = Q @ K.transpose(-2, -1)
scores = scores_raw / math.sqrt(Q.shape[-1])
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
masked_scores = scores.masked_fill(mask, float("-inf"))
weights = F.softmax(masked_scores, dim=-1)
output = weights @ V

# TRANSFORMER BLOCK: MiniGPT.forward
normalized1 = self.norm1(x)
x = x + attention_result
normalized2 = self.norm2(x)
mlp_result = self.mlp(normalized2)
x = x + mlp_result
logits = self.output(x)

# TRAINING
logits = model(X)
loss = F.cross_entropy(logits.reshape(B*T, V), Y.reshape(B*T))
optimizer.zero_grad()
loss.backward()
optimizer.step()

# INFERENCE
logits = model(prompt_token_ids)
last_logits = logits[:, -1, :]
probs = F.softmax(last_logits, dim=-1)
next_token = torch.argmax(probs, dim=-1)
'''

# All forward matrices are REAL snapshots produced by MiniGPT.forward.
# A frame chooses the matrix and highlighted corresponding source statement.
FRAMES = [
    ("Token IDs & shifted targets", "X = tensor(", "tokens", "The words are mapped to IDs. X is the input, Y contains targets shifted one position."),
    ("Token embeddings", "token_vectors =", "token embeddings", "Look up a trainable 32-number vector for each token ID."),
    ("Position embeddings", "position_vectors =", "position embeddings", "Look up a trainable 32-number vector for each token position."),
    ("Combined embeddings", "x = token_vectors +", "combined embeddings", "Add word and position vectors; shape is 4 × 32 for one sentence."),
    ("Queries Q", "Q = self.query", "Q", "Learned projection W_Q transforms each token into a query vector."),
    ("Keys K", "K = self.key", "K", "Learned projection W_K transforms each token into a key vector."),
    ("Values V", "V = self.value", "V", "Learned projection W_V transforms each token into a value vector."),
    ("Dot-product scores", "scores_raw =", "QK^T", "Q times transposed K produces a 4 × 4 pairwise score matrix."),
    ("Scaled scores", "scores = scores_raw /", "scaled QK^T", "Divide by sqrt(32) to help prevent excessively sharp softmax probabilities."),
    ("Causal mask", "mask = torch.triu", "mask (1=blocked)", "Upper-right entries are blocked: no token can attend to future tokens."),
    ("Masked scores", "masked_scores =", "masked scores", "Replace future-position scores with negative infinity."),
    ("Attention softmax", "weights = F.softmax", "attention weights", "Each row is a distribution across visible token positions (row sums to 1)."),
    ("Attention × Values", "output = weights @ V", "attention output", "Take a weighted sum of value vectors for each token position."),
    ("Residual attention", "x = x + attention_result", "post-attention + residual", "Add the attention result to the original input representation."),
    ("Feed-forward MLP", "mlp_result =", "MLP output", "Each token runs through the same nonlinear 32 → 128 → 32 MLP."),
    ("Vocabulary logits", "logits = self.output(x)", "logits", "Convert 32 features into one score for each of the ten words."),
    ("Next-word probabilities", "probs = F.softmax(last_logits", "probabilities", "Softmax gives probabilities; look at the last position for next-token prediction."),
    ("Cross-entropy loss", "loss = F.cross_entropy", "probabilities", "Compare all 20 predictions against the shifted target words."),
    ("Backpropagation", "loss.backward()", "probabilities", "Gradients estimate how loss changes when each weight changes."),
    ("Optimizer update", "optimizer.step()", "probabilities", "AdamW updates the model's parameters to reduce future prediction errors."),
]


class Visualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("MiniGPT Lab — Learn Attention by Watching It")
        self.root.geometry("1530x925")
        self.root.minsize(1170, 700)
        self.device = torch.device("cpu")  # small teaching model; CPU also makes Tk responsive
        self.model = MiniGPT(VOCAB_SIZE).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.003)
        self.step = 0
        self.frame = 0
        self.auto = False
        self.mode = "training"
        self.latest_loss = None
        self.previous_params = None
        self.delta = None
        self.gradient_norm = None
        self.initial_params = {k: p.detach().clone() for k, p in self.model.named_parameters()}
        self.selected_sentence = tk.IntVar(value=0)
        self.focus_pos = tk.IntVar(value=3)
        self.history = []
        self.matrix_name = tk.StringVar(value="tokens")
        self.prompt = tk.StringVar(value="the cat chased the")
        self.interval = tk.IntVar(value=450)
        self._build()
        self._forward_snapshot()
        self.record_history()
        self.show_frame(0)

    def _build(self):
        style = ttk.Style()
        try: style.theme_use("clam")
        except tk.TclError: pass
        style.configure("Treeview", rowheight=23)
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)
        control = ttk.Frame(main)
        control.pack(fill="x", pady=(0, 8))
        ttk.Label(control, text="MiniGPT Lab", font=("Segoe UI", 16, "bold")).pack(side="left", padx=(0, 20))
        ttk.Button(control, text="◀ Previous stage", command=lambda: self.show_frame(self.frame - 1)).pack(side="left", padx=2)
        ttk.Button(control, text="Next stage ▶", command=self.next_frame).pack(side="left", padx=2)
        ttk.Button(control, text="Train 1 step", command=self.train_one).pack(side="left", padx=2)
        ttk.Button(control, text="Train 50 steps", command=lambda: self.train_many(50)).pack(side="left", padx=2)
        self.play_btn = ttk.Button(control, text="▶ Auto", command=self.toggle_auto)
        self.play_btn.pack(side="left", padx=2)
        ttk.Label(control, text="ms/stage").pack(side="left", padx=(12, 3))
        ttk.Spinbox(control, from_=150, to=3000, increment=100, textvariable=self.interval, width=6).pack(side="left")
        ttk.Button(control, text="Reset model", command=self.reset).pack(side="right")

        # Code at the right. Left: architecture + matrix + learning/predictions.
        split = ttk.Panedwindow(main, orient="horizontal")
        split.pack(fill="both", expand=True)
        left = ttk.Frame(split)
        right = ttk.Frame(split)
        split.add(left, weight=3)
        split.add(right, weight=2)

        top_left = ttk.LabelFrame(left, text="① Transformer flow — select a stage to inspect", padding=7)
        top_left.pack(fill="x", pady=(0, 8))
        self.flow_canvas = tk.Canvas(top_left, height=115, bg="#101b2b", highlightthickness=0)
        self.flow_canvas.pack(fill="x")
        self.flow_canvas.bind("<Button-1>", self.flow_click)
        self.flow_canvas.bind("<Configure>", lambda event: self.draw_flow())
        self.stage_title = ttk.Label(top_left, text="", font=("Segoe UI", 11, "bold"))
        self.stage_title.pack(anchor="w", pady=(6, 1))
        self.explain = ttk.Label(top_left, text="", wraplength=760)
        self.explain.pack(anchor="w")

        focus = ttk.LabelFrame(left, text="② What can the model see? (training teacher-forcing)", padding=7)
        focus.pack(fill="x", pady=(0, 8))
        focus_bar = ttk.Frame(focus)
        focus_bar.pack(fill="x")
        ttk.Label(focus_bar, text="Predict at position:").pack(side="left")
        self.focus_combo = ttk.Combobox(focus_bar, state="readonly", width=8, values=["1", "2", "3", "4"])
        self.focus_combo.current(3)
        self.focus_combo.pack(side="left", padx=5)
        self.focus_combo.bind("<<ComboboxSelected>>", self.on_focus)
        self.focus_summary = ttk.Label(focus_bar, text="", font=("Segoe UI", 10, "bold"))
        self.focus_summary.pack(side="left", padx=10)
        self.token_canvas = tk.Canvas(focus, height=92, bg="#111b2b", highlightthickness=0)
        self.token_canvas.pack(fill="x", pady=(5, 0))
        self.token_canvas.bind("<Configure>", lambda event: self.update_focus())
        matrix_area = ttk.LabelFrame(left, text="③ Live tensor inspector — actual model data", padding=7)
        matrix_area.pack(fill="both", expand=True, pady=(0, 8))
        chooser = ttk.Frame(matrix_area)
        chooser.pack(fill="x", pady=(0, 5))
        ttk.Label(chooser, text="Example:").pack(side="left")
        self.sentence_combo = ttk.Combobox(chooser, state="readonly", width=30,
            values=SENTENCES, textvariable=tk.StringVar(value=SENTENCES[0]))
        self.sentence_combo.current(0)
        self.sentence_combo.pack(side="left", padx=5)
        self.sentence_combo.bind("<<ComboboxSelected>>", self.on_sentence)
        ttk.Label(chooser, text="Matrix:").pack(side="left", padx=(10, 2))
        self.matrix_combo = ttk.Combobox(chooser, state="readonly", textvariable=self.matrix_name, width=25)
        self.matrix_combo["values"] = list(self.model.trace) or ["tokens"]
        self.matrix_combo.pack(side="left", padx=4)
        self.matrix_combo.bind("<<ComboboxSelected>>", lambda event: self.update_matrix())
        self.shape_label = ttk.Label(matrix_area, text="")
        self.shape_label.pack(anchor="w")
        mat_frame = ttk.Frame(matrix_area)
        mat_frame.pack(fill="both", expand=True)
        self.matrix_text = tk.Text(mat_frame, height=12, wrap="none", font=("Consolas", 10), bg="#0f1723", fg="#e4eaf4", insertbackground="white", borderwidth=0)
        self.matrix_text.pack(side="left", fill="both", expand=True)
        scroll_y = ttk.Scrollbar(mat_frame, orient="vertical", command=self.matrix_text.yview)
        scroll_y.pack(side="right", fill="y")
        self.matrix_text.configure(yscrollcommand=scroll_y.set)
        scroll_x = ttk.Scrollbar(matrix_area, orient="horizontal", command=self.matrix_text.xview)
        scroll_x.pack(fill="x")
        self.matrix_text.configure(xscrollcommand=scroll_x.set)
        self.matrix_text.tag_configure("emphasis", foreground="#ffd878")

        bottom = ttk.LabelFrame(left, text="④ Learning + confidence and accuracy", padding=7)
        bottom.pack(fill="x")
        self.metrics = ttk.Label(bottom, text="", font=("Consolas", 10))
        self.metrics.pack(anchor="w")
        self.chart_canvas = tk.Canvas(bottom, height=135, bg="#fbfcff", highlightthickness=1, highlightbackground="#d5dce7")
        self.chart_canvas.pack(fill="x", pady=(4, 4))
        self.chart_canvas.bind("<Configure>", lambda event: self.draw_history())
        self.weight_text = tk.Text(bottom, height=3, wrap="none", font=("Consolas", 10), bg="#f2f5fa")
        self.weight_text.pack(fill="x", pady=(4, 4))
        predict_bar = ttk.Frame(bottom)
        predict_bar.pack(fill="x")
        ttk.Label(predict_bar, text="Prompt:").pack(side="left")
        ttk.Entry(predict_bar, textvariable=self.prompt, width=28).pack(side="left", padx=6)
        ttk.Button(predict_bar, text="Predict next word", command=self.predict).pack(side="left")
        ttk.Button(predict_bar, text="Generate 4 words", command=lambda: self.predict(generate=4)).pack(side="left", padx=5)
        self.prob_canvas = tk.Canvas(bottom, height=126, bg="#ffffff", highlightthickness=0)
        self.prob_canvas.pack(fill="x", pady=(4, 1))
        self.predict_label = ttk.Label(bottom, text="")
        self.predict_label.pack(anchor="w")

        code_group = ttk.LabelFrame(right, text="⑤ Executing source — highlighted conceptual operation", padding=7)
        code_group.pack(fill="both", expand=True)
        ttk.Label(code_group, text="Highlights map to the actual PyTorch forward/train steps; training runs as a complete gradient update.", wraplength=460).pack(anchor="w", pady=(0, 5))
        text_frame = ttk.Frame(code_group)
        text_frame.pack(fill="both", expand=True)
        self.code_text = tk.Text(text_frame, wrap="none", font=("Consolas", 10), bg="#0b1421", fg="#ced8e8", padx=10, pady=10)
        self.code_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.code_text.yview)
        scroll.pack(side="right", fill="y")
        self.code_text.configure(yscrollcommand=scroll.set)
        self.code_text.insert("1.0", SOURCE)
        self.code_text.tag_configure("active", background="#375c45", foreground="#ffffff", font=("Consolas", 10, "bold"))
        self.code_text.tag_configure("heading", foreground="#7ab8fd")
        for n, line in enumerate(SOURCE.splitlines(), 1):
            if line.startswith("#"):
                self.code_text.tag_add("heading", f"{n}.0", f"{n}.end")
        self.code_text.configure(state="disabled")
        self.draw_flow()

    def _forward_snapshot(self, prompt_ids=None):
        self.model.eval()
        with torch.no_grad():
            if prompt_ids is None:
                self.model(X)
            else:
                self.model(prompt_ids)
        self.matrix_combo["values"] = list(self.model.trace)

    def on_sentence(self, event=None):
        if self.mode == "prediction":
            self.mode = "training"
            self._forward_snapshot()
        self.update_matrix()
        self.update_probs()
        self.update_focus()

    def on_focus(self, event=None):
        self.focus_pos.set(self.focus_combo.current())
        if self.mode == "prediction":
            self.mode = "training"
            self._forward_snapshot()
        self.update_focus()
        self.update_probs()
        self.update_matrix()

    def update_focus(self):
        if not hasattr(self, "token_canvas"):
            return
        c = self.token_canvas
        c.delete("all")
        idx = self.sentence_combo.current()
        if idx < 0: idx = 0
        pos = self.focus_pos.get()
        words = SENTENCES[idx].split()
        target = words[pos+1]
        prefix = " ".join(words[:pos+1])
        self.focus_summary.configure(text=f"Visible: '{prefix}'    →    TARGET: '{target}'")
        width = max(c.winfo_width(), 650)
        x0, gap = 12, 8
        cell = (width-24-4*gap)/5
        for j, word in enumerate(words):
            x = x0+j*(cell+gap)
            visible = j <= pos
            target_cell = j == pos+1
            color = "#206a49" if visible else "#773535"
            c.create_rectangle(x, 7, x+cell, 62, fill=color, outline="#f3cc63" if target_cell else "#5c7898", width=3 if target_cell else 1)
            c.create_text(x+cell/2, 23, text=word, fill="white", font=("Segoe UI", 11, "bold"))
            status = "VISIBLE" if visible else ("TARGET · HIDDEN" if target_cell else "FUTURE · HIDDEN")
            c.create_text(x+cell/2, 47, text=status, fill="#e7ecf4", font=("Segoe UI", 8, "bold"))
        c.create_text(12, 77, anchor="w", text="Green = usable context  |  Red = cannot be read at this position  |  Gold outline = answer we're grading", fill="#e7ecf4", font=("Segoe UI", 9))

    def training_stats(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = F.softmax(logits, dim=-1)
            guesses = probs.argmax(dim=-1)
            correct = probs.gather(-1, Y.unsqueeze(-1)).squeeze(-1)
            accuracy = (guesses == Y).float().mean().item()
            avg_conf = correct.mean().item()
            batch_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), Y.reshape(-1)).item()
        return accuracy, avg_conf, batch_loss

    def record_history(self):
        acc, conf, loss = self.training_stats()
        self.history.append((self.step, acc, conf, loss))
        # restore trace to complete training batch (training_stats ran model(X))
        self.draw_history()

    def draw_history(self):
        if not hasattr(self, "chart_canvas"): return
        c = self.chart_canvas
        c.delete("all")
        w = max(c.winfo_width(), 450)
        top, bottom, left, right = 21, 110, 47, w-18
        c.create_text(8, 7, text="Training progress: accuracy and confidence in CORRECT target", anchor="nw", font=("Segoe UI", 9, "bold"), fill="#162d42")
        for v in (0, .5, 1):
            y = bottom-(bottom-top)*v
            c.create_line(left, y, right, y, fill="#e1e5ed")
            c.create_text(left-7, y, text=f"{v:.0%}", anchor="e", font=("Consolas", 8), fill="#4f5c6c")
        if not self.history: return
        samples = self.history
        xmax = max(1, samples[-1][0])
        if len(samples) == 1:
            xx = left
            for k,col in [(1,"#187342"),(2,"#2a62ba")]:
                yy = bottom-(bottom-top)*samples[0][k]
                c.create_oval(xx-3,yy-3,xx+3,yy+3,fill=col,outline="")
        for k,col in [(1,"#187342"),(2,"#2a62ba")]:
            points=[]
            for step,acc,conf,loss in samples:
                xx=left+(right-left)*step/xmax
                yy=bottom-(bottom-top)*(acc if k==1 else conf)
                points.extend((xx,yy))
            if len(points)>=4: c.create_line(*points, fill=col, width=2, smooth=False)
        c.create_text(right, 7, text="■ accuracy    ■ target confidence", anchor="ne", font=("Segoe UI", 9), fill="#187342")
        c.create_text(right, 124, text=f"Step {samples[-1][0]}  |  Accuracy {samples[-1][1]:.1%}  |  Mean target confidence {samples[-1][2]:.1%}", anchor="se", font=("Consolas", 9), fill="#162d42")

    def show_frame(self, frame):
        self.frame = frame % len(FRAMES)
        name, code_needle, tensor, explanation = FRAMES[self.frame]
        self.stage_title.configure(text=f"Stage {self.frame + 1}/{len(FRAMES)} — {name}")
        self.explain.configure(text=explanation)
        if tensor in self.model.trace:
            self.matrix_name.set(tensor)
        self.update_matrix()
        self.highlight_code(code_needle)
        self.draw_flow()
        self.update_metrics()
        self.update_probs()
        self.update_focus()
        self.draw_history()

    def next_frame(self):
        next_i = (self.frame + 1) % len(FRAMES)
        # At the optimizer stage the model makes exactly one parameter update.
        if FRAMES[next_i][0] == "Optimizer update":
            self.train_one(show_stage=False)
        self.show_frame(next_i)

    def highlight_code(self, needle):
        self.code_text.configure(state="normal")
        self.code_text.tag_remove("active", "1.0", "end")
        for index, line in enumerate(SOURCE.splitlines(), 1):
            if needle in line:
                self.code_text.tag_add("active", f"{index}.0", f"{index}.end")
                self.code_text.see(f"{index}.0")
                break
        self.code_text.configure(state="disabled")

    def update_matrix(self):
        key = self.matrix_name.get()
        tensor = self.model.trace.get(key)
        if tensor is None:
            return
        if tensor.ndim >= 3:
            idx = 0 if self.mode == "prediction" else self.sentence_combo.current()
            idx = max(0, min(idx, tensor.shape[0] - 1))
            array = tensor[idx]
        else:
            array = tensor
        shape = tuple(tensor.shape)
        self.shape_label.configure(text=f"{key} | Full tensor shape {shape} | Displayed shape {tuple(array.shape)} | values rounded to 3 decimals")
        self.matrix_text.configure(state="normal")
        self.matrix_text.delete("1.0", "end")
        if key == "tokens":
            raw = array.reshape(-1).tolist()
            self.matrix_text.insert("end", "IDs: " + str(raw) + "\nWords: " + " ".join(itos.get(int(n), "?") for n in raw) + "\n")
            if self.mode == "training":
                y = Y[self.sentence_combo.current()]
                self.matrix_text.insert("end", "Targets: " + str(y.tolist()) + "\nNext words: " + " ".join(itos[int(n)] for n in y) + "\n")
        else:
            if array.dtype == torch.bool:
                array = array.int()
            with torch.no_grad():
                if array.ndim == 1:
                    txt = " ".join(f"{v.item():8.3f}" for v in array)
                elif array.ndim == 2:
                    txt = "\n".join((">>>" if self.mode == "training" and i == self.focus_pos.get() else "   ") + f"row {i:2}: " + " ".join(f"{v.item():8.3f}" for v in row) for i, row in enumerate(array))
                else:
                    txt = str(array)
            self.matrix_text.insert("end", txt)
            if key == "attention weights":
                self.matrix_text.insert("end", "\n\nEach row sums to 1. Last row = where the final token attends.")
            if key == "logits":
                self.matrix_text.insert("end", "\n\nColumns are in sorted vocabulary order: " + ", ".join(WORDS))
        self.matrix_text.configure(state="disabled")

    def draw_flow(self):
        c = self.flow_canvas
        c.delete("all")
        width = max(c.winfo_width(), 780)
        nodes = ["Token IDs", "Embeddings", "Q · K · V", "Attention", "+ Residual", "MLP", "Logits", "Softmax"]
        stage_to_node = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7]
        active = stage_to_node[self.frame]
        margin, gap = 10, 10
        n = len(nodes)
        box_w = (width - margin * 2 - gap * (n - 1)) / n
        for i, label in enumerate(nodes):
            x = margin + i * (box_w + gap)
            fill = "#277b56" if i == active else "#22364f"
            c.create_rectangle(x, 24, x + box_w, 83, fill=fill, outline="#7392b3", width=2 if i == active else 1)
            c.create_text(x + box_w / 2, 54, text=label, fill="white", font=("Segoe UI", 9, "bold"), width=box_w - 5)
            if i < n - 1:
                c.create_line(x + box_w + 1, 54, x + box_w + gap - 1, 54, arrow="last", fill="#9ab6d8", width=2)
        c.create_text(width / 2, 113, text="INPUT: the cat chased the   →   TARGET: mouse     |     1 attention head · 1 block · 32 dimensions", fill="#c4d7ee", font=("Consolas", 10))

    def flow_click(self, event):
        w = max(self.flow_canvas.winfo_width(), 780)
        n = 8
        i = min(n - 1, max(0, int((event.x - 10) / ((w - 20) / n))))
        stages = [0, 1, 4, 7, 13, 14, 15, 16]
        self.show_frame(stages[i])

    def train_one(self, show_stage=True):
        self.auto = False if show_stage else self.auto
        self.model.train()
        self.previous_params = {k: p.detach().clone() for k, p in self.model.named_parameters()}
        logits = self.model(X)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B*T, V), Y.reshape(B*T))
        self.optimizer.zero_grad()
        loss.backward()
        self.gradient_norm = math.sqrt(sum(p.grad.detach().square().sum().item() for p in self.model.parameters() if p.grad is not None))
        self.optimizer.step()
        self.latest_loss = loss.item()
        self.step += 1
        diffs = [(p.detach() - self.previous_params[k]).square().sum().item() for k, p in self.model.named_parameters()]
        self.delta = math.sqrt(sum(diffs))
        self.mode = "training"
        self._forward_snapshot()
        self.record_history()
        if show_stage:
            self.show_frame(19)
        else:
            self.update_metrics()

    def train_many(self, n):
        self.auto = False
        self.play_btn.configure(text="▶ Auto")
        for _ in range(n):
            self.train_one(show_stage=False)
        self.show_frame(19)

    def update_metrics(self):
        acc, conf, current_loss = self.history[-1][1:] if self.history else (0, 0, 0)
        self.metrics.configure(text=f"Accuracy (20 predictions): {acc:.1%}     Mean target confidence: {conf:.1%}     Current loss: {current_loss:.4f}\nUpdates: {self.step:4d}    Loss (last update): {self.latest_loss if self.latest_loss is not None else 'not trained'}    "
            f"Gradient L2: {self.gradient_norm if self.gradient_norm is not None else '—'}    Update L2: {self.delta if self.delta is not None else '—'}")
        lines = []
        for name in ["token_embedding.weight", "attention.query.weight", "attention.key.weight", "attention.value.weight", "mlp.0.weight", "output.weight"]:
            p = dict(self.model.named_parameters())[name].detach()
            baseline = self.initial_params[name]
            change = torch.linalg.vector_norm(p - baseline).item()
            lines.append(f"{name:28s}  norm={torch.linalg.vector_norm(p).item():8.3f}  Δ from init={change:8.4f}  first 3={p.reshape(-1)[:3].tolist()}")
        self.weight_text.configure(state="normal")
        self.weight_text.delete("1.0", "end")
        self.weight_text.insert("1.0", "\n".join(lines))
        self.weight_text.configure(state="disabled")

    def update_probs(self):
        tr = self.model.trace
        if "probabilities" not in tr: return
        idx = 0 if self.mode == "prediction" else self.sentence_combo.current()
        pos = -1 if self.mode == "prediction" else self.focus_pos.get()
        probs = tr["probabilities"][idx, pos]
        top = sorted([(itos[i], probs[i].item(), i) for i in range(VOCAB_SIZE)], key=lambda t: -t[1])
        target = None if self.mode == "prediction" else int(Y[idx, pos])
        displayed = top[:5]
        if target is not None and all(tid != target for _,_,tid in displayed):
            displayed[-1] = (itos[target], probs[target].item(), target)
        c = self.prob_canvas
        c.delete("all")
        w = max(c.winfo_width(), 600)
        max_bar = max(w - 290, 150)
        for row, (word, prob, tid) in enumerate(displayed):
            y = 7 + row * 22
            c.create_text(5, y + 7, anchor="w", text=f"{word:9s}", font=("Consolas", 10, "bold" if tid==target else "normal"))
            fill = "#20a36b" if tid == target else "#457dcb"
            c.create_rectangle(110, y, 110 + max_bar * prob, y + 16, fill=fill, outline="")
            c.create_text(120 + max_bar, y + 7, anchor="w", text=f"{prob:6.1%}" + ("  ← correct answer" if tid==target else ""), font=("Consolas", 10))
        if target is not None:
            guess = top[0][2]
            right = guess == target
            self.predict_label.configure(text=f"Position {pos+1}: expected '{itos[target]}' | predicted '{itos[guess]}' | {'✓ CORRECT' if right else '✗ INCORRECT'} | confidence in target {probs[target].item():.1%} | confidence in guess {top[0][1]:.1%}")

    def predict(self, generate=1):
        words = self.prompt.get().strip().lower().split()
        if not words:
            messagebox.showerror("Empty prompt", "Please enter at least one known vocabulary word.")
            return
        unknown = [w for w in words if w not in stoi]
        if unknown:
            messagebox.showerror("Unknown words", f"Words not in this model's vocabulary: {', '.join(unknown)}\n\nKnown words: {', '.join(WORDS)}")
            return
        if len(words) + generate - 1 > self.model.max_len:
            messagebox.showerror("Too long", f"Maximum context length is {self.model.max_len} words.")
            return
        for _ in range(generate):
            tokens = torch.tensor([[stoi[w] for w in words]])
            self._forward_snapshot(tokens)
            p = self.model.trace["probabilities"][0, -1]
            next_word = itos[int(torch.argmax(p))]
            words.append(next_word)
        self.mode = "prediction"
        self.matrix_name.set("probabilities")
        self.show_frame(16)
        self.predict_label.configure(text="Generated: " + " ".join(words) + " | shown matrix corresponds to context BEFORE final generated word")

    def toggle_auto(self):
        self.auto = not self.auto
        self.play_btn.configure(text="❚❚ Pause" if self.auto else "▶ Auto")
        if self.auto:
            self.auto_tick()

    def auto_tick(self):
        if not self.auto: return
        self.next_frame()
        if self.auto:
            try: interval = max(150, int(self.interval.get()))
            except (ValueError, tk.TclError): interval = 450
            self.root.after(interval, self.auto_tick)

    def reset(self):
        self.auto = False
        self.play_btn.configure(text="▶ Auto")
        torch.manual_seed(SEED)
        self.model = MiniGPT(VOCAB_SIZE).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.003)
        self.initial_params = {k: p.detach().clone() for k, p in self.model.named_parameters()}
        self.step, self.latest_loss, self.gradient_norm, self.delta = 0, None, None, None
        self.mode = "training"
        self.history = []
        self.focus_pos.set(3)
        self.focus_combo.current(3)
        self._forward_snapshot()
        self.record_history()
        self.show_frame(0)


def main():
    root = tk.Tk()
    app = Visualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
