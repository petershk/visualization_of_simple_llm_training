"""Inside the Learning Machine — an interactive, real-PyTorch MiniGPT exhibit.

Run: uv run python inside_learning_machine.py
Requires PyTorch and a Python installation with Tkinter.
No matplotlib, web server, or external assets needed.
"""
import os
import shutil
import subprocess
import sys

# If torch isn't available (e.g. launched with system Python instead of the
# project venv), re-launch this script via `uv run`, which installs the
# dependencies from pyproject.toml automatically.
try:
    import torch  # noqa: F401
except ImportError:
    uv = shutil.which("uv")
    if uv is None or os.environ.get("ILM_BOOTSTRAPPED"):
        sys.exit("PyTorch is missing. Install uv (https://docs.astral.sh/uv/) and run:\n"
                 "    uv run python inside_learning_machine.py")
    print("PyTorch not found in this Python; launching via uv (first run may download packages)...")
    env = dict(os.environ, ILM_BOOTSTRAPPED="1")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.exit(subprocess.call([uv, "run", "--project", project_dir, "python",
                              os.path.abspath(__file__), *sys.argv[1:]], env=env))

import math
import re
import threading
import time
import urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 123
BOS = "<BOS>"
EOS = "<EOS>"
MAX_CONTENT = 24
HEADS = 4
torch.set_num_threads(1)
DEFAULT_TEXT = """the cat chased the mouse
the dog chased the ball
the cat ate the fish
the dog ate the food
the fish ate the blargh"""
SONNETS_URLS = (
    'https://www.gutenberg.org/cache/epub/1041/pg1041.txt',
    'https://www.gutenberg.org/ebooks/1041.txt.utf-8',
    'https://www.gutenberg.org/files/1041/1041-0.txt',
)
CORPUS_CACHE = Path(__file__).with_name('shakespeare_sonnets_154.txt')


def roman(n):
    out = ''
    for value, symbol in [(100,'C'), (90,'XC'), (50,'L'), (40,'XL'), (10,'X'),
                          (9,'IX'), (5,'V'), (4,'IV'), (1,'I')]:
        while n >= value:
            out += symbol
            n -= value
    return out


def parse_sonnets(text):
    """Extract I–CLIV, excluding Gutenberg's license and non-poem material."""
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\ufeff','')
    start = re.search(r'\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*', text, re.I)
    if start: text = text[start.end():]
    end = re.search(r'\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK', text, re.I)
    if end: text = text[:end.start()]
    lines = text.splitlines()
    results, active, expected = [], [], 1
    for line in lines:
        word = line.strip()
        if word.upper() == roman(expected) and expected <= 154:
            if active:
                results.append('\n'.join(active))
                active=[]
            expected += 1
            continue
        if expected > 1 and expected <= 155 and word:
            active.append(word)
        if expected == 155 and len(active) >= 14:
            break
    if active: results.append('\n'.join(active))
    # The count also protects against a download returning HTML or an error page.
    if len(results) != 154 or any(len(re.findall(r"[a-zA-Z]+", sonnet)) < 60 for sonnet in results):
        raise ValueError(f'Could not recognize all 154 sonnets (found {len(results)}). Try a local Gutenberg plain-text file.')
    return results


def load_sonnets():
    if CORPUS_CACHE.exists():
        content = CORPUS_CACHE.read_text(encoding='utf-8-sig')
        try: return parse_sonnets(content)
        except ValueError: pass
    errors=[]
    for url in SONNETS_URLS:
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'MiniGPT-Education/1.0'})
            with urllib.request.urlopen(req,timeout=15) as response:
                content=response.read().decode('utf-8-sig')
            poems=parse_sonnets(content)
            try: CORPUS_CACHE.write_text(content,encoding='utf-8')
            except OSError: pass
            return poems
        except Exception as err:
            errors.append(str(err))
    raise ValueError('Could not download the sonnets. Check your internet connection, '
                     'or save Project Gutenberg eBook #1041 (plain text) as '
                     f'{CORPUS_CACHE.name} next to this script.\n' + errors[-1][:180])


def sonnet_tokens(poem):
    return re.findall(r"[a-z]+(?:['’][a-z]+)*",poem.lower())


def shakespeare_examples(poems, context=16, stride=8):
    """Keep poem boundaries separate. Each window trains multiple next-token targets."""
    output=[]
    for poem in poems:
        ids=[BOS]+sonnet_tokens(poem)+[EOS]
        for begin in range(0, len(ids)-1, stride):
            # A sliding window is NOT a new poem: only the first window has BOS,
            # and only the actual ending of a sonnet contains EOS.
            chunk=ids[begin:begin+context+1]
            if len(chunk)>=2: output.append(chunk)
    return output

# ---- design tokens -----------------------------------------------------------
FONT_UI='Segoe UI'; FONT_MONO=('Cascadia Mono','Consolas')
T_XS,T_S,T_M,T_L,T_XL,T_XXL=9,10,11,13,16,20       # type scale in points (Tk scaling handles DPI)
BG='#f5f6f3'; PANEL='#ffffff'; PANEL2='#e7eae6'; TOUR_BG='#eef2ee'
LINE='#c9d1cd'; LINE2='#8a9aa3'                     # decorative hairlines / outlines that must be seen
FG='#202d37'; MUTED='#4b5c66'                       # 14.1:1 and 7.0:1 on white
BRAND='#173248'; BRAND2='#315e72'; STATUS_FG='#e3eef1'
BLUE='#1f4e85'      # primary: buttons, selection, confidence curve (white text 8.4:1)
GREEN='#167a55'     # go, correct, accuracy
RED='#a23a43'       # loss, errors, hidden / future words
GOLD='#7a4300'      # the target word and the selected row, nothing else
TIME='#5b3fa0'      # the rewound-in-time state
GREEN_TINT='#dcf3e6'; RED_TINT='#f8e0df'; GOLD_TINT='#ffebb9'; TIME_TINT='#ece6f7'
CODE_BG='#1f2d3a'; CODE_FG='#e6efe8'
HEAT_SEQ=('#eef4fb','#8fb6dc','#1f4e85')   # 0 → max, for tensors that cannot be negative
HEAT_DIV=('#b2405a','#f7f7f7','#1f4e85')   # −max → 0 → +max, for signed tensors
HEAT_MASK='#eceff1'                        # masked / −∞ cells
GRID=LINE
FONT=(FONT_UI,T_M)


def _rgb(color):
    return tuple(int(color[i:i+2],16) for i in (1,3,5))

def blend(c1,c2,t):
    return '#%02x%02x%02x'%tuple(int(round(a+(b-a)*t)) for a,b in zip(_rgb(c1),_rgb(c2)))

def ramp(stops,t):
    """Colour at position t in [0,1] along a list of hex stops."""
    t=min(max(t,0.0),1.0); n=len(stops)-1; i=min(int(t*n),n-1)
    return blend(stops[i],stops[i+1],t*n-i)

def luminance(color):
    def channel(v):
        v/=255
        return v/12.92 if v<=0.03928 else ((v+0.055)/1.055)**2.4
    r,g,b=(channel(v) for v in _rgb(color))
    return 0.2126*r+0.7152*g+0.0722*b

class Attention(nn.Module):
    """Multi-head causal self-attention; display per-head and combined tensors."""
    def __init__(self, dim, heads=HEADS):
        super().__init__()
        if dim % heads: raise ValueError('Embedding width must divide evenly by attention heads.')
        self.heads=heads
        self.head_dim=dim//heads
        self.query=nn.Linear(dim,dim,bias=False)
        self.key=nn.Linear(dim,dim,bias=False)
        self.value=nn.Linear(dim,dim,bias=False)
        self.proj=nn.Linear(dim,dim,bias=False)
        self.last={}

    def forward(self,x):
        b,t,d=x.shape
        def split(linear):
            return linear(x).reshape(b,t,self.heads,self.head_dim).transpose(1,2)
        q,k,v=split(self.query),split(self.key),split(self.value)
        scores=q @ k.transpose(-2,-1) / math.sqrt(self.head_dim)
        mask=torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1)
        masked=scores.masked_fill(mask,float('-inf'))
        weights=masked.softmax(-1)
        mixed=weights @ v
        concatenated=mixed.transpose(1,2).contiguous().reshape(b,t,d)
        result=self.proj(concatenated)
        # All tensor snapshots are real values from this forward pass.
        snapshots={}
        for name,tensor in [('Q',q),('K',k),('V',v),('Scores',scores),
                            ('Masked scores',masked),('Attention weights',weights),
                            ('Attention output',mixed)]:
            # Store head-specific matrices under explicit names for the inspector.
            for h in range(self.heads):
                snapshots[f'{name} (head {h+1})']=tensor[:,h].detach().cpu()
            # For attention, an average across heads is useful; for Q/K/V,
            # concatenate the per-head feature vectors in original feature order.
            if name in ('Scores','Masked scores','Attention weights'):
                snapshots[name]=tensor.mean(dim=1).detach().cpu()
            else:
                snapshots[name]=tensor.transpose(1,2).contiguous().reshape(b,t,-1).detach().cpu()
        self.last=snapshots
        return result

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, dim=32, max_len=26):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_len, dim)
        self.attention = Attention(dim,HEADS)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.output = nn.Linear(dim, vocab_size)
        self.last = {}

    def forward(self, tokens):
        b, t = tokens.shape
        x = self.token_embedding(tokens) + self.position_embedding(torch.arange(t, device=tokens.device))
        emb = x
        x = x + self.attention(self.norm1(x))
        after_attn = x
        x = x + self.mlp(self.norm2(x))
        logits = self.output(x)
        self.last = {name: val.detach().cpu() for name, val in
                     [('Token + position embeddings', emb), ('After attention', after_attn),
                      ('After MLP', x), ('Logits', logits),
                      ('Probabilities', logits.softmax(dim=-1))]}
        return logits

class Engine:
    def __init__(self, lines, mode='small'):
        self.mode=mode
        self.source=list(lines)   # untouched input, so Reset can rebuild an identical engine
        raw_lines=[line.lower().split() if isinstance(line,str) else list(line) for line in lines]
        # Small dataset lines are whole sentences; sonnet windows already have
        # poem-boundary markers only where those boundaries truly occur.
        if mode=='small':
            self.lines=[[BOS]+row+[EOS] for row in raw_lines]
        else:
            self.lines=raw_lines
        if mode=='small':
            if not 2 <= len(self.lines) <= 40:
                raise ValueError('Enter between 2 and 40 nonempty training sentences.')
            if any(not 5 <= len(row) <= MAX_CONTENT+2 for row in self.lines):
                raise ValueError('Every sentence must have 3–24 space-separated words.')
        if not self.lines or any(len(row)<2 or len(row)>MAX_CONTENT+2 for row in self.lines):
            raise ValueError('Each training window needs between 2 and 26 tokens.')
        self.words=[BOS,EOS]+sorted({word for row in self.lines for word in row if word not in (BOS,EOS)})
        if mode=='small' and len(self.words)>100:
            raise ValueError('Use at most 100 unique words for a readable exhibit.')
        self.stoi={word:i for i,word in enumerate(self.words)}
        self.itos={i:word for word,i in self.stoi.items()}
        self.max_t=max(len(row)-1 for row in self.lines)
        self.x=torch.zeros((len(self.lines),self.max_t),dtype=torch.long)
        self.y=torch.full((len(self.lines),self.max_t),-100,dtype=torch.long)
        for i,row in enumerate(self.lines):
            ids=[self.stoi[w] for w in row]
            self.x[i,:len(ids)-1]=torch.tensor(ids[:-1])
            self.y[i,:len(ids)-1]=torch.tensor(ids[1:])
        torch.manual_seed(SEED)
        self.dim=64 if mode=='shakespeare' else 32
        self.lr=0.002 if mode=='shakespeare' else 0.003
        self.batch_size=32 if mode=='shakespeare' else len(self.lines)
        self.model=MiniGPT(len(self.words),dim=self.dim,max_len=MAX_CONTENT+2)
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=self.lr)
        # A second model of the same shape receives saved checkpoints so the app
        # can show any earlier moment of training without touching the live model.
        self._scratch=MiniGPT(len(self.words),dim=self.dim,max_len=MAX_CONTENT+2).eval()
        self._scratch_step=None
        self.step=0
        self.history=[]
        self.viewing=None          # None = live model; otherwise a past step being viewed
        self.checkpoints=[]        # [(step, state_dict)] — thinned when over budget
        bytes_per_checkpoint=sum(p.numel()*4 for p in self.model.parameters())
        self.max_checkpoints=max(24,min(400,self.CHECKPOINT_BUDGET//bytes_per_checkpoint))
        self._trace={}             # (sentence, position) -> {step: P(correct word)}
        self.display_indices=list(range(len(self.lines))) if mode=='small' else list(range(0,len(self.lines),max(1,len(self.lines)//60)))[:60]
        self.eval_indices = (torch.randperm(len(self.lines),generator=torch.Generator().manual_seed(204))[:min(96,len(self.lines))]
                             if mode=='shakespeare' else torch.arange(len(self.lines)))
        self._save_checkpoint()
        self.initial_q=self.checkpoints[0][1]['attention.query.weight']
        self.measure()

    CHECKPOINT_BUDGET=48*1024*1024

    def facts(self):
        """Numbers the explanations quote, so the text stays true in both modes."""
        return {'dim':self.dim,'dim4':self.dim*4,'head_dim':self.dim//HEADS,'lr':self.lr,
                'vocab':len(self.words),'sentences':len(self.lines),
                'batch':('all training sentences at once (full batch)' if self.mode=='small'
                         else f'a random batch of {self.batch_size} windows')}

    # ----- training history -------------------------------------------------
    def _save_checkpoint(self):
        state={k:v.detach().clone() for k,v in self.model.state_dict().items()}
        self.checkpoints.append((self.step,state))
        if len(self.checkpoints)>self.max_checkpoints:
            # Keep step 0 and the latest; drop every other one in between.
            self.checkpoints=self.checkpoints[:-1:2]+[self.checkpoints[-1]]

    @property
    def checkpoint_steps(self):
        return [s for s,_ in self.checkpoints]

    def nearest_checkpoint(self,step):
        return min(self.checkpoint_steps+[self.step],key=lambda s:abs(s-step))

    def model_at(self,step):
        """The model as it was at `step` (a saved checkpoint), or the live model."""
        if step is None or step>=self.step:
            return self.model
        if self._scratch_step!=step:
            state=next(st for s,st in reversed(self.checkpoints) if s<=step)
            self._scratch.load_state_dict(state)
            self._scratch_step=step
        return self._scratch

    @property
    def view_step(self):
        return self.step if self.viewing is None else self.viewing

    def metric_at(self,step):
        return self.history[min(step,len(self.history)-1)]

    def trace(self,si,pos):
        """P(correct next word) for one training example at every saved step."""
        cache=self._trace.setdefault((si,pos),{})
        for s in self.checkpoint_steps+[self.step]:
            if s not in cache:
                cache[s]=self._target_prob(si,pos,s)
        return sorted(cache.items())

    def _target_prob(self,si,pos,step):
        row=self.lines[si]
        ids=torch.tensor([[self.stoi[w] for w in row[:pos]]],dtype=torch.long)
        model=self.model_at(step)
        with torch.no_grad(): probs=model(ids)[0,-1].softmax(-1)
        return probs[self.stoi[row[pos]]].item()

    def predict_prompt(self,prompt,step=None):
        words=sonnet_tokens(prompt) if self.mode=='shakespeare' else prompt.lower().strip().split()
        if not words: raise ValueError('Type at least one word.')
        if BOS in words or EOS in words:
            raise ValueError('Do not enter special markers; BOS is inserted automatically.')
        unknown=sorted(set(words)-set(self.stoi))
        if unknown:
            preview=', '.join(unknown[:10])
            raise ValueError(f'Unknown word(s): {preview}. The model only knows its training vocabulary.')
        context=self.model.position_embedding.num_embeddings
        # Autoregressive decoding keeps the most recent tokens, as in a context window.
        # BOS is supplied once at the beginning if it fits within the context.
        window=([BOS]+words) if len(words)<context else words[-context:]
        ids=torch.tensor([[self.stoi[w] for w in window[-context:]]],dtype=torch.long)
        model=self.model_at(self.viewing if step is None else step)
        model.eval()
        with torch.no_grad():
            logits=model(ids)[0,-1]
            probs=logits.softmax(-1)
        probs=probs.detach().cpu()
        # BOS cannot be generated mid-text. EOS remains an eligible stopping token.
        probs[self.stoi[BOS]]=0
        probs/=probs.sum()
        return words,probs

    def measure(self):
        self.model.eval()
        with torch.no_grad():
            logits=self.model(self.x[self.eval_indices])
            probs=logits.softmax(-1)
            targets=self.y[self.eval_indices]
            valid=targets!=-100
            target_p=probs.gather(-1,targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            acc=((logits.argmax(-1)==targets)&valid).sum().item()/valid.sum().item()
            conf=target_p[valid].mean().item()
            loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=-100).item()
        self.history.append({'step':self.step,'acc':acc,'conf':conf,'loss':loss})
        return self.history[-1]

    def train_one(self):
        self.model.train()
        if self.mode=='shakespeare':
            indices=torch.randint(len(self.lines),(32,))
            xb,yb=self.x[indices],self.y[indices]
        else: xb,yb=self.x,self.y
        logits=self.model(xb)
        loss=F.cross_entropy(logits.reshape(-1,len(self.words)),yb.reshape(-1),ignore_index=-100)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.step+=1
        self._save_checkpoint()
        return self.measure()

    def inspect(self,sentence_index,position,step=None):
        """Full forward pass for one training example, at the viewed step unless given."""
        row=self.lines[sentence_index]
        tokens=torch.tensor([[self.stoi[w] for w in row[:position]]],dtype=torch.long)
        step=self.viewing if step is None else step
        model=self.model_at(step)
        model.eval()
        with torch.no_grad(): logits=model(tokens)
        probs=logits[0,-1].softmax(-1)
        target=self.stoi[row[position]]
        self._trace.setdefault((sentence_index,position),{})[self.step if step is None else step]=probs[target].item()
        return {'tokens':row,'position':position,'target':target,'prob':probs[target].item(),
                'prediction':probs.argmax().item(),'top':torch.topk(probs,k=min(8,len(self.words))),
                'arrays':{**model.attention.last,**model.last},
                'wq_change':model.attention.query.weight.detach().cpu()-self.initial_q}

STAGES = [
 ('The question', 'A learner sees some words and tries to predict the next. Every output word has a probability.',
  'logits = model(tokens)\nprobs = logits[0, -1].softmax(-1)'),
 ('Token IDs', 'Words are mapped to integer IDs. The numbers identify tokens; they are not the learned meaning.',
  'ids = [stoi[word] for word in sentence]'),
 ('Embeddings', 'Each token ID looks up a learned {dim}-number vector. The model adds a vector for its position.',
  'x = token_embedding(tokens) + position_embedding(positions)'),
 ('Q, K and V', 'Three learned weight matrices turn the token vectors into queries, keys, and values.',
  'Q = query(x)\nK = key(x)\nV = value(x)'),
 ('Attention scores', 'Query–key dot products compare how relevant tokens are to each other.',
  'scores = (Q @ K.transpose(-2, -1)) / sqrt({head_dim})'),
 ('Causal mask', 'A token cannot read future input positions. The target next token is not supplied as an input.',
  'scores = scores.masked_fill(future_mask, -inf)'),
 ('Attention weights', 'Softmax makes the allowed scores into attention weights; every row sums to 100%.',
  'weights = softmax(scores, dim=-1)'),
 ('Context mixture', 'Attention forms new vectors as weighted sums of the value vectors.',
  'attention_output = weights @ V'),
 ('MLP and output', 'Residual connections and the MLP transform vectors. A linear layer produces a logit for each vocabulary word.',
  'x = x + attention(norm1(x))\nx = x + mlp(norm2(x))\nlogits = output(x)'),
 ('Prediction and loss', 'Softmax makes vocabulary probabilities. Training rewards probability assigned to the correct target.',
  'loss = cross_entropy(logits, targets)'),
 ('Update the weights', 'Backpropagation finds gradients. AdamW adjusts many numerical weights to reduce prediction loss.',
  'optimizer.zero_grad()\nloss.backward()\noptimizer.step()'),
]


# Optional beginner-friendly narration, indexed to STAGES.
PLAIN_STAGES = [
    "The machine sees a few words and takes a guess about the word that comes next. At first, it mostly guesses badly. Training helps it make better guesses.",
    "Computers work with numbers. We give every unique word a number, like a numbered card in a box. The number is just an address, not the word's meaning.",
    "Each numbered word has its own little list of {dim} numbers. These lists can change as the machine learns. We also add numbers that tell it where the word appears.",
    "The model makes three different versions of every word's number-list. One asks what to look for, one describes what's available, and one carries the information to share.",
    "For every pair of words, the model calculates a matching score. A bigger score means it may pay more attention to that word.",
    "No peeking! To guess the next word, the model may use only the words already shown. Later words are covered up, even though we know the answer during training.",
    "The model converts matching scores into shares of attention. For each word, those shares add up to 100% across the words it is allowed to see.",
    "The model blends information from earlier words, giving more weight to words it paid more attention to. The result is another set of numbers for each word.",
    "The blended information passes through more calculations. At the end, the model gives every word in its vocabulary a score.",
    "The model turns those scores into percentages. It checks how much probability it gave the actual next word. Low probability for the right answer means a bigger penalty.",
    "The model figures out which settings contributed to the error and adjusts lots of them by tiny amounts. Do this repeatedly and its guesses can improve.",
]

# Each glossary entry: plain-language explanation, technical detail, and a concrete tie to this program.
GLOSSARY = {
    'token': ('A small piece of text the model processes. Here, a token is one space-separated word.',
              'Real language models commonly use subword tokens; this example uses whole words.',
              'Look at the word cards in “The question”.'),
    'vocabulary': ('The complete collection of words this mini-model knows about.',
                   'Each vocabulary item has an integer ID and an output score.',
                   'Editing training text changes the vocabulary.'),
    'token id': ('A number that identifies a word, like a library call number.',
                 'These IDs index into the embedding table; they do not represent numerical meaning.',
                 'Select “Token IDs” in the Transformer inspector.'),
    'embedding': ('A list of learned numbers representing a word to the model.',
                  'This model has {dim}-dimensional token vectors and learned position vectors.',
                  'Select “Token + position embeddings” in the inspector.'),
    'position embedding': ('Extra numbers that tell the model where each word is in the sentence.',
                            'A learned position vector is added to each token embedding.',
                            'See the Embeddings stage.'),
    'tensor': ('A structured collection of numbers, like a list, table, or stack of tables.',
               'PyTorch tensors hold token IDs, learned vectors, scores, and gradients.',
               'Every matrix in the inspector comes from an actual PyTorch tensor.'),
    'matrix': ('A rectangular table of numbers.',
               'Matrices let the model transform many numbers at once using multiplication.',
               'Inspect Q, K, V, or Attention weights.'),
    'parameter': ('An adjustable number inside the model that training can change.',
                  'Weights and some offsets are learned parameters updated by the optimizer.',
                  'The “Weight changes (Wq)” matrix shows changes since reset.'),
    'weight': ('A number that controls how strongly one signal influences another.',
               'Learned weight matrices are distinct from attention weights, which are calculated for each input.',
               'Inspect “Weight changes (Wq)” and “Attention weights” to compare.'),
    'query': ('The kind of information a word is looking for.',
              'Q is formed with a learned linear projection of the token representations.',
              'Select Q in the tensor inspector.'),
    'key': ('A label describing what information a word might offer.',
            'A query is compared with keys using dot products.',
            'Select K in the tensor inspector.'),
    'value': ('The numerical information a word can contribute.',
              'Attention mixes these V vectors using calculated attention weights.',
              'Select V or Attention output in the inspector.'),
    'attention': ('The process of deciding how much information to gather from each available word.',
                   'Scaled dot-product attention computes softmax(QKᵀ / √d) V with causal masking.',
                   'Explore Q, K, V, Scores, and Attention weights.'),
    'causal mask': ('A rule that covers up words from the future so the model cannot cheat.',
                    'Future-position scores become negative infinity before softmax.',
                    'In “The question”, red cards are future words; the gold card is the target.'),
    'softmax': ('A calculation that turns a group of scores into percentages adding to 100%.',
                'softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ).',
                'It produces attention distributions and next-word probabilities.'),
    'logit': ('A raw score before it is converted into a probability.',
              'The output layer produces one logit per vocabulary item.',
              'Compare “Logits” and “Probabilities” in the inspector.'),
    'probability': ('The model’s estimated chance for one possible next word.',
                    'Softmax produces a distribution whose entries sum to one.',
                    'Watch the next-word bars in “The question”.'),
    'confidence': ('How much probability the model assigned to a particular word.',
                   'Target confidence is the probability given to the correct next token; it is not accuracy.',
                   'See the blue learning curve and the target bar.'),
    'accuracy': ('How often the model’s top guess was actually the correct word.',
                 'Training accuracy is correct argmax predictions divided by all unpadded training targets.',
                 'See the green learning curve.'),
    'loss': ('A single score describing how wrong the model’s guesses are. Lower is better.',
             'Cross-entropy averages −log(probability of the correct token) over valid targets.',
             'Watch the red Loss curve in “Watch the model learn”.'),
    'cross-entropy': ('A way to penalize the model for assigning too little chance to the right answer.',
                      'For one target, cross-entropy is −log p(target).',
                      'See the Prediction and loss stage.'),
    'gradient': ('A hint telling how a small parameter change would affect the error.',
                 'Backpropagation computes partial derivatives of loss with respect to parameters.',
                 'See the Update the weights stage.'),
    'backpropagation': ('The way the model figures out how its adjustable numbers contributed to its mistakes.',
                        'Automatic differentiation applies the chain rule through the whole computation.',
                        'The code uses loss.backward().'),
    'optimizer': ('The rule used to adjust the model’s settings during training.',
                  'This program uses AdamW with learning rate {lr}.',
                  'The code calls optimizer.step() after backward().'),
    'learning rate': ('A setting that controls how large parameter updates tend to be.',
                      'The optimizer here uses lr={lr}; AdamW also adapts updates using gradient history.',
                      'Look at Engine.__init__.'),
    'mlp': ('A small network that further processes each word’s numbers after attention.',
            'This model uses {dim} → {dim4} → {dim} linear layers with GELU in between.',
            'See the MLP and output stage.'),
    'residual connection': ('A shortcut that adds the previous representation back to a new calculation.',
                            'This program computes x = x + attention(norm1(x)), then x = x + mlp(norm2(x)).',
                            'See the MLP and output stage.'),
    'training step': ('One round of guessing, measuring mistakes, and updating the model.',
                      'Each step uses {batch}.',
                      'Press “Train 1” and watch the changes.'),
    'generalization': ('Being able to make useful predictions on word combinations not directly seen in training.',
                       'A tiny model trained on five sentences may memorize them instead.',
                       'Try an unseen prompt after training.'),
}

def normalize_term(value):
    key=value.lower()
    if key not in GLOSSARY and key.endswith('s'):
        key=key[:-1]
    return key

class App:
    # Training speed: (steps per GUI tick, milliseconds between ticks).
    SPEEDS={'Slow':(1,450),'Normal':(1,140),'Fast':(4,60),'Max':(64,30)}
    PER_HEAD={'Q','K','V','Scores','Masked scores','Attention weights','Attention output'}
    DESIGN=(1480,1100)   # window size, in 96-dpi pixels, that the layout was drawn for

    def __init__(self, root, ui_scale=1.0):
        self.root = root
        self.S = ui_scale
        root.title('Inside the Learning Machine | Watch words become predictions')
        sw,sh=root.winfo_screenwidth(),root.winfo_screenheight()
        w=min(int(self.DESIGN[0]*self.S),sw-40); h=min(int(self.DESIGN[1]*self.S),sh-110)
        root.geometry(f'{w}x{h}+{max(0,(sw-w)//2)}+{max(0,(sh-h)//2-30)}')
        root.minsize(min(960,sw-40), min(600,sh-110))
        self.win_h=h
        root.configure(bg=BG)
        self.stage = 0
        self.plain_mode = tk.BooleanVar(value=True)
        self.sent_idx = 0
        self.position = 4
        self.focus_token = None  # None means the final visible token (the prediction position)
        self.running = False
        self.continuous = False
        self.training_left = 0
        self.engine = Engine(DEFAULT_TEXT.splitlines())
        self.shakespeare_poems=None
        self.training_mode='small'
        self._scrubbing=False
        self._token_widths={}
        self._token_keys={}
        self._style()
        self._build()
        self.refresh()

    def px(self,n):
        return int(n*self.S)

    def font(self,size=T_M,bold=False,mono=False):
        return (self.mono_family if mono else self.ui_family,size,'bold' if bold else 'normal')

    def _style(self):
        import tkinter.font as tkfont
        families=set(tkfont.families())
        self.ui_family=FONT_UI if FONT_UI in families else 'Helvetica'
        self.mono_family=next((f for f in FONT_MONO if f in families),'Courier New')
        px=self.px
        style=ttk.Style(); style.theme_use('clam')
        style.configure('TNotebook',background=BG,borderwidth=0)
        style.configure('TNotebook.Tab',font=self.font(T_M,True),background=PANEL2,foreground=FG,padding=(px(18),px(10)))
        style.map('TNotebook.Tab',background=[('selected',PANEL)],foreground=[('selected',BLUE)])
        style.configure('TCombobox',fieldbackground=PANEL,foreground=FG,padding=(px(6),px(5)),arrowsize=px(18))
        self.root.option_add('*TCombobox*Listbox.font',self.font(T_M))
        style.configure('Horizontal.TScale',background=PANEL,troughcolor=LINE,sliderlength=px(28))
        style.configure('Head.TCheckbutton',background=BRAND,foreground='white',font=self.font(T_S),
                        indicatorbackground=PANEL,indicatorforeground=FG,indicatorsize=px(16),padding=(px(4),px(2)))
        style.map('Head.TCheckbutton',background=[('active',BRAND)],foreground=[('active','white')],
                  indicatorbackground=[('pressed',PANEL2),('selected',PANEL)])

    def label(self,parent,text='',size=T_M,fg=FG,bold=False,mono=False,**kw):
        bg=kw.pop('bg',parent.cget('bg'))
        return tk.Label(parent,text=text,font=self.font(size,bold,mono),fg=fg,bg=bg,**kw)

    def button(self,parent,text,fn,color=PANEL2,small=False,outline=False):
        """Flat button. Dark fills get white text; outline=True gives a quiet bordered button."""
        px=self.px
        dark=luminance(color)<0.4 and not outline
        return tk.Button(parent,text=text,command=fn,bg=PANEL if outline else color,fg='#fff' if dark else FG,
                         activebackground='#bdcfe2',activeforeground=FG,relief='flat',font=self.font(T_S,True),
                         padx=px(8 if small else 12),pady=px(4 if small else 8),cursor='hand2',bd=0,
                         highlightthickness=1 if outline else 0,highlightbackground=LINE2)

    def combo(self,parent,**kw):
        return ttk.Combobox(parent,font=self.font(T_M),state='readonly',**kw)

    def _wrap_on_resize(self,frame,*widgets):
        """Keep paragraph labels at a readable measure as their panel resizes."""
        def fit(e):
            wl=max(self.px(300),min(self.px(760),e.width-self.px(36)))
            for wd in widgets: wd.config(wraplength=wl)
        frame.bind('<Configure>',fit,add='+')

    # ----- layout -----------------------------------------------------------
    def _build(self):
        px=self.px
        header=tk.Frame(self.root,bg=BRAND); header.pack(fill='x')
        head=tk.Frame(header,bg=BRAND); head.pack(fill='x',padx=px(20),pady=px(10))
        self.label(head,'HOW AN AI LEARNS TO PREDICT WORDS',T_XXL,bold=True,fg='white').pack(side='left')
        self.status=self.label(head,'',T_S,fg=STATUS_FG)
        self.status.pack(side='right',padx=(px(12),0))
        self.button(head,'Glossary',self.show_glossary,color=BRAND2).pack(side='right')
        ttk.Checkbutton(head,text='Everyday explanations',variable=self.plain_mode,
                        command=self.refresh_explanation,style='Head.TCheckbutton').pack(side='right',padx=px(14))
        self.notebook=ttk.Notebook(self.root);self.notebook.pack(fill='both',expand=True,padx=px(8),pady=(px(4),px(6)))
        dashboard=tk.Frame(self.notebook,bg=BG);self.dataset=tk.Frame(self.notebook,bg=BG)
        self.notebook.add(dashboard,text='  LIVE LEARNING  ')
        self.notebook.add(self.dataset,text='  EDIT TRAINING SENTENCES  ')
        self.try_page=tk.Frame(self.notebook,bg=BG)
        self.notebook.add(self.try_page,text='  TRY IT OUT  ')
        self.notebook.bind('<<NotebookTabChanged>>',self._tab_changed)

        # Guided tour strip -> (question | inside the transformer) -> learning curve.
        # The tour and the learning chart are packed first so they always keep their
        # space; the two upper panels absorb whatever height is left.
        self._build_tour(dashboard)
        self.learn=tk.Frame(dashboard,bg=PANEL,highlightbackground=LINE,highlightthickness=1)
        self.learn.pack(side='bottom',fill='x',padx=2,pady=(1,2))
        self._build_learn()
        upper=tk.PanedWindow(dashboard,orient='horizontal',bg=BG,sashwidth=px(8),relief='flat')
        upper.pack(fill='both',expand=True,padx=2,pady=(px(3),px(5)))
        self.story=tk.Frame(upper,bg=PANEL,highlightbackground=LINE,highlightthickness=1)
        self.inside=tk.Frame(upper,bg=PANEL,highlightbackground=LINE,highlightthickness=1)
        upper.add(self.story,minsize=px(380),stretch='always')
        upper.add(self.inside,minsize=px(420),stretch='always')
        self._build_story();self._build_inside()
        self._build_dataset()
        self._build_try()
        # Place the sash once, when the pane first learns its real width (45/55: the matrix needs room).
        def place_sash(event):
            if event.width>1:
                upper.unbind('<Configure>',sash_bind)
                upper.sash_place(0,max(px(380),int(event.width*0.45)),0)
        sash_bind=upper.bind('<Configure>',place_sash)
        self.root.bind('<Configure>',self._root_resized)
        self._highlight_stage_panel()

    def _root_resized(self,event):
        if event.widget is not self.root or not hasattr(self,'graph'): return
        height=self._graph_height(event.height)
        if height!=int(self.graph.cget('height')): self.graph.config(height=height)

    def _graph_height(self,window_h):
        return min(self.px(215),max(self.px(130),int(0.17*window_h)))

    def _build_tour(self,parent):
        px=self.px
        tour=tk.Frame(parent,bg=TOUR_BG,highlightbackground=LINE,highlightthickness=1)
        tour.pack(fill='x',padx=2,pady=(px(3),1))
        row=tk.Frame(tour,bg=TOUR_BG); row.pack(fill='x',padx=px(8),pady=(px(6),px(5)))
        self.button(row,'‹ Back',lambda:self.change_stage(-1)).pack(side='left')
        self.button(row,'Next ›',lambda:self.change_stage(1),BLUE).pack(side='left',padx=(px(4),0))
        self.tour_title=self.label(row,'',T_S,bold=True,fg=BLUE); self.tour_title.pack(side='left',padx=(px(12),px(8)))
        self.tour_text=self.label(row,'',T_M,fg=FG,justify='left',wraplength=px(720))
        self.tour_text.pack(side='left',fill='x',expand=True)
        # The code line sits under the explanation and is shown in technical mode only.
        self.tour_code=self.label(tour,'',T_S,mono=True,fg=CODE_FG,bg=CODE_BG,justify='left',anchor='w',
                                  padx=px(10),pady=px(4))
        row.bind('<Configure>',lambda e:self.tour_text.config(wraplength=max(px(300),min(px(720),e.width-px(420)))))

    def _build_story(self):
        px=self.px; P=px(16)
        top=tk.Frame(self.story,bg=PANEL);top.pack(fill='x',padx=P,pady=(px(12),px(4)))
        self.story_title=self.label(top,'THE QUESTION',T_L,bold=True); self.story_title.pack(side='left',padx=(0,px(12)))
        self.sentence_var=tk.StringVar()
        self.sentence_box=self.combo(top,textvariable=self.sentence_var,width=36)
        self.sentence_box.pack(side='left',fill='x',expand=True)
        self.sentence_box.bind('<<ComboboxSelected>>',lambda e:self.choose_sentence())
        self.prediction_help=self.label(self.story,'',T_M,fg=MUTED,wraplength=px(600),justify='left')
        self.prediction_help.pack(anchor='w',padx=P,pady=(0,px(4)))
        self.token_frame=tk.Frame(self.story,bg=PANEL);self.token_frame.pack(fill='x',padx=px(12),pady=px(4))
        self.token_frame.bind('<Configure>',lambda e:self._reflow('story',e.width,self._draw_tokens))
        self.label(self.story,'WHAT COMES NEXT?',T_S,bold=True,fg=MUTED).pack(anchor='w',padx=P,pady=(px(4),0))
        self.question=self.label(self.story,'',T_L,bold=True,fg=FG,justify='left',wraplength=px(600))
        self.question.pack(anchor='w',padx=P,pady=(0,px(2)))
        self.info=self.label(self.story,'',T_M,fg=MUTED,justify='left',wraplength=px(600))
        self.info.pack(anchor='w',padx=P,pady=(0,px(8)))
        self.label(self.story,'PROBABILITY OF EACH NEXT WORD   (ticks and ▲▼ show the change since the previous step)',T_S,bold=True,fg=MUTED).pack(anchor='w',padx=P)
        self.bar_canvas=tk.Canvas(self.story,bg=PANEL,highlightthickness=0,height=px(120))
        self.bar_canvas.pack(fill='both',expand=True,padx=px(12),pady=px(4))
        self.bar_canvas.bind('<Configure>',lambda e:self.draw_bars())
        self.delta=self.label(self.story,'',T_L,bold=True,fg=GREEN,wraplength=px(600),justify='left')
        self.delta.pack(anchor='w',padx=P,pady=(0,px(12)))
        self._wrap_on_resize(self.story,self.prediction_help,self.question,self.info,self.delta)

    def _build_inside(self):
        px=self.px; P=px(16)
        head=tk.Frame(self.inside,bg=PANEL);head.pack(fill='x',padx=P,pady=(px(12),px(4)))
        self.inside_title=self.label(head,'INSIDE THE TRANSFORMER',T_L,bold=True); self.inside_title.pack(side='left',padx=(0,px(8)))
        # Tensors in the order data flows through the model.
        self.tensor_var=tk.StringVar(value='Attention weights')
        names=['Token IDs','Token + position embeddings','Q','K','V','Scores','Masked scores',
               'Attention weights','Attention output','After attention','After MLP','Logits',
               'Probabilities','Weight changes (Wq)']
        self.head_var=tk.StringVar(value='All heads')
        head_box=self.combo(head,textvariable=self.head_var,width=9,values=['All heads',*[f'Head {h}' for h in range(1,HEADS+1)]])
        head_box.pack(side='right')
        head_box.bind('<<ComboboxSelected>>',lambda e:(self.refresh_matrix_help(),self.draw_matrix()))
        self.tensor_box=self.combo(head,textvariable=self.tensor_var,values=names,width=24)
        self.tensor_box.pack(side='right',padx=(0,px(8)))
        self.tensor_box.bind('<<ComboboxSelected>>',lambda e:(self.refresh_matrix_help(),self.draw_matrix()))
        self.matrix_help=self.label(self.inside,'',T_M,fg=MUTED,wraplength=px(600),justify='left')
        self.matrix_help.pack(anchor='w',padx=P,pady=(0,px(4)))
        self.inside_tokens=tk.Frame(self.inside,bg=PANEL)
        self.inside_tokens.pack(fill='x',padx=px(12),pady=(0,px(2)))
        self.inside_tokens.bind('<Configure>',lambda e:self._reflow('inside',e.width,self._draw_inside_tokens))
        self.word_trace=self.label(self.inside,'',T_S,fg=FG,justify='left',wraplength=px(600))
        self.word_trace.pack(anchor='w',padx=P,pady=(0,px(4)))
        self.matrix_detail=self.label(self.inside,'',T_S,fg=MUTED,justify='left',wraplength=px(600))
        self.matrix_detail.pack(anchor='w',padx=P,pady=(0,px(2)))
        self.matrix_canvas=tk.Canvas(self.inside,bg=PANEL,highlightthickness=0,height=px(240))
        self.matrix_canvas.pack(fill='both',expand=True,padx=px(12),pady=(0,px(8)))
        self.matrix_canvas.bind('<Configure>',lambda e:self.draw_matrix())
        self._wrap_on_resize(self.inside,self.matrix_help,self.word_trace,self.matrix_detail)

    def _build_learn(self):
        px=self.px
        head=tk.Frame(self.learn,bg=PANEL);head.pack(fill='x',padx=px(16),pady=(px(8),px(4)))
        self.learn_title=self.label(head,'WATCH THE MODEL LEARN',T_L,bold=True); self.learn_title.pack(side='left',padx=(0,px(16)))
        # Segmented Train group (1 px separators), then the single primary action: Run / Stop share one slot.
        group=tk.Frame(head,bg=LINE2); group.pack(side='left')
        for n in (1,10,100):
            self.button(group,f'Train {n}',lambda k=n:self.start_train(k)).pack(side='left',padx=(0,1 if n<100 else 0))
        slot=tk.Frame(head,bg=PANEL); slot.pack(side='left',padx=(px(8),0))
        self.run_button=self.button(slot,'Run ▶',lambda:self.start_train(None),GREEN)
        self.stop_button=self.button(slot,'Stop ■',self.stop_training,RED)
        for b in (self.run_button,self.stop_button): b.grid(row=0,column=0,sticky='nsew')
        self.label(head,'Speed',T_S,fg=MUTED).pack(side='left',padx=(px(12),px(4)))
        self.speed_var=tk.StringVar(value='Normal')
        self.combo(head,textvariable=self.speed_var,values=list(self.SPEEDS),width=7).pack(side='left')
        self.view_pill=self.label(head,'',T_S,bold=True,fg=TIME,bg=TIME_TINT,padx=px(10),pady=px(4))
        self.button(head,'Reset',self.reset_model,outline=True).pack(side='right')
        self.learn_stats=self.label(head,'',T_M,bold=True,fg=FG)
        self.learn_stats.pack(side='right',padx=(0,px(16)))
        self.graph=tk.Canvas(self.learn,bg=PANEL,highlightthickness=0,height=self._graph_height(self.win_h),cursor='sb_h_double_arrow')
        self.graph.pack(fill='x',padx=px(14),pady=(0,2))
        self.graph.bind('<Configure>',lambda e:self.draw_graph())
        self.graph.bind('<Button-1>',self._graph_scrub)
        self.graph.bind('<B1-Motion>',self._graph_scrub)
        # Rewind control: drag back through saved checkpoints without disturbing the live model.
        row=tk.Frame(self.learn,bg=PANEL);row.pack(fill='x',padx=px(16),pady=(0,px(8)))
        self.label(row,'REWIND',T_S,bold=True,fg=MUTED).pack(side='left',padx=(0,px(8)))
        self.time_var=tk.DoubleVar(value=0)
        self.time_scale=ttk.Scale(row,from_=0,to=0,variable=self.time_var,command=self._scrub)
        self.time_scale.pack(side='left',fill='x',expand=True)
        self.button(row,'Latest ⏭',lambda:self._set_view(None),small=True).pack(side='left',padx=(px(8),0))
        self.view_label=self.label(row,'',T_S,fg=MUTED)
        self.view_label.pack(side='left',padx=(px(10),0))

    def _build_dataset(self):
        px=self.px; P=px(18)
        self.label(self.dataset,'CHANGE THE TEXT, CHANGE WHAT THE MODEL LEARNS',T_XL,bold=True).pack(anchor='w',padx=P,pady=(P,px(6)))
        self.label(self.dataset,'Custom mode: one sentence per line; 2–40 sentences; 3–24 words each. '
                   'Different sentence lengths are supported. Space-separated words become tokens.',T_M,fg=MUTED,
                   wraplength=px(640),justify='left').pack(anchor='w',padx=P,pady=px(4))
        self.editor=tk.Text(self.dataset,bg=PANEL,fg=FG,insertbackground=FG,font=self.font(T_L,mono=True),relief='flat',
                            padx=px(16),pady=px(12),undo=True)
        self.editor.pack(fill='both',expand=True,padx=px(16),pady=px(12))
        self.editor.insert('1.0',DEFAULT_TEXT)
        bottom=tk.Frame(self.dataset,bg=BG); bottom.pack(fill='x',padx=px(16),pady=(0,px(12)))
        self.button(bottom,'Rebuild model from this text',self.apply_dataset,BLUE).pack(side='left')
        self.button(bottom,'Restore example sentences',lambda:self.replace_text(DEFAULT_TEXT)).pack(side='left',padx=px(8))
        self.button(bottom,'Load all 154 Shakespeare sonnets',self.load_shakespeare).pack(side='left',padx=px(8))
        self.corpus_status=self.label(self.dataset,'Shakespeare mode downloads Project Gutenberg eBook #1041 once and caches it locally. It trains on short windows from all 154 poems.',T_S,fg=MUTED,wraplength=px(900),justify='left')
        self.corpus_status.pack(anchor='w',padx=P,pady=px(4))
        self.dataset_detail=self.label(self.dataset,'',T_M,fg=MUTED)
        self.dataset_detail.pack(anchor='w',padx=P,pady=(0,px(12)))

    def _build_try(self):
        px=self.px; P=px(22)
        page=self.try_page
        outer=tk.Frame(page,bg=PANEL,highlightbackground=LINE,highlightthickness=1)
        outer.pack(fill='both',expand=True,padx=px(12),pady=px(12))
        self.label(outer,'TRY A SENTENCE OF YOUR OWN',T_XXL,bold=True).pack(anchor='w',padx=P,pady=(px(20),px(6)))
        self.label(outer,'Type a few words and see which word the trained model would predict next. This does NOT train the model.',
                   T_M,fg=MUTED,wraplength=px(900),justify='left').pack(anchor='w',padx=P,pady=(0,px(12)))
        controls=tk.Frame(outer,bg=PANEL);controls.pack(fill='x',padx=P)
        self.try_var=tk.StringVar(value='the cat chased the')
        self.try_input=tk.Entry(controls,textvariable=self.try_var,font=self.font(T_XL),bg='#f3f6f5',fg=FG,
                                insertbackground=FG,relief='solid',bd=1)
        self.try_input.pack(side='left',fill='x',expand=True,ipady=px(8))
        self.try_input.bind('<Return>',lambda ev:self.run_try())
        self.button(controls,'Predict next word →',self.run_try,BLUE).pack(side='left',padx=(px(10),0))
        self.button(controls,'Generate response ▶',self.start_generate,GREEN).pack(side='left',padx=(px(8),0))
        opts=tk.Frame(outer,bg=PANEL);opts.pack(fill='x',padx=P,pady=(px(10),0))
        self.label(opts,'Generate up to',T_M,fg=MUTED).pack(side='left')
        self.generate_count=tk.IntVar(value=8)
        ttk.Spinbox(opts,from_=1,to=20,textvariable=self.generate_count,width=4,font=self.font(T_M)).pack(side='left',padx=(px(6),px(14)))
        self.label(opts,'words   ·   Selection:',T_M,fg=MUTED).pack(side='left')
        self.generate_mode=tk.StringVar(value='Most likely')
        self.combo(opts,textvariable=self.generate_mode,values=['Most likely','Sample (varied)'],width=17).pack(side='left',padx=px(6))
        self.button(opts,'Stop',self.stop_generate,outline=True).pack(side='left',padx=px(8))
        self.label(outer,'Generate response takes each predicted word, adds it to the input, then predicts the following word. '
                   'The model begins each sequence with <BOS> and can end by generating <EOS>. It may still stop at the word limit if it has not learned endings. Long prompts use the most recent context.',
                   T_S,fg=MUTED,wraplength=px(900),justify='left').pack(anchor='w',padx=P,pady=(px(6),0))
        self.generated_title=self.label(outer,'',T_L,bold=True,fg=GREEN,wraplength=px(900),justify='left')
        self.generated_title.pack(anchor='w',padx=P,pady=(px(6),0))
        self.generated_detail=self.label(outer,'',T_S,fg=MUTED,wraplength=px(900),justify='left')
        self.generated_detail.pack(anchor='w',padx=P,pady=(px(2),0))
        self.generating=False
        self.generation_id=0
        examples=tk.Frame(outer,bg=PANEL);examples.pack(fill='x',padx=P,pady=(px(10),px(12)))
        self.label(examples,'Try:',T_S,fg=MUTED).pack(side='left')
        for val in ['the cat chased the','the dog ate the','shall i compare thee to a']:
            self.button(examples,val,lambda v=val:self._example_try(v),small=True).pack(side='left',padx=px(4))
        self.try_error=self.label(outer,'',T_M,fg=RED,wraplength=px(900),justify='left')
        self.try_error.pack(anchor='w',padx=P)
        self.try_result=self.label(outer,'',T_XL,bold=True,fg=FG,wraplength=px(900),justify='left')
        self.try_result.pack(anchor='w',padx=P,pady=(px(12),px(6)))
        self.try_context=self.label(outer,'',T_M,fg=MUTED,wraplength=px(900),justify='left')
        self.try_context.pack(anchor='w',padx=P,pady=(0,px(6)))
        self.try_canvas=tk.Canvas(outer,bg=PANEL,highlightthickness=0,height=px(275))
        self.try_canvas.pack(fill='both',expand=True,padx=px(20),pady=px(8))
        self.try_canvas.bind('<Configure>',lambda ev:self.draw_try())
        self.try_note=self.label(outer,'',T_M,fg=MUTED,wraplength=px(900),justify='left')
        self.try_note.pack(anchor='w',padx=P,pady=(px(2),px(20)))
        self.try_probs=None
        self.try_prompt=None
        self.run_try()

    # ----- try it out -------------------------------------------------------
    def _tab_changed(self,_event=None):
        if self._try_visible(): self.run_try()

    def _try_visible(self):
        return hasattr(self,'try_page') and self.notebook.select()==str(self.try_page)

    def stop_generate(self):
        self.generation_id += 1
        self.generating=False
        if hasattr(self,'generated_detail') and self.generated_title.cget('text'):
            self.generated_detail.config(text=self.generated_detail.cget('text')+'  ·  Stopped.')

    def start_generate(self):
        self.stop_generate()
        try:
            words,probs=self.engine.predict_prompt(self.try_var.get())
            count=int(self.generate_count.get())
            if not 1 <= count <= 20:
                raise ValueError('Choose between 1 and 20 words to generate.')
        except (ValueError,tk.TclError) as exc:
            self.try_error.config(text=str(exc))
            self.generated_title.config(text='')
            self.generated_detail.config(text='')
            return
        self.try_error.config(text='')
        self.generating=True
        self.generation_words=list(words)
        self.generation_scores=[]
        self.generation_remaining=count
        self.generation_id += 1
        current_id=self.generation_id
        self.generated_title.config(text='Generating: '+' '.join(words)+'  ▌')
        self.generated_detail.config(text='Each next word appears here with its probability.')
        self.root.after(100,lambda:self._generate_one(current_id))

    def _generate_one(self,run_id):
        if not self.generating or run_id!=self.generation_id:
            return
        try:
            words,probs=self.engine.predict_prompt(' '.join(self.generation_words))
            if self.generate_mode.get()=='Sample (varied)':
                idx=int(torch.multinomial(probs,1).item())
            else:
                idx=int(probs.argmax().item())
            word=self.engine.itos[idx]
            if word==EOS:
                self.generating=False
                self.generated_title.config(text='Generated: '+' '.join(self.generation_words)+'  ■')
                self.generated_detail.config(text='The model predicted <EOS> ('+f'{probs[idx].item():.1%}'+') and ended the response naturally.')
                self.try_result.config(text='Predicted <EOS>: sequence complete')
                self.try_prompt=' '.join(words)
                self.try_probs=probs
                self.draw_try()
                return
            self.generation_words.append(word)
            self.generation_scores.append((word,probs[idx].item()))
            self.generation_remaining-=1
            self.generated_title.config(text='Generated: '+' '.join(self.generation_words)+
                                        ('  ▌' if self.generation_remaining else ''))
            self.generated_detail.config(text='Next-word choices: '+'  →  '.join(
                f'{w} ({p:.1%})' for w,p in self.generation_scores))
            # Show the probabilities that actually produced each new word.
            self.try_prompt=' '.join(words)
            self.try_probs=probs
            self.try_result.config(text=f'At step {len(self.generation_scores)}: "{self.try_prompt}" → "{word}" ({probs[idx]:.1%})')
            self.try_context.config(text=f'Generated {len(self.generation_scores)} words  ·  model at training step {self.engine.view_step}')
            self.try_note.config(text='The bars show the distribution used to select the latest generated word. '
                                 'Generating does not train the model. Generation ends if the model predicts <EOS>.')
            self.draw_try()
            if self.generation_remaining:
                self.root.after(350,lambda:self._generate_one(run_id))
            else:
                self.generating=False
        except ValueError as exc:
            self.generating=False
            self.generated_detail.config(text=str(exc))

    def _example_try(self,prompt):
        self.try_var.set(prompt)
        self.run_try()

    def run_try(self):
        e=self.engine
        try:
            words,probs=e.predict_prompt(self.try_var.get())
        except ValueError as err:
            self.try_probs=None
            self.try_error.config(text=str(err))
            self.try_result.config(text='')
            self.try_context.config(text='')
            self.try_note.config(text='')
            self.draw_try()
            return
        self.try_error.config(text='')
        self.try_prompt=' '.join(words)
        self.try_probs=probs
        top=probs.argmax().item()
        self.try_result.config(text=f'{self.try_prompt}  →  {e.itos[top]}   ({probs[top]:.1%})')
        self.try_context.config(text=f'Input tokens: '+ '   '.join(f'{w} [{e.stoi[w]}]' for w in words) +
                                     f'     ·     model at training step {e.view_step}')
        _,initial=e.predict_prompt(self.try_var.get(),step=0)
        change=probs[top].item()-initial[top].item()
        self.try_note.config(text=f'For "{e.itos[top]}": {initial[top]:.1%} before any training (step 0) → {probs[top]:.1%} at step {e.view_step} '
                                  f'({change*100:+.1f} percentage points). Train more on LIVE LEARNING, or rewind there, then return here.')
        self.draw_try()

    def draw_try(self):
        if not hasattr(self,'try_canvas'):return
        c=self.try_canvas;c.delete('all')
        if self.try_probs is None:return
        probs=self.try_probs
        n=min(10,len(probs))
        top=torch.topk(probs,n)
        px=self.px
        w=max(480,c.winfo_width());height=max(260,c.winfo_height())
        row_h=min(px(34),max(px(23),(height-px(15))/n)); bar_h=px(19)
        for i,(idx,p) in enumerate(zip(top.indices.tolist(),top.values.tolist())):
            y=px(8)+i*row_h
            c.create_text(px(12),y+bar_h/2,anchor='w',text=self.engine.itos[idx],fill=FG,font=self.font(T_L,i==0))
            left=px(145); bw=max(100,w-px(245))
            c.create_rectangle(left,y,left+bw,y+bar_h,fill=PANEL2,outline='')
            c.create_rectangle(left,y,left+max(1,p*bw),y+bar_h,fill=GREEN if i==0 else BLUE,outline='')
            c.create_text(left+bw+px(10),y+bar_h/2,anchor='w',text=f'{p:.1%}',fill=FG,font=self.font(T_M))

    # ----- dataset ----------------------------------------------------------
    def replace_text(self,text):
        self.editor.delete('1.0','end'); self.editor.insert('1.0',text)

    def _install_engine(self,engine,mode):
        """Swap in a freshly built model and point the dashboard at a sensible first example."""
        self.stop_generate()
        self.engine=engine; self.training_mode=mode
        self.sent_idx=engine.display_indices[0]
        line=engine.lines[self.sent_idx]
        self.position=len(line)-1 if mode=='shakespeare' else min(4,len(line)-1)
        self.focus_token=None
        self.refresh()

    def apply_dataset(self):
        if self.running: return
        lines=[row.strip() for row in self.editor.get('1.0','end').splitlines() if row.strip()]
        try: candidate=Engine(lines)
        except (ValueError,RuntimeError) as err:
            messagebox.showerror('Check training text',str(err)); return
        self.stage=0
        self._install_engine(candidate,'small')
        self.notebook.select(0)

    def load_shakespeare(self,announce=True):
        """Build the sonnets model on a worker thread. announce=False is the silent startup load."""
        if self.running:return
        self.corpus_status.config(text='Loading 154 sonnets from Project Gutenberg… (first time requires internet)',fg=BLUE)
        self.status.config(text='Loading the 154 Shakespeare sonnets…')
        self.root.update_idletasks()
        def worker():
            try:
                poems=load_sonnets()
                examples=shakespeare_examples(poems)
                candidate=Engine(examples,mode='shakespeare')
                self.root.after(0,lambda:self._finish_shakespeare(poems,candidate,announce))
            except Exception as err:
                message=str(err)
                self.root.after(0,lambda:self._shakespeare_failed(message,announce))
        threading.Thread(target=worker,daemon=True).start()

    def _finish_shakespeare(self,poems,candidate,announce=True):
        self.shakespeare_poems=poems
        self.corpus_status.config(text=f'Loaded all {len(poems)} sonnets: {len(candidate.lines):,} short training windows, {len(candidate.words):,} unique words. Training uses 32 windows per step.',fg=GREEN)
        self.try_var.set('shall i compare thee to a')
        self.speed_var.set('Fast')
        self._install_engine(candidate,'shakespeare')
        self.notebook.select(0)
        if announce:
            messagebox.showinfo('Shakespeare dataset ready','All 154 sonnets are loaded. Train a few hundred steps, then try generating text. Results take considerably longer to improve than with five sentences.')

    def _shakespeare_failed(self,message,announce):
        self.corpus_status.config(text=message,fg=RED)
        self.refresh()
        if not announce:
            # Startup load: the user never asked for it, so explain why they are seeing five sentences instead.
            self.notebook.select(1)
            messagebox.showwarning('Sonnets unavailable','Could not load the Shakespeare sonnets, so the five example sentences are loaded instead.\n\n'+message)

    def reset_model(self):
        if self.running:return
        self.stop_generate()
        self.engine=Engine(self.engine.source,mode=self.training_mode)
        self.refresh()

    # ----- selection --------------------------------------------------------
    def choose_sentence(self):
        if self.running:return
        self.sent_idx=self.engine.display_indices[max(0,self.sentence_box.current())]
        self.focus_token=None
        self.position=min(self.position,len(self.engine.lines[self.sent_idx])-1)
        self.refresh()

    def choose_position(self,p):
        self.position=p; self.focus_token=None; self.refresh()

    def choose_focus(self,index):
        self.focus_token=index
        self._draw_inside_tokens()
        self.draw_matrix()

    def _selected_row(self):
        if self.focus_token is None or self.focus_token >= self.position:
            return self.position-1
        return self.focus_token

    def _reflow(self,key,width,redraw):
        # Re-wrap the token cards only when the panel's width actually changed.
        if self._token_widths.get(key)!=width and hasattr(self,'tr'):
            self._token_widths[key]=width
            self.root.after_idle(lambda:redraw(force=True))

    @staticmethod
    def _req_width(widget):
        """Requested width without waiting for geometry propagation (a Frame's own value lags)."""
        children=widget.winfo_children()
        if not children:
            return widget.winfo_reqwidth()
        border=2*int(widget.cget('highlightthickness') or 0)
        return border+max(App._req_width(ch)+2*int(ch.pack_info().get('padx',0) or 0) for ch in children)

    def _flow(self,parent,items,build):
        """Pack widgets left to right, starting a new row whenever the current one is full."""
        for child in parent.winfo_children(): child.destroy()
        avail=max(parent.winfo_width(),300)
        row=tk.Frame(parent,bg=parent.cget('bg')); row.pack(fill='x',anchor='w'); used=0
        gap=self.px(3)
        for item in items:
            widget=build(row,item)
            need=self._req_width(widget)+2*gap+2
            if used and used+need>avail:
                widget.destroy()
                row=tk.Frame(parent,bg=parent.cget('bg')); row.pack(fill='x',anchor='w'); used=0
                widget=build(row,item)
            widget.pack(side='left',padx=gap,pady=self.px(2)); used+=need

    def _draw_tokens(self,force=False):
        words=self.tr['tokens']
        # Rebuilding the cards destroys clickable widgets, so skip it while nothing about them changed
        # (every training tick calls refresh; the cards only depend on the sentence and position).
        key=(tuple(words),self.position)
        if not force and self._token_keys.get('story')==key: return
        self._token_keys['story']=key
        compact=len(words)>10   # long sonnet windows: smaller cards, colour alone says visible/target/hidden
        def build(row,item):
            i,word=item
            status='VISIBLE' if i<self.position else ('TARGET' if i==self.position else 'HIDDEN')
            bg={'VISIBLE':GREEN_TINT,'TARGET':GOLD_TINT,'HIDDEN':RED_TINT}[status]
            px=self.px
            box=tk.Frame(row,bg=bg,highlightthickness=px(2) if i==self.position else 0,highlightbackground=GOLD)
            if compact:
                self.label(box,word,T_M,bold=True,fg=FG).pack(padx=px(7),pady=(px(4),px(1) if status=='TARGET' else px(4)))
                if status=='TARGET': self.label(box,status,T_XS,fg=GOLD).pack(padx=px(6),pady=(0,px(3)))
            else:
                self.label(box,word,T_L,bold=True,fg=FG).pack(padx=px(10),pady=(px(7),px(1)))
                self.label(box,status,T_XS,fg=GOLD if status=='TARGET' else MUTED).pack(padx=px(8),pady=(0,px(7)))
            if i>=1:
                for child in (box,*box.winfo_children()):
                    child.bind('<Button-1>',lambda ev,p=i:self.choose_position(p))
                    child.config(cursor='hand2')
            return box
        self._flow(self.token_frame,list(enumerate(words)),build)

    def _draw_inside_tokens(self,force=False):
        tokens=self.tr['tokens']
        chosen=self._selected_row()
        compact=self.position>8   # long Shakespeare windows: word-only chips, IDs stay in the trace line
        key=(tuple(tokens),self.position,chosen)
        if not force and self._token_keys.get('inside')==key: return
        self._token_keys['inside']=key
        def build(row,item):
            i,word=item
            if i<self.position:
                selected=i==chosen
                b=self.button(row,word if compact else f'{word}  [{self.engine.stoi[word]}]',
                              lambda j=i:self.choose_focus(j),BLUE if selected else GREEN_TINT,small=True)
                b.config(font=self.font(T_XS if compact else T_S,selected))
                return b
            return self.label(row,f'→ {word} (TARGET, NOT INPUT)',T_S,bold=True,fg=GOLD)
        self._flow(self.inside_tokens,list(enumerate(tokens))[:self.position+1],build)
        self.word_trace.config(text=f'Selected input: "{tokens[chosen]}" (ID {self.engine.stoi[tokens[chosen]]}) → highlighted matrix row {chosen+1}. Click another green word to trace it.')

    # ----- guided tour ------------------------------------------------------
    def change_stage(self,delta):
        self.stage=(self.stage+delta)%len(STAGES)
        tensor={1:'Token IDs',2:'Token + position embeddings',3:'Q',4:'Scores',5:'Masked scores',
                6:'Attention weights',7:'Attention output',8:'After MLP',9:'Probabilities',10:'Weight changes (Wq)'}
        if self.stage in tensor:
            self.tensor_var.set(tensor[self.stage]); self.head_var.set('All heads')
        self.notebook.select(0)
        self._highlight_stage_panel()
        self.refresh_explanation(); self.draw_matrix()

    def _highlight_stage_panel(self):
        relevant={0:self.story,1:self.inside,2:self.inside,3:self.inside,4:self.inside,5:self.inside,
                  6:self.inside,7:self.inside,8:self.inside,9:self.story,10:self.learn}[self.stage]
        titles={self.story:self.story_title,self.inside:self.inside_title,self.learn:self.learn_title}
        for frame in (self.story,self.inside,self.learn):
            active=frame is relevant
            frame.config(highlightbackground=BLUE if active else LINE,highlightthickness=self.px(3) if active else 1)
            titles[frame].config(fg=BLUE if active else FG)

    # ----- training & time travel -------------------------------------------
    def stop_training(self):
        self.running=False; self.continuous=False
        self.training_left=0
        self.refresh()

    def start_train(self,count):
        if self.running:return
        self.engine.viewing=None   # training always continues from the latest weights
        self.running=True; self.continuous=count is None; self.training_left=count or 0
        self._train_batch()

    def _train_batch(self):
        if not self.running:return
        per_tick,delay=self.SPEEDS[self.speed_var.get()]
        limit=per_tick if self.continuous else min(per_tick,self.training_left)
        # Time-box each tick (at most ~50 ms of training) so the window keeps repainting and
        # buttons stay clickable even when one step is slow, e.g. Shakespeare mode at Max.
        deadline=time.perf_counter()+0.05; done=0
        while done<limit:
            self.engine.train_one(); done+=1
            if not self.continuous: self.training_left-=1
            if time.perf_counter()>deadline: break
        self.refresh()
        if self.continuous or self.training_left:
            self.root.after(delay,self._train_batch)
        else:
            self.running=False; self.refresh()

    def _set_view(self,step):
        e=self.engine
        if self.running: return
        if step is not None:
            step=e.nearest_checkpoint(int(round(step)))
            if step>=e.step: step=None
        if step!=e.viewing:
            e.viewing=step
            self.refresh()

    def _scrub(self,value):
        if not self._scrubbing: self._set_view(float(value))

    def _graph_scrub(self,event):
        c=self.graph; w=max(c.winfo_width(),350)
        pad_l,pad_r=self._graph_pads()[:2]
        frac=(event.x-pad_l)/max(1,w-pad_l-pad_r)
        self._set_view(min(1,max(0,frac))*self.engine.step)

    def _sync_time_controls(self):
        e=self.engine
        self._scrubbing=True
        try:
            self.time_scale.config(to=max(e.step,1),state='disabled' if self.running else 'normal')
            self.time_var.set(e.view_step)
        finally:
            self._scrubbing=False
        if e.step==0:
            text='Train a few steps, then drag here (or click the chart) to rewind.'
        elif e.viewing is None:
            text=f'Showing the latest model (step {e.step}).'
        else:
            text=f'Rewound to step {e.viewing} of {e.step}: every panel shows the model as it was then.'
        self.view_label.config(text=text,fg=TIME if e.viewing is not None else MUTED)
        if e.viewing is not None:
            self.view_pill.config(text=f'⏪ Viewing step {e.viewing} of {e.step}')
            if not self.view_pill.winfo_ismapped(): self.view_pill.pack(side='left',padx=(self.px(12),0))
        else:
            self.view_pill.pack_forget()
        (self.stop_button if self.running else self.run_button).lift()

    # ----- repaint everything -----------------------------------------------
    def refresh(self):
        e=self.engine
        choices=[self._sentence_label(e.lines[i]) for i in e.display_indices]
        self.sentence_box['values']=choices
        self.sent_idx=min(self.sent_idx,len(e.lines)-1)
        self.sentence_box.current(e.display_indices.index(self.sent_idx) if self.sent_idx in e.display_indices else 0)
        self.position=max(1,min(self.position,len(e.lines[self.sent_idx])-1))
        self.tr=e.inspect(self.sent_idx,self.position)
        view=e.view_step
        metric=e.metric_at(view)
        self.status.config(text=(('TRAINING…  ' if self.running else '')+
                                 f'Step {e.step}   •   vocabulary {len(e.words)} words   •   '
                                 f'{sum(len(s)-1 for s in e.lines)} training targets' + ('  •  SHAKESPEARE' if e.mode=='shakespeare' else '')))
        self.refresh_explanation()
        self.dataset_detail.config(text=f'{len(e.lines)} sentences  ·  {len(e.words)} vocabulary words  ·  '
                                    f'{sum(len(s)-1 for s in e.lines)} prediction targets')
        self._draw_tokens()
        self._draw_inside_tokens()
        target_word=self.tr['tokens'][self.position]
        shown=self.tr['tokens'][:self.position]
        if len(shown)>9: shown=['…',*shown[-8:]]   # long windows: the cards above already show every word
        self.question.config(text=' '.join(shown)+'  →  ?')
        self.info.config(text=f'Correct word: {target_word}     •     Model guesses: {e.itos[self.tr["prediction"]]}     •     {"✓ Correct" if self.tr["prediction"]==self.tr["target"] else "✗ Not yet correct"}')
        # How this one example's correct-word probability has moved over training.
        self.trace_points=e.trace(self.sent_idx,self.position)
        before_p=self.trace_points[0][1]
        now_p=self.tr['prob']
        earlier=[(s,p) for s,p in self.trace_points if s<view]
        text=f'P("{target_word}"):  {before_p:.1%} at step 0  →  {now_p:.1%} at step {view}   ({(now_p-before_p)*100:+.1f} pts)'
        if earlier:
            ps,pp=earlier[-1]
            text+=f'\nsince step {ps}: {(now_p-pp)*100:+.2f} pts'
        self.delta.config(text=text)
        progress=('  ·  running' if self.continuous else f'  ·  {self.training_left} remaining') if self.running else ''
        # Loss / accuracy / confidence values live in the chart legend; the head row only says where we are.
        self.learn_stats.config(text=(f'Step {view} of {e.step}' if e.viewing is not None else f'Step {e.step}')+progress)
        self._sync_time_controls()
        self.draw_bars(); self.draw_matrix(); self.draw_graph()
        if self._try_visible() and self.try_var.get().strip(): self.run_try()

    @staticmethod
    def _sentence_label(line):
        """Combobox entry for a training example; long sonnet windows are abbreviated."""
        text=' '.join(line)
        return text if len(text)<=70 else ' '.join(line[:5])+' … '+' '.join(line[-4:])

    def refresh_explanation(self):
        """Refresh the local explanations when data, stage, or the audience mode changes."""
        if not hasattr(self, 'tr') or not hasattr(self, 'tour_text'):
            return
        simple = self.plain_mode.get()
        if simple: self.tour_code.pack_forget()
        elif not self.tour_code.winfo_ismapped(): self.tour_code.pack(fill='x',padx=self.px(8),pady=(0,self.px(6)))
        facts = self.engine.facts()
        target = self.tr['tokens'][self.position]
        if simple:
            prediction = (f'The green words are all the model can read. It must guess "{target}" '
                          'without seeing the answer. The bars show the chances it gives each word.')
        else:
            prediction = ('Causal attention restricts this next-token prediction to the prefix. '
                          'The bars display softmax probabilities over vocabulary tokens.')
        self.prediction_help.config(text=prediction)
        title, technical, code = STAGES[self.stage]
        self.tour_title.config(text=f'TOUR {self.stage+1}/{len(STAGES)}  ·  {title.upper()}')
        self.tour_text.config(text=(PLAIN_STAGES[self.stage] if simple else technical).format(**facts))
        self.tour_code.config(text=code.format(**facts))
        self.refresh_matrix_help()

    def _tensor_key(self):
        name=self.tensor_var.get(); head=self.head_var.get()
        if name in self.PER_HEAD and head!='All heads':
            return f'{name} (head {head.split()[-1]})'
        return name

    def refresh_matrix_help(self):
        if not hasattr(self, 'matrix_help'):
            return
        name=self.tensor_var.get(); dim=self.engine.dim
        simple={
            'Token IDs':'Each word receives an integer ID. This is an address, not its meaning.',
            'Token + position embeddings':f'Each visible word becomes {dim} numbers. Its position adds another {dim}-number vector.',
            'Q':'For every word: numbers that describe what information it is looking for.',
            'K':'For every word: numbers that help other words decide whether to pay attention to it.',
            'V':'For every word: numbers containing the information it can contribute.',
            'Scores':'Each cell compares two visible word positions. Bigger numbers can mean more attention.',
            'Masked scores':'Future word positions are blocked by negative infinity, so they cannot affect earlier predictions.',
            'Attention weights':'Each row shows where one word looks for information. The row adds to 100%.',
            'Attention output':'Each word now contains a mixture of information taken from allowed word positions.',
            'After attention':'The original word vectors are combined with what attention gathered.',
            'After MLP':'The model further transforms each contextual word vector before guessing.',
            'Logits':'Raw next-word scores before turning them into percentages.',
            'Probabilities':'Each row gives the next-word probabilities after that input position.',
            'Weight changes (Wq)':'These are updates to shared learned settings, not word values. They change as the model trains.',
        }
        technical={
            'Q':'Query projection: Q = XWq. Rows are token positions; columns are latent features.',
            'K':'Key projection: K = XWk. Each row is a position-specific key vector.',
            'V':'Value projection: V = XWv. Attention calculates weighted sums of these rows.',
            'Scores':'Scaled dot-product scores: QKᵀ / sqrt(dk). Rows and columns are token positions.',
            'Masked scores':'An upper-triangular causal mask sets future scores to -∞ before softmax.',
            'Attention weights':'Softmax of masked scores; each row is a distribution over allowed keys.',
            'Weight changes (Wq)':'Current Wq − initial Wq; axes are features, not tokens.',
        }
        desc=(simple.get(name,'') if self.plain_mode.get() else technical.get(name,simple.get(name,'')))
        if name in self.PER_HEAD:
            head=self.head_var.get()
            desc+=(f'  Showing {head.lower()} of {HEADS}.' if head!='All heads' else
                   ('  Attention matrices are averaged over all heads.' if name in ('Scores','Masked scores','Attention weights')
                    else f'  All {HEADS} heads side by side, {self.engine.dim//HEADS} features each.'))
        self.matrix_help.config(text=desc)

    def show_term(self,term):
        term=normalize_term(term)
        if term not in GLOSSARY:return
        facts=self.engine.facts()
        plain,technical,example=(s.format(**facts) for s in GLOSSARY[term])
        pop=tk.Toplevel(self.root)
        pop.title('Explain: '+term.title());pop.configure(bg=PANEL)
        px=self.px
        pop.geometry(f'{px(500)}x{px(360)}');pop.minsize(px(400),px(300))
        pop.transient(self.root)
        self.label(pop,term.upper(),T_XL,bold=True,fg=BLUE).pack(anchor='w',padx=px(22),pady=(px(18),px(12)))
        for header,content in [('IN EVERYDAY WORDS',plain),('A BIT MORE TECHNICAL',technical),('WHERE TO SEE IT',example)]:
            self.label(pop,header,T_S,bold=True,fg=MUTED).pack(anchor='w',padx=px(22),pady=(px(5),px(2)))
            self.label(pop,content,T_M,fg=FG,justify='left',wraplength=px(440)).pack(anchor='w',padx=px(22),pady=(0,px(10)))
        self.button(pop,'Close',pop.destroy,outline=True).pack(side='bottom',pady=px(12))
        pop.bind('<Escape>',lambda e:pop.destroy())

    def show_glossary(self):
        pop=tk.Toplevel(self.root);pop.title('MiniGPT — technical term glossary')
        px=self.px
        pop.geometry(f'{px(770)}x{px(570)}');pop.configure(bg=PANEL);pop.transient(self.root)
        self.label(pop,'TECHNICAL TERMS — CLICK TO LEARN',T_XL,bold=True).pack(anchor='w',padx=px(18),pady=(px(14),px(8)))
        search=tk.StringVar()
        entry=tk.Entry(pop,textvariable=search,bg=PANEL2,fg=FG,insertbackground=FG,font=self.font(T_M),relief='flat')
        entry.pack(fill='x',padx=px(18),pady=(0,px(10)),ipady=px(6))
        outer=tk.Frame(pop,bg=PANEL);outer.pack(fill='both',expand=True,padx=px(18),pady=(0,px(14)))
        lb=tk.Listbox(outer,bg=PANEL2,fg=FG,selectbackground=BLUE,selectforeground='white',font=self.font(T_M),relief='flat',activestyle='none')
        scroll=tk.Scrollbar(outer,command=lb.yview);scroll.pack(side='right',fill='y')
        lb.config(yscrollcommand=scroll.set);lb.pack(side='left',fill='both',expand=True)
        def fill(*_):
            lb.delete(0,'end')
            for term in sorted(GLOSSARY):
                if search.get().lower() in term or search.get().lower() in GLOSSARY[term][0].lower(): lb.insert('end',term)
        def open_selected(_=None):
            if lb.curselection():self.show_term(lb.get(lb.curselection()[0]))
        search.trace_add('write',fill);fill()
        lb.bind('<Double-Button-1>',open_selected)
        lb.bind('<Return>',open_selected)
        self.button(pop,'Explain selected term',open_selected,BLUE).pack(pady=(0,px(12)))
        entry.focus_set()

    # ----- drawing ----------------------------------------------------------
    def draw_bars(self):
        c=self.bar_canvas;c.delete('all'); w=max(c.winfo_width(),360);h=max(c.winfo_height(),175)
        top_ids=self.tr['top'].indices.tolist(); top_ps=self.tr['top'].values.tolist()
        target=self.tr['target']
        if target not in top_ids:
            top_ids=top_ids[:-1]+[target]
            top_ps=top_ps[:-1]+[self.tr['prob']]
        # The same example one saved step earlier gives each bar a tick showing where it just was.
        e=self.engine; view=e.view_step
        earlier=[s for s,_ in getattr(self,'trace_points',[]) if s<view]
        prev=None
        if earlier:
            prev=e.inspect(self.sent_idx,self.position,step=earlier[-1])['arrays']['Probabilities'][0,self.position-1]
        px=self.px
        n=len(top_ids); row_h=max(px(24),min(px(34),(h-px(25))/max(n,1))); bar_h=px(18)
        x=px(110); max_bar=max(w-px(195),80)
        for i,(idx,p) in enumerate(zip(top_ids,top_ps)):
            y=px(12)+i*row_h
            is_target=idx==target
            c.create_text(px(13),y+bar_h/2,anchor='w',text=self.engine.itos[idx],fill=GOLD if is_target else FG,font=self.font(T_M,is_target))
            c.create_rectangle(x,y,x+max_bar,y+bar_h,fill=PANEL2,outline='')   # light track: tiny bars stay visible
            c.create_rectangle(x,y,x+max(1,max_bar*p),y+bar_h,fill=GOLD if is_target else BLUE,outline='')
            label=f'{p:.1%}'
            if prev is not None:
                q=prev[idx].item(); px_=x+max_bar*q
                c.create_line(px_,y-2,px_,y+bar_h+2,fill='#fff',width=px(3))
                c.create_line(px_,y-2,px_,y+bar_h+2,fill=FG,width=max(1,px(1)))
                if abs(p-q)>=0.0005: label+=f'  {"▲" if p>q else "▼"}{abs(p-q)*100:.1f}'
            c.create_text(x+max_bar+px(8),y+bar_h/2,anchor='w',text=label,fill=FG,font=self.font(T_S))
            if is_target:c.create_rectangle(3,y-1,px(7),y+bar_h+1,fill=GOLD,outline='')

    def draw_matrix(self):
        c=self.matrix_canvas;c.delete('all')
        name=self._tensor_key();tr=self.tr
        base=name.split(' (head ')[0]
        if name=='Token IDs':
            arr=torch.tensor([[self.engine.stoi[w] for w in tr['tokens'][:self.position]]],dtype=torch.float)
        elif name=='Weight changes (Wq)':arr=tr['wq_change']
        else:arr=tr['arrays'].get(name)
        if arr is None:return
        if arr.dim()==3:arr=arr[0]
        if arr.dim()==1:arr=arr.unsqueeze(0)
        rows,cols=arr.shape
        tokens=tr['tokens'][:self.position]
        vocab=self.engine.words
        selected=self._selected_row()
        pair=base in ('Scores','Masked scores','Attention weights')
        token_rows=base in ('Token + position embeddings','Q','K','V','Attention output',
                            'After attention','After MLP','Logits','Probabilities') or pair
        output_vocab=name in ('Logits','Probabilities')
        weight_matrix=name=='Weight changes (Wq)'
        if name=='Token IDs':
            description='One row of integer LOOKUP IDs. Each column is a visible input word. IDs are not learned vectors.'
        elif pair:
            description=(f'Each ROW is the word asking (query). Each COLUMN is the word it examines (key). '
                         f'Highlighted row = "{tokens[selected]}". The target word is not in this matrix. '
                         'Future positions are blocked in Masked scores / Attention weights.')
        elif output_vocab:
            description=(f'Each ROW uses the words up through that input position to predict the NEXT word. '
                         f'Each COLUMN is a vocabulary candidate. Highlighted row = "{tokens[selected]}"; '
                         f'to predict "{tr["tokens"][self.position]}", read the FINAL input row.')
        elif weight_matrix:
            description='These are changes in a learned parameter matrix. BOTH axes are features, NOT words. Every token shares this same matrix.'
        else:
            description=(f'Each ROW belongs to exactly one visible input word. '
                         f'Highlighted row = "{tokens[selected]}" at position {selected+1}. '
                         f'Columns 0–{cols-1} are numerical FEATURES, not other words.')
        if name=='Token IDs':
            row_detail='Word ↔ ID: '+', '.join(f'{word}={self.engine.stoi[word]}' for word in tokens)
        elif output_vocab:
            row_detail=f'Output candidate words: {", ".join(vocab[:40])}{"…" if len(vocab)>40 else ""}. Select the last visible word to inspect the next-word prediction.'
        elif pair:
            row_detail=f'Highlighted attention question: what does "{tokens[selected]}" read from the visible words?'
        elif weight_matrix:
            row_detail='Weight changes reveal learning, but individual weight cells do not correspond to individual words.'
        else:
            row_detail=f'Rows: '+', '.join(f'{i}={word}' for i,word in enumerate(tokens))+'.'
        friendly={'Token IDs':'Word-to-number lookup', 'Token + position embeddings':'The starting numbers for each word', 'Q':'What each word seeks (query)', 'K':'What each word offers (key)', 'V':'Numbers each word contributes (value)', 'Scores':'How strongly words match', 'Masked scores':'Future words are blocked here', 'Attention weights':'How words share attention (%)', 'Attention output':'Combined information for each word', 'After attention':'Updated word representations', 'After MLP':'Processed word representations', 'Logits':'Raw next-word scores', 'Probabilities':'Chance of each possible next word', 'Weight changes (Wq)':'How one shared weight matrix has changed'}
        self.matrix_detail.config(text=f'{friendly.get(base,base)} · {tuple(arr.shape)} — '+(row_detail if base=='Attention weights' or name in ('Probabilities','Weight changes (Wq)') else description))
        px=self.px
        w=max(300,c.winfo_width());h=max(195,c.winfo_height())
        nr=min(rows,16);nc=min(cols,16)
        left=px(130) if token_rows else px(88)
        top=px(62) if (pair or output_vocab or name=='Token IDs') else px(46)
        cell=min(px(58),max(px(12),min((w-left-18)/max(nc,1),(h-top-px(50))/max(nr,1))))
        gap=max(1,px(1))
        ox=left;oy=top
        finite=arr[torch.isfinite(arr)]
        scale=max(finite.abs().max().item(),1e-6) if finite.numel() else 1.
        nonneg=base in ('Attention weights','Probabilities','Token IDs')   # sequential ramp; everything else diverges around 0
        masked=base in ('Masked scores','Attention weights')
        for j in range(nc):
            if pair: col_label=tokens[j]
            elif output_vocab:col_label=vocab[j]
            elif name=='Token IDs':col_label=tokens[j]
            else:col_label=f'f{j}'
            # Narrow columns get slanted labels so neighbours do not overprint each other.
            slanted=cell<px(28) or len(col_label)>5
            if slanted and len(col_label)>10:col_label=col_label[:9]+'…'
            c.create_text(ox+j*cell+cell/2,oy-px(6),text=col_label,
                          fill=GOLD if pair and j==selected else MUTED,anchor='w' if slanted else 's',
                          font=self.font(T_XS,pair and j==selected),angle=60 if slanted else 0)
        for i in range(nr):
            if token_rows:
                row_label=f'{i}: {tokens[i]}'
            elif name=='Token IDs':row_label='token IDs'
            else:row_label=f'feature {i}'
            row_active=token_rows and i==selected
            c.create_text(ox-px(10),oy+i*cell+cell/2,text=row_label,anchor='e',
                          fill=GOLD if row_active else MUTED,font=self.font(T_S,row_active))
            for j in range(nc):
                val=arr[i,j].item()
                blocked=(masked and j>i) or not math.isfinite(val)
                if blocked:color=HEAT_MASK
                elif nonneg:color=ramp(HEAT_SEQ,abs(val)/scale)
                else:color=ramp(HEAT_DIV,0.5+0.5*val/scale)
                x=ox+j*cell;y=oy+i*cell
                col_active=pair and j==selected
                c.create_rectangle(x,y,x+cell-gap,y+cell-gap,fill=color,
                                   outline=GOLD if row_active else (BLUE if col_active else LINE),
                                   width=max(2,px(2)) if (row_active or col_active) else 1)
                cx,cy=x+cell/2,y+cell/2
                if blocked:
                    text='−∞' if (base=='Masked scores' and cell>=px(30)) else '×'
                    c.create_text(cx,cy,text=text,fill=RED,font=self.font(T_S,True))
                elif cell>=px(30):
                    s=f'{val*100:.0f}%' if base=='Attention weights' else f'{val:.2f}'
                    c.create_text(cx,cy,text=s,fill=FG if luminance(color)>0.35 else '#fff',font=self.font(T_XS,mono=True))
        note_y=oy+nr*cell+px(16)
        if nr<rows or nc<cols:
            c.create_text(ox,note_y,anchor='w',fill=MUTED,width=w-ox-px(8),
                          text=f'Preview shows {nr}/{rows} rows and {nc}/{cols} columns; all values remain in the model.',font=self.font(T_XS))
            note_y+=px(18)
        if pair and nr>selected and base=='Attention weights':
            vals=arr[selected]
            order=list(range(min(len(tokens),selected+1)))
            if len(order)>8:   # long windows: name only the strongest targets so the line stays on one row
                order=sorted(order,key=lambda j:-vals[j].item())[:6]; suffix='  (strongest 6)'
            else: suffix=''
            label='  '.join(f'{tokens[j]}: {vals[j].item():.0%}' for j in order)
            c.create_text(ox,note_y,anchor='nw',fill=FG,width=w-ox-px(8),
                          text=f'"{tokens[selected]}" attends to → {label}{suffix}',font=self.font(T_S))
        if output_vocab and nr>0:
            idx=self.position-1
            if idx < rows:
                probability=tr['arrays']['Probabilities'][0,idx]
                best=probability.argmax().item()
                c.create_text(ox,note_y,anchor='nw',fill=GREEN,width=w-ox-px(8),
                              text=f'FINAL ROW predicts: {vocab[best]} ({probability[best].item():.1%}); '
                                   f'correct target: {tr["tokens"][self.position]} ({probability[tr["target"]].item():.1%})',
                              font=self.font(T_S,True))

    GRAPH_PAD=(56,70,34,40)   # left, right, top, bottom (96-dpi pixels)

    def _graph_pads(self):
        return tuple(self.px(v) for v in self.GRAPH_PAD)

    def draw_graph(self):
        c=self.graph;c.delete('all');w=max(c.winfo_width(),350);h=max(c.winfo_height(),140)
        px=self.px
        pad_l,pad_r,pad_t,pad_b=self._graph_pads()
        e=self.engine; history=e.history
        plot_w=w-pad_l-pad_r; plot_h=h-pad_t-pad_b
        max_step=max(e.step,1)
        loss_max=max(0.1,max(row['loss'] for row in history))
        for frac in [0,0.25,0.5,0.75,1.0]:
            y=pad_t+(1-frac)*plot_h
            c.create_line(pad_l,y,w-pad_r,y,fill=LINE)
            c.create_text(pad_l-px(8),y,text=f'{frac:.0%}',anchor='e',fill=MUTED,font=self.font(T_S))
            c.create_text(w-pad_r+px(8),y,text=f'{frac*loss_max:.2f}',anchor='w',fill=RED,font=self.font(T_S))
        # X ticks at round step counts.
        tick=max(1,10**max(0,len(str(max_step))-1))
        tick=tick if max_step/tick<=8 else tick*2
        for s in range(0,max_step+1,tick):
            x=pad_l+s/max_step*plot_w
            c.create_line(x,pad_t+plot_h,x,pad_t+plot_h+px(4),fill=MUTED)
            c.create_text(x,h-pad_b+px(14),text=str(s),fill=MUTED,font=self.font(T_S))
        def xy(step,frac):
            return pad_l+step/max_step*plot_w, pad_t+(1-min(max(frac,0),1))*plot_h
        at=e.metric_at(e.view_step)
        # (name, colour, width, dash, points, value) — dashes keep the four lines apart without relying on hue.
        series=[('Accuracy',GREEN,px(2),None,[xy(r['step'],r['acc']) for r in history],f'{at["acc"]:.0%}'),
                ('Confidence',BLUE,px(2),(8,4),[xy(r['step'],r['conf']) for r in history],f'{at["conf"]:.0%}'),
                ('Loss (right axis)',RED,px(2),(2,3),[xy(r['step'],r['loss']/loss_max) for r in history],f'{at["loss"]:.2f}'),
                ('This example: P(correct)',GOLD,px(3),None,[xy(s,p) for s,p in getattr(self,'trace_points',[])],f'{self.tr["prob"]:.0%}')]
        for name,color,width,dash,pts,_ in series:
            style={'dash':dash} if dash else {}
            if len(pts)>1:c.create_line(*[v for pt in pts for v in pt],fill=color,width=width,**style)
            elif pts:c.create_oval(pts[0][0]-3,pts[0][1]-3,pts[0][0]+3,pts[0][1]+3,fill=color,outline='')
        # Legend: a sample of each line plus its value at the viewed step.
        lx=pad_l; ly=pad_t-px(16)
        for name,color,width,dash,_,value in series:
            style={'dash':dash} if dash else {}
            c.create_line(lx,ly,lx+px(22),ly,fill=color,width=width,**style)
            item=c.create_text(lx+px(28),ly,text=f'{name}: {value}',anchor='w',fill=color,font=self.font(T_S,True))
            lx=c.bbox(item)[2]+px(18)
        view=e.view_step
        if e.viewing is not None:
            x,_=xy(view,0)
            c.create_line(x,pad_t-4,x,pad_t+plot_h,fill=TIME,width=px(2),dash=(5,3))
            c.create_text(x+5,pad_t+2,text=f'step {view}',anchor='nw',fill=TIME,font=self.font(T_XS,True))
        if e.step>0:
            c.create_text(w-pad_r,h-px(8),text='click or drag on the chart to rewind',anchor='e',fill=MUTED,font=self.font(T_XS))


def main():
    if sys.platform=='win32':
        # Ask Windows for real pixels instead of a blurry bitmap-scaled window.
        try:
            import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError,OSError): pass
    root=tk.Tk()
    dpi_scale=root.winfo_fpixels('1i')/96
    sw,sh=root.winfo_screenwidth(),root.winfo_screenheight()
    # Shrink fonts when the screen is smaller than the size the layout was drawn for.
    dw,dh=App.DESIGN
    # Never below 0.9: on small screens the canvases give up height instead of the type becoming unreadable.
    fit=max(0.9,min(1,(sw-40)/(dw*dpi_scale),(sh-110)/(dh*dpi_scale)))
    ui=dpi_scale*fit
    root.tk.call('tk','scaling',ui*96/72)
    app=App(root,ui)
    # Default corpus: the 154 sonnets (cached next to the script after the first download).
    root.after(0,lambda:app.load_shakespeare(announce=False))
    root.mainloop()

if __name__=='__main__':main()
