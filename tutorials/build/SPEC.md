# Tutorial spec (Deep Learning 89-6877, Fall 2026, TA: Tal Fiskus)

A tutorial = **(a) a short slide recap** (10–15 min, selected pages of that week's lecture PDFs) followed by
**(b) a Jupyter implementation session** (~60 min). `T01_neurons_to_backprop` is the reference: read
`build/make_t01.py` and match its structure, density and voice.

## Files you produce for tutorial NN
- `tutorials/build/make_tNN.py`: the single source. Uses `nbkit.Builder` (one source → two notebooks) and calls
  `recap.make_recap(...)` so the recap PDF is rebuilt with the notebooks.
- Generated: `tutorials/TNN_<name>.ipynb` (student, blanks), `tutorials/solutions/TNN_<name>.ipynb` (solution,
  executed), `tutorials/recap/TNN_<name>_recap.pdf`.
- Do not edit `nbkit.py`, `recap.py`, `README.md`, other tutorials, or anything outside your files. Do not commit.

## Markers (see nbkit.py docstring)
- `#>> hint` … `#<<` around the core implementation lines → student gets `# TODO: hint` + `...`.
- `code  # @student: replacement` → single-line blank (replacement must be a full valid line).
- Markdown `<<STUDENT>>…<</STUDENT>>` / `<<SOLUTION>>…<</SOLUTION>>` for version-specific text.
- 6–12 TODOs per tutorial, each followed by a check cell that compares with the PyTorch/library result
  (print max abs difference, or assert). The student version must parse (nbkit checks).

## Notebook structure
1. Title cell: `# Tutorial N: <title>`, course line, `TA: Tal Fiskus`, a Colab badge for the student notebook
   `https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/TNN_<name>.ipynb`, a link to
   the recap slides `https://github.com/idansc/dl-course-2026/blob/main/tutorials/recap/TNN_<name>_recap.pdf`,
   and a timed plan. In the student version add the "fill in every TODO before the tutorial" note (as in T01).
2. `## 1. Lecture recap`: short, with the key equations (what the recap slides cover, in text).
3. Implementation sections from the week's plan, each: brief math → code → check against the library.
4. **One extra topic** beyond the lecture: a short, concrete experiment that teaches a practical insight
   (T01's "Initial weights" is the model). Title it as a normal numbered section. Never call it "anecdote".
5. ✏️ exercises sprinkled in (2–4), optional "If time" section at the end, `## Summary`, `**Further watching:**`.

## Code rules
- Self-contained: never import from another tutorial. If you reuse an earlier component (mini-GPT, ResNet, ViT),
  re-define a compact version in a cell titled "From earlier tutorials".
- `device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")`.
- Datasets download to `./data` (torchvision MNIST/CIFAR-10/FashionMNIST, Tiny Shakespeare from
  `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`, small HF datasets).
  Pretrained models: small ones only from the HF Hub (e.g. `facebook/dinov2-small`, `gpt2`,
  `HuggingFaceTB/SmolLM2-135M`, `openai/clip-vit-base-patch32`). Seed everything.
- **The whole solution notebook must execute in ≤ 10 minutes on a laptop CPU** with `OMP_NUM_THREADS=2`
  (subsets, few epochs, small widths). Say in markdown how to scale up on a Colab GPU.
- Plots: matplotlib, readable, titled, labeled axes.

## Build and verify (mandatory)
```bash
PY=/private/tmp/claude-501/-Users-idanschwartz-Library-CloudStorage-OneDrive-BarIlanUniversity-playground/bb5fb5c5-72a5-41d3-b3fc-41e1b89dd2e2/scratchpad/tut/bin
cd ~/projects/dl-course-2026/tutorials
OMP_NUM_THREADS=2 $PY/python build/make_tNN.py
OMP_NUM_THREADS=2 $PY/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=tut --ExecutePreprocessor.timeout=1200 solutions/TNN_<name>.ipynb
```
(`pymupdf`, `transformers`, `datasets`, `peft`, `torchvision`, `matplotlib` are installed in that venv; install
anything else with `uv pip install -p $PY/python <pkg>`.) Then:
- Zero errors in the executed solution; every check prints a small difference / passes.
- **Read the outputs and look at every figure** (extract PNGs and view them). Make the markdown match what the
  outputs actually show; if a claim in the text contradicts a number, fix the text (T01 had exactly this bug).
- Open the recap PDF page list and confirm the chosen pages are the right slides (titles via `recap.page_texts`).

## Recap slide selection
Use `recap.page_texts(pdf)` to list page titles; pick 8–15 pages that a TA would use to re-teach the lecture
concisely: the definitions, the key equations/diagrams, one result slide. Skip title, outline, history and
duplicated build-up animation pages (pick the final page of a build-up). Teaching order may differ from the deck.

## Voice
Technical and terse, like T01: setup, rule, number, finding. No rhetorical lines, no hype. English.

## Report back
Sections and TODO count, extra topic, recap pages chosen (pdf:pages), solution runtime, any caveat
(e.g. something that needs a GPU to look good).
