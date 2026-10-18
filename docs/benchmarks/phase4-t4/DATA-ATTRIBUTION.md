# Data attribution

The reference transcripts in [`raw/eval_audio5/`](raw/eval_audio5/)
(`1272-135031-0000.txt` … `1272-135031-0004.txt`) are from the
`dev-clean-2` set of **Mini LibriSpeech** ([OpenSLR SLR31](https://openslr.org/31/)).
Mini LibriSpeech is a subset of **LibriSpeech** ([OpenSLR SLR12](https://openslr.org/12)).
The audio used in the experiment comes from the same files. It is not
committed.

* **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
  as stated on both OpenSLR pages.
* **Origin:**
  * LibriSpeech audio is derived from read audiobooks of the
    [LibriVox](https://librivox.org) project.
  * The texts come from [Project Gutenberg](https://www.gutenberg.org).
* **Citation:** V. Panayotov, G. Chen, D. Povey and S. Khudanpur,
  "LibriSpeech: an ASR corpus based on public domain audio books,"
  *Proc. IEEE ICASSP*, 2015.
* **Changes:** none. The files are committed exactly as they were used in
  the experiment, and their checksums are in [`raw/SHA256SUMS`](raw/SHA256SUMS).

This file is deliberately outside `raw/`, so the evidence checksums cover
only the experiment's own files.
