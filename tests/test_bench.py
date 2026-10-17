"""Benchmark helper correctness (pure functions; no model, no server)."""

from __future__ import annotations

from benchmarks.common import corpus_wer, normalize_text, percentiles


def test_normalize_text():
    assert normalize_text("  Hello, WORLD -- it's   twenty-one! ") == "hello world it's twenty one"


def test_corpus_wer_uses_normalization():
    assert corpus_wer(["HELLO WORLD"], ["Hello, world."]) == 0.0
    # 1 substitution over 4 reference words
    assert corpus_wer(["a b c d", "  "], ["a b x d", "ignored"]) == 0.25
    assert corpus_wer([], []) is None


def test_percentiles():
    p = percentiles([float(i) for i in range(1, 101)])
    assert p["n"] == 100 and p["p50"] == 50.5 and abs(p["p95"] - 95.05) < 1e-9
    assert percentiles([])["p50"] is None


def test_aggregate_rtf_is_total_time_over_total_audio_mean_of_repeats():
    from benchmarks.compare_precision import rtf_stats

    # two files (10 s and 2 s), two repeats
    per_file = {"long": [2.0, 3.0], "short": [1.0, 1.0]}
    stats = rtf_stats(per_file, {"long": 10.0, "short": 2.0})
    # repeat 1: 3.0/12 = 0.25, repeat 2: 4.0/12 = 0.3333
    assert stats["aggregate_rtf_per_repeat"] == [0.25, 4.0 / 12]
    assert abs(stats["aggregate_rtf"] - (0.25 + 4.0 / 12) / 2) < 1e-12
    assert abs(stats["aggregate_rtf_spread"] - (4.0 / 12 - 0.25)) < 1e-12
    # secondary view uses per-file minima: long 0.2, short 0.5
    assert stats["rtf_per_file_best_of_repeats"]["min"] == 0.2
    assert stats["rtf_per_file_best_of_repeats"]["max"] == 0.5


def test_int8_float16_is_labelled_mixed_precision():
    from benchmarks.compare_precision import precision_label

    assert precision_label("float16") == "FP16"
    assert "mixed precision" in precision_label("int8_float16")


def test_normalizer_v2_fixes_quote_marks_and_v1_reproduces_phase1():
    from benchmarks.common import normalize_text

    hyp = "\"'He has gone, and gone for good,' answered Polychrome."
    assert normalize_text(hyp, "v1") == "'he has gone and gone for good' answered polychrome"
    assert normalize_text(hyp, "v2") == "he has gone and gone for good answered polychrome"
    assert normalize_text("It’s o'er, don't", "v2") == "it's o'er don't"  # internal kept
    assert normalize_text("the students' books", "v2") == "the students books"


def test_phase1_wer_gap_disappears_under_v2():
    """The two Phase 1 T4 hypotheses that differed (FP16 vs INT8/FP16), scored
    against the matching LibriSpeech-style reference."""
    from benchmarks.common import corpus_wer

    ref = ("HE HAS GONE AND GONE FOR GOOD ANSWERED POLYCHROME WHO HAD MANAGED TO SQUEEZE INTO "
           "THE ROOM BESIDE THE DRAGON AND HAD WITNESSED THE OCCURRENCES WITH MUCH INTEREST")
    fp16 = ("\"'He has gone, and gone for good,' answered Polychrome, who had managed to squeeze "
            "into the room beside the dragon, and had witnessed the occurrences with much interest.")
    int8 = fp16.replace("good,'", 'good,"')
    assert corpus_wer([ref], [fp16], "v1") > corpus_wer([ref], [int8], "v1")  # Phase 1 artifact
    assert corpus_wer([ref], [fp16], "v2") == corpus_wer([ref], [int8], "v2") == 0.0


def test_rescore_reads_both_result_formats(tmp_path):
    from benchmarks.rescore import load_references, rescore

    (tmp_path / "u1.txt").write_text("HELLO WORLD\n")
    refs = load_references(tmp_path)
    stream_result = {
        "label": "fp16 c=4",
        "streams": [{"utterance": "u1", "ok": True, "final_text": "'Hello world.'"},
                    {"utterance": "u1", "ok": False, "final_text": None}],
    }
    precision_result = {"runs": [{"compute_type": "float16", "hypotheses": {"u1": "Hello, word."}}]}
    s = rescore(stream_result, refs)["fp16 c=4"]
    assert s["num_scored"] == 1 and s["num_reference_words"] == 2
    assert s["wer_v1"] == 1.0 and s["wer_v2"] == 0.0
    assert s["hypotheses_changed_by_v2"][0]["v2"] == "hello world"
    p = rescore(precision_result, refs)["float16"]
    assert p["wer_v2"] == 0.5
