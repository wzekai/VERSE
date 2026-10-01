"""verse — code for VERSE, a verified self-evolving optimizer for agent harnesses.

A frozen LLM executor solves benchmark tasks (SWE-rebench, Terminal-Bench) under an
evolvable executor harness (markdown files plus executable hooks). Each round, an optimizer
LLM inspects the training trajectories and edits the harness, and a val sweep scores the
result. The selected round is then evaluated on the held-out test set.

Methods are selected by config (verse/configs/):

    baselines/     the four baselines (meta_harness, ahe, harnessx, self_harness),
                   reimplemented on one shared engine; they differ in how the optimizer
                   sees the trajectories
    ours/          baseline evidence plus every VERSE component except optimizer
                   self-evolution: the verification tools (fix_probe, replay,
                   ablate/substitute), attribution with trace minimization, the training
                   audit and candidate guidance. verified_ahe_* are ablations.
    self_teacher/  configurations with an optimizer harness (meta_teacher): after each
                   round the optimizer can also edit its own harness (prompt, skills,
                   notes, executable hooks). verse_* is full VERSE; the rest are ablations.

Package layout:
    evolution/   the engine: driver (round loop), intervener (optimizer episode),
                 evidence sources, edit spaces, probes, hooks runtime, gates,
                 meta_teacher (optimizer self-evolution), eval (test-set scoring)
    runtime/     execution: SWE docker sessions and grading, Terminal-Bench containers,
                 the executor loop and LLM transport, step certificates
    configs/     one yaml per method; see configs/README.md
"""
