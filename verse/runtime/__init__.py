"""verse.runtime — task execution: container sessions, grading, the executor loop, replay.

Only LLM API calls and container mechanics, with no training dependencies; the
harness-evolution experiments run them on CPU machines.

Modules:
    executor            the frozen executor's episode loop (native tool calling) and the
                        LLM transport
    swe_env             SWE container/chroot sessions and action parsing (the executor's world)
    swe_judge           SWE grading (phi): apply the patch, run the tests, return 0 or 1
    tb_env              Terminal-Bench sessions, graded by each task's own tests
    chroot_exec         dockerless SWE backend (runs instance images without a docker daemon)
    step_certificates   per-step replay and leave-one-out certificates
    b1_evidence         replay-checked evidence layers (replay check, leave-one-out,
                        masked-read probe)
"""
