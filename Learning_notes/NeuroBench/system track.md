## metrics
**Correctness:** must be measured to verify the validity of the solution. No thresholds are imposed so the benchmark leaderboard must be analysed to evaluate correctness - efficiency trade offs of solutions.
**Timing:** measurements can be either sample throughput or execution time depending on the task.

**Efficiency:** blah blah blah

In timing and efficiency measurements both pre and post-processing must be taken into account.

## Benchmarks
**Acoustic scene classification:** Challenges systems to sort audio into predefined categories based on the envinronmental audio context. This can allow embedded devices to adjust sound equalization profiles, target microphone denoising, and support active noise cancellation. 

Also challenges systems to fulfill technical requirements like always-on and real time operation. timing results report on device average execution time per sample. Power should be reported under idle and active contexts. Idle power measures the system prepared for inference with the model loaded, and active power measures the system running pre-processing or inference.

**QUBO:** Quadratic unconstrained binary optimization (QUBO). basically its a series of yes or no decisions to find the best outcome. Tests the SUT using data that creates graphs based on 3 parameters, size, density, and random seeds. It runs 5 different versions of each to verify the results. After a specific amount of time neurobench cuts off the test and measures energy consumption and compares the solution to the best known solution.

