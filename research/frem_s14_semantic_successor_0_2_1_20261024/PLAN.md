# FREM-S14 0.2.1 release-identity correction

Version 0.2.1 is a pre-execution packaging correction to the unrun 0.2.0
semantic-reproducibility successor. It does not change the model, data, cohort,
arms, seeds, folds, endpoints, statistical rules or global notebook ceiling.

It repairs three fail-closed release defects discovered before upload:

1. wheel metadata and `frem_s14.__version__` disagreed;
2. the packaged historical configuration still said 2,800 seconds while the
   frozen successor runtime used 4,000 seconds per fold;
3. the packaged README/specification described the predecessor rather than the
   new successor and the release verifier did not reject those mismatches.

The 0.2.0 wheel, archive, notebook, hashes and receipt remain immutable. Build
0.2.1 to new versioned paths, run dependency-free tests from source and a clean
installed wheel, verify all payload hashes and contracts, and only then consider
the exact private Kaggle target. Numerical/CUDA evidence does not transfer from
any predecessor. Historical metrics remain uninterpretable until target and
independent semantic verification both pass.
