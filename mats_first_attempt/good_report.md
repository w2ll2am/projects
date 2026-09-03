
Success features

* Good execution.
  * Clear hypothesis to proven or rejected.
    * Make an empirical, falsifiable prediction based on a hypothesis and test it.
  * Use techniques that work.
  * Graduate to harder methods only when simpler ones fail.
  * Baseline comparison. 
  * Do a lot of experiments and don’t be a perfectionist.
* Interesting question.
* Skeptical testing of assumptions.
  * Critically interpreate each result.
  * Check if intervention actually worked.
* Interesting results.
* Competence. 
  * Technical depth.
  * Always offer a mechanistic analysis to find a root cause.
  * Conceptual analysis
    * Interrogate the object of study, rather than the measurement.
    * Use the understanding to tease apart the other factors that may impact the result and isolate findings better.
    * 
* Good writing.
* Analysis scope. 
  * Changing:
    * Model confirms applicability.
    * Dataset and score confirms generalisation.
    * Variation confirms robustness.
  * 1 model, 1 dataset, 1 score, 1 variation.

Structure

* Clear hypothesis.
  * Explicitly link to existing work. Identify and fill gaps.
* Produce a result.
* Result stats.
  * Identify interesting correlations or trends on several axes.
  * Investigate and use these to inform later experiments.
* Confirm impact of interventions.
* Confirm positive result against a baseline e.g:
  * No change.
  * Same data, but different algo.
  * Opposite change
  * Hard positive or negative
  * Neutral option.
* Identify result mechanism.
  * Mechanistic analysis: 
    * Neuron activations, vectors, or logits. 
    * Attention.
    * How these change depending on the layer.
  * Causal analysis.
    * Resampling
    * SAE on an important token.
  * Clarify exactly what was happening.
* Conceptual analysis.
  * Interpretation - what does the result mean?
  * Characterisation - What are the properties of this result?
  * Separation - Increase hypothesis conformation signal.
    * Name alternative explanations.
    * Designing a discriminating experiments
    * Rank by cost to test.
    * Inteperate results and decide to what extent each explanation can be ruled out
    * Quantify the remaining signal.
