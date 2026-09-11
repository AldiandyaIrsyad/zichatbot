"""NLI infra adapters.

Each adapter is imported lazily (from its submodule, inside the application
selector) so that importing ``nli/`` never pulls in ``httpx``, ``torch``, or
``transformers``. Import concrete classes from their submodules directly:

- ``local_client.LocalNLIClient`` — in-process transformers (benchmark).
- ``sequence_classify_client.SequenceClassifyNLIClient`` — Infinity 3-way.
- ``pair_classify_client.PairClassifyNLIClient`` — dedicated ``services/nli``.
- ``zeroshot_client.ZeroshotNLIClient`` — binary zero-shot baseline.
"""
