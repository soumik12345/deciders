# Data

Data encoders turn states, questions, and answer options into token sequences
and readout positions. The data collator batches the encoded question rows with
right padding, attention and option masks, labels, and per-record loss weights.
Import the public classes from `deciders.data`:

```python
from deciders.data import DataCollator, DataEncoder, KevDataEncoder
```

Use `KevDataEncoder` for Kev's encoding scheme and pass it to `DataCollator`.
Subclass `DataEncoder` and override `get_token_patterns` to customize the
delimiter tokens and control-token matching pattern.

::: deciders.data.encoder.DataEncoder
    options:
      heading_level: 2
      members:
        - get_token_patterns
        - encode_question
        - encode_sequence

::: deciders.data.encoder.KevDataEncoder
    options:
      heading_level: 2
      inherited_members: true
      members:
        - get_token_patterns
        - encode_question
        - encode_sequence

::: deciders.data.collator.DataCollator
    options:
      heading_level: 2
      members:
        - __call__
