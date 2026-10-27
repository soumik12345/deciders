# Models

Decision models score answer options using a language-model backbone and a
pointer head. Import the Kev implementation from `deciders.models.kev`:

```python
from deciders.models.kev import KevDecisionModel
```

::: deciders.models.kev.KevDecisionModel
    options:
      heading_level: 2
      members:
        - encode
        - forward
        - from_pretrained
