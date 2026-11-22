# Data dictionary
articles.csv: article_id (unique key), group_id (scenario shared by HTML/Markdown siblings), format, title, content, split (train/validation/test).
labels.csv: article_id, category_ids (pipe-separated taxonomy IDs), tags (pipe-separated canonical tags). These are evaluation targets, not input features.
taxonomy.csv: category_id (key), parent_category, category, canonical_tags. A flat two-level hierarchy is enough for this exercise; each category has one parent.
aliases.csv: alias, canonical_tag. Resolve aliases with context; exact text substitution alone is insufficient for ambiguous Apple mentions.

All text is authored synthetic content. No external articles or pretrained weights are bundled. HTML and Markdown versions have the same labels and remain in the same split. File order and IDs must not be used as predictive signals. Optional libraries/model downloads require internet and retain their own licenses.
