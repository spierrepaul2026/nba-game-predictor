# Results

| split          | model                 |   accuracy |   log_loss |   brier |
|:---------------|:----------------------|-----------:|-----------:|--------:|
| valid (2024)   | Always pick home team |      0.545 |      0.691 |   0.249 |
| valid (2024)   | Elo rating only       |      0.658 |      0.628 |   0.218 |
| valid (2024)   | Logistic regression   |      0.661 |      0.606 |   0.209 |
| valid (2024)   | Gradient boosting     |      0.659 |      0.614 |   0.213 |
| test (2025-26) | Always pick home team |      0.551 |      0.689 |   0.248 |
| test (2025-26) | Elo rating only       |      0.671 |      0.624 |   0.215 |
| test (2025-26) | Logistic regression   |      0.675 |      0.603 |   0.208 |
| test (2025-26) | Gradient boosting     |      0.674 |      0.609 |   0.21  |
