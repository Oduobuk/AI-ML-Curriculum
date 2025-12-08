# Month 5, Week 3: Explainable & Responsible AI Cheatsheet

## Explainable AI (XAI)

### Key Concepts

| Term | Description |
| :--- | :--- |
| **Interpretability** | The degree to which a human can understand the cause and effect of a model's internal workings. |
| **Explainability** | The degree to which a human can understand the reasoning behind a specific decision made by a model. |
| **Black Box Model** | A complex model (e.g., deep neural network, gradient boosting) whose internal workings are not easily understood. |
| **Local Explanation** | Explaining a single prediction made by the model. |
| **Global Explanation** | Understanding the overall behavior of the model as a whole. |

### XAI Techniques

| Technique | Type | Description |
| :--- | :--- | :--- |
| **LIME (Local Interpretable Model-agnostic Explanations)** | Local, Model-agnostic | Approximates a black-box model locally with a simpler, interpretable model (e.g., linear regression) to explain a single prediction. |
| **SHAP (SHapley Additive exPlanations)** | Local & Global, Model-agnostic | Uses game theory to calculate the contribution of each feature to a prediction. Can be aggregated for global explanations. |
| **Feature Importance** | Global, Model-specific | A score that indicates how important each feature is for the model's predictions. Often available in tree-based models. |
| **Partial Dependence Plots (PDP)** | Global, Model-agnostic | Shows the marginal effect of one or two features on the predicted outcome of a model. |
| **Individual Conditional Expectation (ICE) Plots** | Local, Model-agnostic | A more granular version of PDP that shows the effect of a feature on the prediction for a single instance. |

## Responsible AI

### Core Principles

| Principle | Description | Key Considerations |
| :--- | :--- | :--- |
| **Fairness** | Ensuring that a model's predictions are not biased against certain groups or individuals. | Unbiased data, fairness metrics (e.g., demographic parity, equalized odds), bias mitigation techniques. |
| **Accountability** | Establishing clear lines of responsibility for the outcomes of AI systems. | Audits, impact assessments, clear documentation, human oversight. |
| **Transparency** | Making the workings and decisions of an AI system understandable to its users and stakeholders. | Explainability (XAI), clear communication about model capabilities and limitations. |
| **Privacy** | Protecting the sensitive data of individuals used in AI systems. | Data anonymization, differential privacy, secure data handling. |
| **Safety & Robustness** | Ensuring that an AI system operates reliably and is resistant to adversarial attacks. | Rigorous testing, adversarial training, monitoring for performance degradation. |
| **Human-in-the-loop** | Incorporating human oversight and intervention in AI systems. | Mechanisms for users to appeal or correct model decisions. |
