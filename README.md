# CIM
CIM: Interpretable model with consistency representation and prompt learning

Training a Deep Neural Network (DNN) based solely on class labels is insufficient to provide adequate supervisory information. Furthermore, existing explanation methods often expose the inherent biases within black-box neural networks, severely limiting their practical applicability. This work models the learning process of DNNs using causal analysis, identifying spurious correlations . To tackle these challenge, a language-object consistency model for interpretable image classification, referred to as the Consistent Interpretable Model (CIM), is proposed. This model leverages consistency in representation and the association of language prompts with classification objects. The approach involves building a feature representation alignment by minimizing the distance between the feature distributions of original images and their corresponding objects. Additionally, prior knowledge from CLIP is incorporated to guide the network in assigning class-specific prompts to the classification objects. Extensive experimental evaluations across multiple datasets and classification scenarios demonstrate that CIM significantly enhances both classification performance and model interpretability. Furthermore, CIM exhibits improved robustness in out-of-distribution tasks.

Code will be uploaded when the manuscript is accepted!!!!
