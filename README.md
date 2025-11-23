# Team_Slay_Point
# AI Transaction Categorisation System
### Overview

This project implements an AI-based Transaction Classification System that automatically categorizes financial transactions into predefined categories. It is designed for both User and Admin roles:
User: Can input raw transactions and instantly get predicted categories along with confidence scores.
Admin: Can manage categories, view analytics, and ensure accurate classification.
The system uses a pre-trained Qwen 2.5B language model with LoRA fine-tuning for efficient and accurate predictions.

### Features
- Dual Roles: Admin and User, each with specific capabilities.
- Real-time Classification: Classifies transactions with confidence scores instantly.
- Custom Categories: Admins can manage categories.
- Offline Capability: The model and adapter can run locally, including on CPU.
- LoRA Adapter: Reduces training time and allows efficient predictions without full model fine-tuning.

### Advantages
- Fast and lightweight inference using LoRA adapter.
- Can run on CPU or GPU.
- Scalable for new categories or datasets.
- Confidence scores provide transparency for predictions.

