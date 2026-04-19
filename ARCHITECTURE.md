# Deepfake Detection System Architecture

```mermaid
graph TB
    subgraph "Data Input Layer"
        A[Raw Media Files] --> B[File Type Detection]
        B --> C{Media Type?}
        C -->|Image| D[Image Loader]
        C -->|Video| E[Video Frame Extractor]
    end

    subgraph "Preprocessing Pipeline"
        D --> F[Image Preprocessor]
        E --> G[Frame Sampling]
        G --> F
        F --> H[Normalization]
        H --> I[Augmentation Engine]
        I --> J[Tensor Conversion]
    end

    subgraph "Feature Extraction"
        J --> K[EfficientNet-B0 Backbone]
        K --> L[Feature Maps]
        L --> M[Global Average Pooling]
        M --> N[Feature Vector 1280D]
    end

    subgraph "Classification Head"
        N --> O[Dropout Layer 0.4]
        O --> P[Linear Layer 1280->2]
        P --> Q[Softmax Activation]
    end

    subgraph "Post-Processing"
        Q --> R[Probability Conversion]
        R --> S[Confidence Scoring]
        S --> T[Threshold Decision]
        T --> U[Final Prediction]
    end

    subgraph "Output Layer"
        U --> V[Web Interface]
        U --> W[JSON Response]
        V --> X[User Report]
        W --> Y[API Response]
    end

    style K fill:#f9f,stroke:#333,stroke-width:4px
    style T fill:#bbf,stroke:#333,stroke-width:4px
    style V fill:#bfb,stroke:#333,stroke-width:4px
```

## Detailed Component Architecture

```mermaid
graph LR
    subgraph "Training Pipeline"
        A1[Dataset Loader] --> B1[Batch Generator]
        B1 --> C1[PyTorch Lightning]
        C1 --> D1[Model Trainer]
        D1 --> E1[Checkpoint Manager]
        E1 --> F1[Model Export]

        G1[Config YAML] --> C1
        H1[Early Stopping] --> C1
    end

    subgraph "Inference Pipeline"
        A2[Input Handler] --> B2[Preprocessor]
        B2 --> C2[Model Loader]
        C2 --> D2[Inference Engine]
        D2 --> E2[Result Processor]
        E2 --> F2[Output Formatter]
    end

    subgraph "Video Processing"
        A3[Video Input] --> B3[Frame Extractor]
        B3 --> C3[Frame Buffer]
        C3 --> D3[Batch Processor]
        D3 --> E3[Probability Aggregator]
        E3 --> F3[Final Score]
    end

    style C1 fill:#f9f,stroke:#333,stroke-width:4px
    style D2 fill:#bbf,stroke:#333,stroke-width:4px
    style E3 fill:#bfb,stroke:#333,stroke-width:4px
```

## Data Flow Architecture

```mermaid
flowchart TD
    subgraph "Input Processing"
        A[User Upload] --> B{Validation}
        B -->|Valid| C[File Storage]
        B -->|Invalid| D[Error Response]
        C --> E[MIME Detection]
    end

    subgraph "Feature Engineering"
        E --> F[Raw Pixel Data]
        F --> G[Resize 224x224]
        G --> H[Color Normalization]
        H --> I[Data Augmentation]
        I --> J[Feature Extraction]
    end

    subgraph "Model Inference"
        J --> K[Load Pretrained Weights]
        K --> L[Forward Pass]
        L --> M[Logit Scores]
        M --> N[Probability Conversion]
        N --> O[Confidence Calculation]
    end

    subgraph "Decision Engine"
        O --> P{Threshold Check}
        P -->|>0.5| Q[Deepfake Detected]
        P -->|<=0.5| R[Real Media]
        Q --> S[Risk Score]
        R --> S
    end

    subgraph "Output Generation"
        S --> T[Result Formatting]
        T --> U[Visualization]
        U --> V[Report Generation]
        V --> W[Web Interface]
        V --> X[Prediction Report]
    end

    style J fill:#f9f,stroke:#333,stroke-width:4px
    style P fill:#bbf,stroke:#333,stroke-width:4px
    style W fill:#bfb,stroke:#333,stroke-width:4px
```

## System Component Interaction

```mermaid
sequenceDiagram
    participant User
    participant Gradio as Gradio Web App
    participant Preprocessor
    participant Model
    participant Output

    User->>Gradio: Upload Media File
    Gradio->>Gradio: Validate File Type
    Gradio->>Preprocessor: Send Valid File

    Preprocessor->>Preprocessor: Extract Features
    Preprocessor->>Preprocessor: Normalize Data
    Preprocessor->>Model: Send Tensor

    Model->>Model: Load Weights
    Model->>Model: Forward Pass
    Model->>Model: Calculate Probabilities
    Model->>Output: Return Prediction

    Output->>Gradio: Format Results
    Gradio->>User: Display Results & Confidence
```

## Deployment Architecture (Current)

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[Gradio UI] --> B[File Upload Handler]
        B --> C[Preview Generator]
        C --> D[Result Display]
    end

    subgraph "Application Layer"
        E[Python Application] --> F[Model Inference]
        F --> G[Result Processing]
    end

    subgraph "Model Layer"
        H[Pretrained Model] --> I[EfficientNet-B0]
        I --> J[Classification Head]
    end

    subgraph "Storage"
        K[Model Weights] --> H
        L[Input Files] --> E
    end

    A --> E
    E --> I
    I --> G
    G --> D

    style I fill:#f9f,stroke:#333,stroke-width:4px
    style E fill:#bbf,stroke:#333,stroke-width:4px
    style D fill:#bfb,stroke:#333,stroke-width:4px
```

## Model Training Architecture

```mermaid
graph LR
    subgraph "Data Pipeline"
        A[Raw Dataset] --> B[Train/Val Split]
        B --> C[Data Augmentation]
        C --> D[Batch Creation]
    end

    subgraph "Training Loop"
        D --> E[Forward Pass]
        E --> F[Loss Calculation]
        F --> G[Backpropagation]
        G --> H[Weight Update]
        H --> E
    end

    subgraph "Optimization"
        I[Adam Optimizer] --> H
    end

    subgraph "Monitoring"
        F --> L[Loss Tracking]
        E --> M[Accuracy Metrics]
        L --> N[Logging]
        M --> N
    end

    subgraph "Model Management"
        H --> O[Checkpoint Save]
        O --> P[Best Model Selection]
        P --> Q[Model Export]
        Q --> R[Production Deploy]
    end

    style E fill:#f9f,stroke:#333,stroke-width:4px
    style I fill:#bbf,stroke:#333,stroke-width:4px
    style R fill:#bfb,stroke:#333,stroke-width:4px
```

## Architecture Summary

This updated architecture now accurately reflects the current Deepfake Detection System implementation:

**Core Components (Implemented ✓):**
- **Data Input & Preprocessing:** Image/video loading with 224x224 resize and normalization
- **Feature Extraction:** EfficientNet-B0 backbone with ImageNet weights
- **Classification Head:** Dropout (0.4) + Linear layer (1280→2) + Softmax
- **Training Pipeline:** PyTorch Lightning with Adam optimizer, early stopping, checkpointing
- **Inference Pipeline:** Model loading, forward pass, probability conversion
- **Video Processing:** Frame extraction and probability aggregation
- **Web Interface:** Gradio-based UI for file upload and results display
- **Model Management:** Checkpoint saving and best model selection

**Simplified Components:**
- Single model (ensemble system removed)
- Direct probability averaging for videos
- Local storage with file handling
- No external database, task queue, or advanced monitoring infrastructure

