# Selene: Multi-Domain Fraud Detection System

A comprehensive machine learning system for detecting fraudulent activities across seven interconnected business domains using cross-domain orchestration and real-time risk signal propagation.

## System Overview

Selene implements an enterprise-grade fraud detection architecture with multi-domain analysis capabilities. The system trains separate specialized models for each domain while maintaining continuous inter-model communication through a RiskBus infrastructure for early fraud detection and risk amplification.

### Supported Domains

1. Access Control - Detection of unauthorized resource access and privilege escalation
2. Finance - Identification of expense anomalies and financial irregularities
3. Payroll - Detection of salary manipulation and compensation fraud
4. Approval - Discovery of circular approvals and workflow manipulation
5. Budget - Identification of budget spending anomalies and variances
6. Contract - Detection of high-risk contract patterns and procurement irregularities
7. Procurement - Discovery of vendor fraud and procurement manipulation

## Architecture Overview

The system operates in two distinct phases:

### Phase 1: Training Pipeline
Initial model development using simulated historical data. The training pipeline:
- Imports seven domain-specific detector classes from individual modules
- Loads historical corpus data spanning 720 simulated days
- Trains all detectors in parallel using domain-specific machine learning models
- Performs cross-validation and metrics evaluation
- Persists trained models to disk with version control

### Phase 2: Live Streaming Pipeline
Continuous operation with real-time fraud detection:
- Extends simulation beyond day 720 with live event generation
- Embeds trained models directly into the simulation loop via ObservableList pattern
- Processes every event through CrossDomainOrchestrator for multi-domain analysis
- Accumulates events and detections for periodic model retraining
- Performs hot-swap model replacement every 360 days without system interruption

## Repository Structure

```
.
├── train_pipeline.py                    # Phase 1: Initial model training
├── run_pipeline.py                      # Phase 2: Live streaming with detections
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── simulated_corpus.parquet             # Historical training data (720 days)
├── dataset/
│   └── dataset-generation-fixed.ipynb   # Dataset generation notebook
├── model_creation/
│   ├── selene-access.py                 # Access detector implementation
│   ├── selene-approval.py               # Approval detector implementation
│   ├── selene-budget.py                 # Budget detector implementation
│   ├── selene-contract.py               # Contract detector implementation
│   ├── selene-finance.py                # Finance detector implementation
│   ├── selene-payroll.py                # Payroll detector implementation
│   ├── selene-procurement.py            # Procurement detector implementation
│   └── model/                           # Trained model storage
│       ├── access_initial.pkl
│       ├── approval_initial.pkl
│       └── [7 domain models per cycle]
├── selene/
│   ├── __init__.py                      # Package initialization
│   ├── orchestrator.py                  # RiskBus and cross-domain orchestration
│   └── (future components)
└── data/
    └── alerts/                          # Alert storage directory
```

## Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies include:
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- scikit-learn: Machine learning algorithms and utilities
- xgboost: Gradient boosting implementation
- joblib: Model persistence

## Usage Guide

### Initial Training

Execute the training pipeline to create initial models from the corpus:

```bash
python train_pipeline.py --suffix initial
```

Command options:
- `--corpus PATH`: Specify corpus file path (default: simulated_corpus.parquet)
- `--suffix NAME`: Model version suffix (default: initial)
- `--no-eval`: Skip evaluation metrics
- `--n-jobs N`: Number of parallel training threads
- `--eval-only`: Evaluate existing models without retraining

### Live Streaming Simulation

Run the simulation pipeline with embedded real-time detection:

```bash
python run_pipeline.py --start-day 720 --run-days 720 --retrain-every 360
```

Command options:
- `--start-day N`: Starting day for simulation (default: 360)
- `--run-days N`: Number of days to simulate (default: 360)
- `--retrain-every N`: Days between retraining cycles (default: 360)
- `--model-suffix`: Specify model version to load

Example: Run 720 days with retrain every 360 days:
```bash
python run_pipeline.py --start-day 720 --run-days 720 --retrain-every 360
```

## Technical Implementation Details

### Training Pipeline (train_pipeline.py)

The training pipeline manages the following operations:

1. **Dynamic Module Loading**: Imports detector classes from hyphenated filenames using dynamic import mechanisms

2. **Domain-Specific Data Preparation**: 
   - Filters corpus data per domain
   - Performs stratified train-test splits (80-20)
   - Normalizes features and handles missing values

3. **Parallel Training**:
   - Trains all seven detectors concurrently
   - Each detector utilizes domain-specific feature engineering
   - Applies class imbalance correction via weighted loss

4. **Metrics Evaluation**:
   - Calculates AUC-ROC scores
   - Computes average precision
   - Generates classification reports

5. **Model Persistence**:
   - Saves fitted models using joblib compression
   - Implements version control through suffix naming
   - Supports model loading by version or latest timestamp

### Streaming Pipeline (run_pipeline.py)

The streaming pipeline implements the following architecture:

1. **ObservableList Pattern**:
   - Replaces standard Python list with callback-enabled variant
   - Triggers model scoring synchronously for each simulated event
   - Enriches events with real-time detection results

2. **CrossDomainOrchestrator**:
   - Coordinates multi-domain scoring
   - Manages RiskBus signal propagation
   - Implements consensus alerting logic

3. **RiskBus Communication**:
   - Thread-safe shared memory for risk signals
   - Exponential decay of historical scores (decay_factor=0.92)
   - Multi-domain aggregation for cross-domain boost calculation

4. **Retraining Strategy**:
   - Accumulates events throughout simulation
   - At checkpoint intervals (e.g., every 360 days), combines initial corpus with new events
   - Retrains all detectors using cumulative data
   - Performs atomic model replacement without simulation interruption

### CrossDomainOrchestrator (selene/orchestrator.py)

The orchestration module provides:

1. **RiskSignal Definition**: Dataclass containing agent_id, domain, score, timestamp, and metadata

2. **RiskBus Implementation**:
   - Memory structure: Dict[agent_id][domain][score_history]
   - Publish method: Thread-safe signal registration
   - Query method: Cross-domain score aggregation with routing filters

3. **Signal Routing Matrix**: Defines detector dependencies between domains based on shared actor patterns

4. **Consensus Engine**: Tracks multi-domain alerts for high-confidence detection

## Signal Routing Architecture

Cross-domain interactions follow defined routing patterns:

- Procurement listens to: Contract, Finance, Approval
- Finance listens to: Payroll, Budget, Approval
- Payroll listens to: Finance, Access
- Access listens to: Approval, Finance
- Approval listens to: Procurement, Contract
- Contract listens to: Procurement, Budget
- Budget listens to: Finance, Approval

This routing enables amplification of fraud signals when the same agent exhibits suspicious behavior across multiple domains.

## Data Flow

The complete data pipeline operates as follows:

```
Historical Corpus (360 days)
    |
    v
Training Phase
    | Train all 7 detectors
    v
Trained Models (version: initial)
    |
    v
Simulation Start (Day 361+)
    | Event generation & scoring loop
    v
RiskBus (inter-model communication)
    | Consensus detection
    v
Alerts & Scored Events
    |
    v (Every 360 days)
Retraining Checkpoint
    | Combine corpus + new events
    v
Retrained Models (version: cycle1, cycle2, ...)
    |
    v (Hot-swap)
Continue Simulation
```

## Model Persistence and Versioning

Models are saved with version suffixes enabling rollback and experimentation:

- `*_initial.pkl`: Models trained on corpus data only
- `*_cycle1.pkl`: Models retrained after first 360-day simulation block
- `*_cycle2.pkl`: Models retrained after second 360-day simulation block
- And so forth...

The `load_latest_detectors()` function automatically selects the most recently modified model for each domain.

## Performance Characteristics

The system is optimized for:

- **Parallelization**: Multi-threaded training across domains
- **Streaming**: Real-time event processing without batching
- **Memory Efficiency**: Exponential decay limiting historical signal storage
- **Extensibility**: Simple addition of new domains through DOMAIN_REGISTRY

## Validation and Testing

Execute model evaluation on existing trained models:

```bash
python train_pipeline.py --eval-only --suffix initial
```

This loads saved models and computes evaluation metrics on the held-out test set without retraining.

## System State and Artifacts

The following artifacts are generated during execution:

1. **Trained Models**: `model_creation/model/{domain}_{suffix}.pkl`
2. **Alert Logs**: `data/alerts/{timestamp}_alerts.csv`
3. **Scored Events**: `data/scored_events_{cycle}.parquet`
4. **Execution Logs**: Console output with timestamps and metrics

## Known Limitations

1. The ObservableList pattern requires synchronous callbacks, potentially affecting simulation speed for high-volume event generation
2. RiskBus memory is bounded (max_history=10) to prevent unbounded memory growth
3. Retraining cycles pause the simulation temporarily
4. Cross-domain signal routing is manually defined rather than dynamically learned

## Future Enhancements

Potential areas for system expansion:

1. Adaptive threshold determination for alert generation
2. Automated signal routing optimization based on empirical correlation
3. Hierarchical clustering of agent behaviors across domains
4. Ensemble methods combining domain-specific models
5. Streaming model updates between major retraining cycles

## Development Status

Core functionality is complete and operational:
- Training and inference pipelines fully implemented
- Cross-domain orchestration architecture functional
- Hot-swap model replacement working without system interruption
- All seven domain detectors implemented and integrated

Remaining tasks:
- Performance optimization for large-scale deployments
- Additional validation and stress testing
- Comprehensive logging and observability enhancements

## License

This project is maintained as part of the fraud detection initiative.

## Contact

For questions or contributions, please reference the project documentation and code comments.
