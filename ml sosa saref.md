# ML Gateway Documentation: SOSA-SAREF Payload Matching System

## Overview

The ML Gateway is a comprehensive machine learning system designed to automatically find correspondences between SOSA (Sensor, Observation, Sample, and Actuator) and SAREF (Smart Applications REFerence) ontology payloads in IoT actuator applications. The system uses multiple ML algorithms to identify semantically equivalent RDF blocks across different ontological representations.

## System Architecture

### Core Components

The system consists of four main modules:

#### 1. **PayloadExtractor** (`ml_payload_matcher.py`)
- **Purpose**: Parses log files to extract structured RDF payload blocks
- **Key Methods**:
  - `extract_rdf_blocks()`: Identifies RDF triple blocks in log files
  - `extract_actuator_payloads()`: Filters for actuator-specific payloads
  - `parse_rdf_line()`: Converts log lines to structured RDF triples

**Algorithm**: Uses regex patterns to identify RDF triple boundaries and extracts metadata including:
- Line ranges in source files
- Entity counts and types
- Predicate relationships
- Temporal information

#### 2. **FeatureExtractor** (`ml_payload_matcher.py`)
- **Purpose**: Converts RDF blocks into numerical feature vectors for ML processing
- **Feature Types**:
  - **Textual Features**: TF-IDF vectorization of RDF content
  - **Structural Features**: Entity/predicate counts, graph topology
  - **Semantic Features**: Ontology-specific term analysis
  - **Model Features**: Trained RandomForest predictions (when available)

**Algorithm**: Creates multi-dimensional feature vectors combining:
```
Feature Vector = [TF-IDF(384), Structural(6), Semantic(12)]
Total Dimensions: 402
```

#### 3. **SimilarityMatcher** (`ml_payload_matcher.py`)
- **Purpose**: Computes similarity scores between SOSA and SAREF payloads
- **Scoring Algorithm**:
  ```
  With Model:     Overall Score = 0.30 × Cosine + 0.20 × Structural + 0.20 × TF-IDF + 0.30 × Model
  Without Model:  Overall Score = 0.40 × Cosine + 0.30 × Structural + 0.30 × TF-IDF
  ```
- **Key Methods**:
  - `find_correspondences()`: Main matching algorithm
  - `compute_similarity()`: Multi-metric similarity computation
  - `validate_matches()`: Quality assessment and confidence scoring

**Algorithm Details**:
1. **Cosine Similarity**: Computes angular similarity between feature vectors
2. **Structural Similarity**: Compares graph topology metrics
3. **TF-IDF Similarity**: Semantic text similarity using term frequency
4. **Model Similarity**: RandomForest prediction confidence (when trained model available)

#### 4. **PayloadTrainer** (`ml_payload_trainer.py`)
- **Purpose**: Provides supervised learning capabilities for continuous improvement
- **Features**:
  - Interactive feedback collection
  - Random Forest classifier training
  - Model persistence and versioning
  - Performance analytics

**Training Pipeline**:
1. Collect user feedback on match quality (correct/incorrect)
2. Extract features from feedback data
3. Train Random Forest classifier (100 estimators)
4. Apply trained model to improve future matches
5. Save model for persistence


## File Output System

### Generated Files

1. **Detailed Match Reports**: `ontology_matching_results/detailed_matches_YYYYMMDD_HHMMSS.txt`
   - Side-by-side RDF block comparison
   - Line-by-line correspondence visualization
   - Detailed scoring breakdown

2. **Training Data**: `ontology_matching_results/training_data.pkl`
   - Serialized match results for training
   - Feature vectors and correspondence data
   - Temporal metadata for versioning

3. **Trained Models**: `ontology_matching_results/trained_model.pkl`
   - Persistent Random Forest classifier
   - Model metadata and performance metrics
   - Training history and feedback counts

4. **Feedback History**: `ontology_matching_results/feedback.json`
   - User corrections and validations
   - Training examples for supervised learning
   - Accuracy tracking over time

## Usage Instructions

### Basic Matching
```bash
# Activate ML environment
source .venv-ml/bin/activate

# Run basic matching
python ml_payload_matcher.py
```

**Output**: Creates detailed match report and JSON results files in `ontology_matching_results/`

### Training Mode
```bash
# Run training system
.venv-ml/bin/python ml_payload_trainer.py --auto-train

# Options:
# 1. Collect new feedback
# 2. Train/retrain model  
# 3. Test model on new data
# 4. Exit
```


## Algorithm Details

### Multi-Metric Similarity Computation

The system employs a weighted ensemble of similarity metrics:

#### 1. **TF-IDF Similarity (30% weight)**
```python
# Vectorize RDF content using scikit-learn TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(rdf_texts)
similarity = cosine_similarity(sosa_vector, saref_vector)
```

#### 2. **Cosine Similarity (30% weight)**
```python
# Feature vectors are dictionaries extracted from RDF payloads
vec1 = feature_extractor.extract_features(sosa_payload)  # Dict with 12+ features
vec2 = feature_extractor.extract_features(saref_payload) # Dict with 12+ features

# Convert to aligned numerical vectors and calculate cosine similarity
all_keys = set(vec1.keys()) | set(vec2.keys())
v1 = [vec1.get(key, 0.0) for key in all_keys]
v2 = [vec2.get(key, 0.0) for key in all_keys]

dot_product = sum(a * b for a, b in zip(v1, v2))
norm1 = math.sqrt(sum(a * a for a in v1))
norm2 = math.sqrt(sum(b * b for b in v2))
cosine_sim = dot_product / (norm1 * norm2)
```

**Feature Vector Components (vec1, vec2):**
- `triple_count`: Number of RDF triples
- `entity_count`: Number of unique entities
- `predicate_count`: Number of unique predicates
- `has_actuation`: Boolean for actuation-related terms
- `has_actuator`: Boolean for actuator-related terms
- `sosa_terms`: Count of SOSA ontology terms
- `saref_terms`: Count of SAREF ontology terms
- `content_diversity`: Text diversity metric
- `binary_value_count`: Count of boolean values
- Plus 3+ additional semantic features

#### 3. **Structural Similarity (20% weight with model, 30% without)**
```python
# Graph structure comparison
entity_sim = 1 - abs(len(sosa_entities) - len(saref_entities)) / max(len(sosa_entities), len(saref_entities))
predicate_sim = 1 - abs(len(sosa_predicates) - len(saref_predicates)) / max(len(sosa_predicates), len(saref_predicates))
structural_sim = (entity_sim + predicate_sim) / 2
```

#### 4. **Model-based Similarity (30% weight when available)**
```python
# Check if trained model exists and is loaded
if self.trained_model is not None:
    # Create 4-feature vector for model prediction
    combined_score = (cosine_sim + structural_sim + tfidf_sim) / 3.0
    feature_vector = [combined_score, cosine_sim, structural_sim, tfidf_sim]
    
    # Get model prediction probability
    probability = trained_model.predict_proba([feature_vector])[0]
    if len(probability) > 1:
        model_sim = probability[1]  # Probability of positive match
    else:
        # Handle single-class model
        prediction = trained_model.predict([feature_vector])[0]
        model_sim = float(prediction) * combined_score
else:
    model_sim = 0.0  # No model available
```

**Model Availability Status:**
- ✅ **Currently Available**: Yes, trained model exists at `ontology_matching_results/trained_model.pkl`
- **Model Type**: RandomForest classifier with 100 estimators
- **Training Data**: Generated from existing SOSA-SAREF correspondences
- **Features**: 4-dimensional vector [combined_score, cosine_sim, structural_sim, tfidf_sim]

### Random Forest Machine Learning - Complete Beginner's Guide

#### What is Random Forest?

Imagine you're trying to decide if two RDF payloads match. Instead of making this decision yourself, you ask 100 different "experts" (called decision trees) to vote on whether they think the payloads match or not. Each expert looks at the same 4 pieces of information but makes their decision slightly differently. The final answer is whatever the majority of experts vote for.

**That's exactly how Random Forest works!**

#### The 4 Pieces of Information (Features)

Each "expert" (decision tree) looks at these 4 numbers to make a decision:
1. **Combined Score** (0.0 to 1.0): Overall similarity between payloads
2. **Cosine Similarity** (0.0 to 1.0): How similar the feature vectors are
3. **Structural Similarity** (0.0 to 1.0): How similar the RDF graph structures are
4. **TF-IDF Similarity** (0.0 to 1.0): How similar the text content is

#### How Training Works (Step by Step)

**Step 1: Collect Training Examples**
```python
# The system looks at previous matches and creates training data
training_examples = [
    [0.85, 0.82, 0.88, 0.85] → "Good Match" (score > 0.7)
    [0.45, 0.40, 0.50, 0.45] → "Bad Match"  (score ≤ 0.7)
    [0.92, 0.90, 0.95, 0.91] → "Good Match"
    [0.25, 0.20, 0.30, 0.25] → "Bad Match"
    # ... hundreds more examples
]
```

**Step 2: Create 100 Decision Trees**

Each tree learns to make decisions like this:
```
Tree #1 asks:
├─ Is combined_score > 0.75?
│  ├─ YES: Is cosine_sim > 0.70? → "Good Match"
│  └─ NO: "Bad Match"

Tree #2 asks:
├─ Is structural_sim > 0.65?
│  ├─ YES: Is tfidf_sim > 0.60? → "Good Match"
│  └─ NO: "Bad Match"

... 98 more trees with different questions
```

**Step 3: Training Process**
```python
# This happens when you run: python3 ml_payload_trainer.py
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(training_features, training_labels)
```

**What happens inside:**
- Each of the 100 trees learns from a random subset of training examples
- Each tree asks different questions about the 4 features
- Trees that make better predictions get more "weight" in the final vote
- After training, the model is saved to `ontology_matching_results/trained_model.pkl`

#### How Prediction Works (When Matching Payloads)

**Step 1: Calculate the 4 Features**
```python
# For a new pair of payloads
new_example = [0.78, 0.75, 0.82, 0.76]  # [combined, cosine, structural, tfidf]
```

**Step 2: Ask All 100 Trees**
```python
# Each tree votes
Tree #1: "Good Match" (confidence: 0.85)
Tree #2: "Bad Match"  (confidence: 0.45)
Tree #3: "Good Match" (confidence: 0.92)
# ... 97 more votes

# Final result: 73 trees say "Good Match", 27 say "Bad Match"
# Probability = 73/100 = 0.73 (73% confident it's a good match)
```

**Step 3: Use in Scoring**
```python
if trained_model_available:
    model_confidence = 0.73  # From Random Forest
    final_score = 0.3*cosine + 0.2*structural + 0.2*tfidf + 0.3*model_confidence
else:
    final_score = 0.4*cosine + 0.3*structural + 0.3*tfidf  # No model
```

#### Why Random Forest Works Well

1. **Wisdom of Crowds**: 100 different "opinions" are better than 1
2. **Handles Complex Patterns**: Can learn non-linear relationships
3. **Robust**: If some trees make mistakes, others compensate
4. **No Overfitting**: Random sampling prevents memorizing training data
5. **Feature Importance**: Shows which of the 4 features matter most

#### Auto-Training Data Sources:

**Automatic Heuristic Labeling:**
- Reads all previous matching results from `ml_payload_matches_*.json` files
- High-scoring correspondences (> 0.7) automatically labeled as "Good Match" (1)
- Low-scoring correspondences (≤ 0.7) automatically labeled as "Bad Match" (0)
- Typically processes 100-500 training examples from historical data
- No human input required - fully automated training pipeline

### Training Algorithm

The system uses supervised learning to improve matching accuracy:

#### How to Train/Retrain the Model

**Automatic Training (Auto-Train Mode)**
```bash
python3 ml_payload_trainer.py --auto-train
```

**What happens during auto-training:**
1. **Load Previous Results**: Automatically reads all `ml_payload_matches_*.json` files from previous matching runs
2. **Generate Training Data**: 
   - Extracts [combined_score, cosine_sim, structural_sim, tfidf_sim] for each correspondence
   - Applies heuristic labeling: matches with score > 0.7 = "Good Match" (1), score ≤ 0.7 = "Bad Match" (0)
   - Creates balanced training dataset from historical correspondences
3. **Train RandomForest Model**:
   - Initializes RandomForestClassifier with 100 decision trees (n_estimators=100)
   - Sets max_depth=5 to prevent overfitting
   - Uses random_state=42 for reproducible results
   - Each tree learns from a random subset of training examples
4. **Model Validation**: Calculates precision, recall, and F1-score on training data
5. **Save Model**: Stores trained model to `ontology_matching_results/trained_model.pkl`
6. **Auto-Exit**: Completes training and exits (no user interaction required)

**Example Auto-Training Output:**
```
🎓 SOSA-SAREF Payload Matcher Training System
==================================================
🚀 Auto-training mode activated
🤖 Training model from correspondence data...
✅ Model trained on 247 samples
   Precision: 0.892
   Recall: 0.856
   F1-Score: 0.874
💾 Model saved to ontology_matching_results/trained_model.pkl
✅ Training session complete!
```

#### RandomForest Model Technical Details

**Confirmed Model Configuration:**
- **Algorithm**: `sklearn.ensemble.RandomForestClassifier`
- **Trees**: 100 decision trees (n_estimators=100)
- **Max Depth**: 5 levels deep (max_depth=5, prevents overfitting)
- **Random Seed**: 42 (random_state=42, ensures reproducible results)
- **Input Features**: 4-dimensional vector [combined_score, cosine_sim, structural_sim, tfidf_sim]
- **Output**: Binary classification probability (0.0 to 1.0 confidence of match)

**Auto-Training Data Source:**
- **Source**: Previous matching results from `ml_payload_matches_*.json` files
- **Labeling**: Automatic heuristic labeling (score > 0.7 = positive, ≤ 0.7 = negative)
- **Features**: Extracted from existing similarity calculations
- **No Manual Input**: Fully automated training process

**Model Files:**
- **Trained Model**: `ontology_matching_results/trained_model.pkl` (37KB)
- **Training Data**: `ontology_matching_results/training_data.pkl` (23KB)

**Performance Metrics (Auto-Training):**
- **Precision**: Percentage of predicted matches that were actually correct
- **Recall**: Percentage of actual matches that were found
- **F1-Score**: Harmonic mean of precision and recall (balanced accuracy measure)