from model.config import ModelConfig, TrainingConfig

MODEL = ModelConfig()
TRAINING = TrainingConfig(stride=MODEL.context_length)
