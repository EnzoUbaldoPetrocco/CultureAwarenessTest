import tensorflow_model_optimization as tfmot
import tensorflow as tf

class Pruning:
    def __init__(self, model, optimizer, loss, metrics, initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=10000):
        self.model = model
        self.pruned_model = self.structured_pruning(optimizer, loss, metrics)
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.begin_step = begin_step
        self.end_step = end_step


    def structured_pruning(self, optimizer, loss, metrics):     
        # Define pruning schedule (from 0% → 50% sparsity over training)
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=self.initial_sparsity,
            final_sparsity=self.final_sparsity,   # prune 50% weights
            begin_step=self.begin_step,
            end_step=self.end_step
        )

        # Apply pruning wrapper
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            self.model,
            pruning_schedule=pruning_schedule
        )

        # Compile for retraining
        pruned_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return pruned_model

    def fine_tune_pruned_model(self, x_train, y_train, x_val, y_val, batch_size=32, epochs=2):
        # Fine-tune the pruned model
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir='/tmp/pruning_logs')
        ]

        self.pruned_model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks
        )

    def strip_pruning(self):
        # Strip pruning wrappers to obtain the final pruned model
        final_model = tfmot.sparsity.keras.strip_pruning(self.pruned_model)
        return final_model
