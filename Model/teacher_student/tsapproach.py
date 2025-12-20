import tensorflow as tf
from tensorflow import keras
import numpy as np

class TeacherStudentApproach:
    def __init__(self, model, learning_rate=0.001, temperature=4.0):
        self.model = model
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.teacher = None
        self.students = {}  # map domain_id -> student model

    def split_by_domain(self, X, y, domain_id):
        """
        Return ((X_domain, y_domain_labels), (X_rest, y_rest_labels))
        y is expected as [one-hot-culture(3), label(1)] per sample.
        """
        # parse culture and labels
        X = np.asarray(X)
        y = np.asarray(y)
        if y.ndim != 2 or y.shape[1] < 4:
            raise ValueError("y must be shape (n, >=4): [one-hot-culture(3), label(1)]")
        culture_one_hot = y[:, :3]
        labels = y[:, 3].astype(np.int32)

        domain_indices = np.argmax(culture_one_hot, axis=1) == domain_id
        other_indices = ~domain_indices

        X_domain = X[domain_indices]
        y_domain = labels[domain_indices]

        X_rest = X[other_indices]
        y_rest = labels[other_indices]

        return (X_domain, y_domain), (X_rest, y_rest)

    def train_teacher(self, X, y, source_domain_id=0, epochs=10, batch_size=32):
        """Train teacher model on source (majority) domain only"""
        source_data, _ = self.split_by_domain(X, y, source_domain_id)
        X_source, y_source = source_data

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        dataset_size = len(X_source)
        for epoch in range(epochs):
            for i in range(0, dataset_size, batch_size):
                X_batch = X_source[i:i+batch_size]
                y_batch = y_source[i:i+batch_size]

                with tf.GradientTape() as tape:
                    outputs = self.model(X_batch, training=True)
                    loss = loss_fn(y_batch, outputs)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # clone teacher
        self.teacher = keras.models.clone_model(self.model)
        self.teacher.set_weights(self.model.get_weights())

    def train_students(self, X, y, minority_ids, epochs=10, batch_size=32, alpha=0.5, temperature=None):
        """
        Train one student per minority domain id.
        alpha: weight for hard-label loss. (1-alpha) for KD loss.
        temperature: if None use self.temperature.
        """
        if self.teacher is None:
            raise RuntimeError("Teacher must be trained before training students.")

        T = temperature if temperature is not None else self.temperature
        hard_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        kld_fn = keras.losses.KLDivergence()
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        X = np.asarray(X)
        y = np.asarray(y)

        for domain_id in minority_ids:
            # prepare student
            student = keras.models.clone_model(self.model)
            # initialize student from teacher weights (optional, helps convergence)
            try:
                student.set_weights(self.teacher.get_weights())
            except Exception:
                # if shapes mismatch, leave student with default init
                pass

            # get domain-specific data
            (domain_X, domain_y), _ = self.split_by_domain(X, y, domain_id)
            if len(domain_X) == 0:
                # no samples for this minority domain; skip
                continue

            ds_size = len(domain_X)
            for epoch in range(epochs):
                for i in range(0, ds_size, batch_size):
                    X_batch = domain_X[i:i+batch_size]
                    y_batch = domain_y[i:i+batch_size]

                    with tf.GradientTape() as tape:
                        student_logits = student(X_batch, training=True)
                        teacher_logits = self.teacher(X_batch, training=False)

                        # hard label loss
                        hard_loss = hard_loss_fn(y_batch, student_logits)

                        # soft targets loss (KD): KL(soft_teacher || soft_student)
                        soft_teacher = tf.nn.softmax(teacher_logits / T, axis=-1)
                        soft_student = tf.nn.softmax(student_logits / T, axis=-1)
                        kd_loss = kld_fn(soft_teacher, soft_student)

                        # scale KD loss by T^2 as in standard KD formulation
                        loss = alpha * hard_loss + (1.0 - alpha) * (T * T) * kd_loss

                    grads = tape.gradient(loss, student.trainable_variables)
                    optimizer.apply_gradients(zip(grads, student.trainable_variables))

            # store trained student
            self.students[domain_id] = student

    def predict_student(self, domain_id, X):
        """Return class predictions (integers) for a specific student"""
        if domain_id not in self.students:
            raise KeyError(f"No student trained for domain {domain_id}")
        logits = self.students[domain_id](X, training=False)
        return tf.argmax(logits, axis=-1).numpy()

    def evaluate_student(self, domain_id, X, y):
        """Return accuracy for a specific student. y provided in same [one-hot(3), label] format."""
        (domain_X, domain_y), _ = self.split_by_domain(X, y, domain_id)
        if domain_id not in self.students:
            raise KeyError(f"No student trained for domain {domain_id}")
        preds = self.predict_student(domain_id, domain_X)
        acc = np.mean(preds == domain_y)
        return float(acc)

    # New helper: train a single student on provided domain data with given KD params
    def _train_single_student_with_params(self, X_train, y_train, alpha, T, epochs=5, batch_size=32):
        """
        Train one student model on (X_train, y_train) with KD from self.teacher using alpha and temperature T.
        Returns the trained student model.
        """
        if self.teacher is None:
            raise RuntimeError("Teacher must be trained before training students.")

        student = keras.models.clone_model(self.model)
        try:
            student.set_weights(self.teacher.get_weights())
        except Exception:
            # leave default init if shapes mismatch
            pass

        hard_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        kld_fn = keras.losses.KLDivergence()
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        ds_size = len(X_train)

        for epoch in range(epochs):
            for i in range(0, ds_size, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                with tf.GradientTape() as tape:
                    student_logits = student(X_batch, training=True)
                    teacher_logits = self.teacher(X_batch, training=False)

                    hard_loss = hard_loss_fn(y_batch, student_logits)

                    soft_teacher = tf.nn.softmax(teacher_logits / T, axis=-1)
                    soft_student = tf.nn.softmax(student_logits / T, axis=-1)
                    kd_loss = kld_fn(soft_teacher, soft_student)

                    loss = alpha * hard_loss + (1.0 - alpha) * (T * T) * kd_loss

                grads = tape.gradient(loss, student.trainable_variables)
                optimizer.apply_gradients(zip(grads, student.trainable_variables))

        return student

    def fit(self, X, y,
            source_domain_id=0,
            minority_ids=None,
            val_split=0.2,
            model_selection_grid=None,
            teacher_epochs=10,
            student_epochs=10,
            batch_size=32):
        """
        Model selection wrapper:
        - trains teacher on source_domain_id,
        - for each minority domain performs simple grid search over (alpha, temperature) using a train/val split,
        - selects best hyperparameters per domain, retrains best student on full domain, stores in self.students.
        Returns a dict summarizing selected params and validation accuracies.
        """
        # infer minority ids if not provided
        y = np.asarray(y)
        if y.ndim != 2 or y.shape[1] < 4:
            raise ValueError("y must be shape (n, >=4): [one-hot-culture(3), label(1)]")
        culture_one_hot = y[:, :3]
        all_domain_ids = list(np.unique(np.argmax(culture_one_hot, axis=1)))
        if minority_ids is None:
            minority_ids = [d for d in all_domain_ids if d != source_domain_id]

        # default grid
        if model_selection_grid is None:
            model_selection_grid = {
                "alpha": [0.3, 0.5, 0.7],
                "temperature": [2.0, 4.0]
            }

        # 1) Train teacher on majority/source domain
        self.train_teacher(X, y, source_domain_id=source_domain_id, epochs=teacher_epochs, batch_size=batch_size)

        selection_summary = {}

        # 2) For each minority domain perform model selection
        for domain_id in minority_ids:
            (domain_X, domain_y), _ = self.split_by_domain(X, y, domain_id)
            if len(domain_X) == 0:
                selection_summary[domain_id] = {"skipped": True, "reason": "no samples"}
                continue

            # prepare train/val split
            n = len(domain_X)
            idx = np.arange(n)
            np.random.shuffle(idx)
            val_n = max(1, int(val_split * n))
            val_idx = idx[:val_n]
            train_idx = idx[val_n:]

            X_train, y_train = domain_X[train_idx], domain_y[train_idx]
            X_val, y_val = domain_X[val_idx], domain_y[val_idx]

            best_acc = -1.0
            best_cfg = None
            best_student = None

            for alpha in model_selection_grid.get("alpha", [0.5]):
                for T in model_selection_grid.get("temperature", [self.temperature]):
                    # train with small number of epochs for selection
                    candidate = self._train_single_student_with_params(
                        X_train, y_train, alpha=alpha, T=T, epochs=student_epochs, batch_size=batch_size
                    )
                    # validate
                    logits = candidate(X_val, training=False)
                    preds = np.argmax(logits.numpy(), axis=-1)
                    acc = float(np.mean(preds == y_val))

                    if acc > best_acc:
                        best_acc = acc
                        best_cfg = {"alpha": alpha, "temperature": T, "val_acc": acc}
                        best_student = candidate

            if best_student is None:
                selection_summary[domain_id] = {"skipped": True, "reason": "no candidate trained"}
                continue

            # Optional: retrain best student on full domain data for stability
            final_student = self._train_single_student_with_params(
                domain_X, domain_y,
                alpha=best_cfg["alpha"],
                T=best_cfg["temperature"],
                epochs=student_epochs,
                batch_size=batch_size
            )

            self.students[domain_id] = final_student
            selection_summary[domain_id] = best_cfg

        return selection_summary
