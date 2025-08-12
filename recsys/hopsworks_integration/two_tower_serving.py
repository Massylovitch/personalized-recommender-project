import tensorflow as tf


class QueryModelModule(tf.Module):
    def __init__(self, model) -> None:
        self.model = model

    @tf.function()
    def compute_embedding(self, instances):
        query_embedding = self.model(instances)

        return {
            "customer_id": instances["customer_id"],
            "month_sin": instances["month_sin"],
            "month_cos": instances["month_cos"],
            "query_emb": query_embedding,
        }


class HopsworksQueryModel:
    deployment_name = "query"

    def __init__(self, model):
        self.model = model

    def save_to_local(self, output_path: str = "query_model") -> str:
        # Define the input specifications for the instances
        instances_spec = {
            "customer_id": tf.TensorSpec(
                shape=(None,), dtype=tf.string, name="customer_id"
            ),  # Specification for customer IDs
            "month_sin": tf.TensorSpec(
                shape=(None,), dtype=tf.float64, name="month_sin"
            ),  # Specification for sine of month
            "month_cos": tf.TensorSpec(
                shape=(None,), dtype=tf.float64, name="month_cos"
            ),  # Specification for cosine of month
            "age": tf.TensorSpec(
                shape=(None,), dtype=tf.float64, name="age"
            ),  # Specification for age
        }

        query_module_module = QueryModelModule(model=self.model)
        # Get the concrete function for the query_model's compute_emb function using the specified input signatures
        inference_signatures = (
            query_module_module.compute_embedding.get_concrete_function(instances_spec)
        )

        # Save the query_model along with the concrete function signatures
        tf.saved_model.save(
            self.model,  # The model to save
            output_path,  # Path to save the model
            signatures=inference_signatures,  # Concrete function signatures to include
        )

        return output_path

    def register(self, mr, feature_view, query_df) -> None:
        local_model_path = self.save_to_local()

        # Sample a query example from the query DataFrame
        query_example = query_df.sample().to_dict("records")

        # Create a tensorflow model for the query_model in the Model Registry
        mr_query_model = mr.tensorflow.create_model(
            name="query_model",  # Name of the model
            description="Model that generates query embeddings from user and transaction features",  # Description of the model
            input_example=query_example,  # Example input for the model
            feature_view=feature_view,
        )

        # Save the query_model to the Model Registry
        mr_query_model.save(local_model_path)  # Path to save the model


class HopsworksCandidateModel:
    def __init__(self, model):
        self.model = model

    def save_to_local(self, output_path="candidate_model"):
        tf.saved_model.save(
            self.model,  # The model to save
            output_path,  # Path to save the model
        )

        return output_path

    def register(self, mr, feature_view, item_df):
        local_model_path = self.save_to_local()

        # Sample a candidate example from the item DataFrame
        candidate_example = item_df.sample().to_dict("records")

        # Create a tensorflow model for the candidate_model in the Model Registry
        mr_candidate_model = mr.tensorflow.create_model(
            name="candidate_model",  # Name of the model
            description="Model that generates candidate embeddings from item features",  # Description of the model
            input_example=candidate_example,  # Example input for the model
            feature_view=feature_view,
        )

        # Save the candidate_model to the Model Registry
        mr_candidate_model.save(local_model_path)  # Path to save the model
