Lecture 1: Distributed Training and Infrastructure as Code (IaC)

As machine learning models and datasets grow, training on a single machine becomes impractical. This lecture covers two advanced MLOps concepts for managing this scale: Distributed Training to accelerate model training, and Infrastructure as Code (IaC) to manage the underlying hardware.

---

 1. Distributed Training

Distributed training is the practice of using multiple processors (like GPUs or TPUs), often across multiple machines, to speed up the training process. There are two primary strategies.

 Data Parallelism
This is the most common strategy. It's used when the model can fit on a single GPU, but the dataset is too large to train in a reasonable amount of time.

- How it works:
    1.  Replicate: The same model is copied onto every worker (GPU).
    2.  Shard Data: The training data is split into smaller, independent partitions.
    3.  Parallel Processing: Each worker processes its own partition of data in parallel, calculating the gradients for its batch.
    4.  Synchronize: The gradients from all workers are aggregated (e.g., averaged).
    5.  Update: The aggregated gradient is used to update the model parameters on all workers, ensuring they stay in sync.
- Best For: Accelerating training on very large datasets.
- Frameworks: `tf.distribute.Strategy` in TensorFlow and `DistributedDataParallel` (DDP) in PyTorch provide easy-to-use APIs for data parallelism.

 Model Parallelism
This strategy is used when the model itself is too large to fit into the memory of a single GPU (e.g., large language models with billions of parameters).

- How it works:
    1.  Partition Model: The model's layers are split across multiple workers. Each worker holds only a piece of the model.
    2.  Process Data: All workers process the same batch of data, but each worker only performs the computations for its assigned layers.
    3.  Pipeline: The output (activations) from one worker is passed as the input to the next worker in the sequence.
- Best For: Training extremely large models that cannot fit on a single device.
- Complexity: Model parallelism is significantly more complex to implement than data parallelism.

| Feature | Data Parallelism | Model Parallelism |
| :--- | :--- | :--- |
| What is split? | The data. | The model. |
| Model Storage | Each worker has a full copy. | Each worker has a partial copy. |
| Primary Use Case | Large dataset, model fits on one GPU. | Model is too large for one GPU. |

---

 2. Infrastructure as Code (IaC)

As your ML projects grow, manually managing cloud resources (like VMs for training, S3 buckets for data, and API endpoints for deployment) becomes tedious and error-prone.

Infrastructure as Code (IaC) is the practice of managing and provisioning infrastructure through code and configuration files, rather than through manual processes. This brings the benefits of software engineering (like version control and automation) to infrastructure management.

 Terraform: The Leading IaC Tool
Terraform is an open-source IaC tool by HashiCorp that allows you to define your cloud infrastructure in a declarative language called HCL (HashiCorp Configuration Language).

- Declarative: You describe the desired state of your infrastructure (e.g., "I want one S3 bucket and one EC2 instance"), and Terraform figures out how to create, update, or delete resources to reach that state.
- Cloud-Agnostic: Terraform works with all major cloud providers (AWS, Google Cloud, Azure) and many other services.

 Core Terraform Concepts
- Providers: Plugins that allow Terraform to interact with a specific API (e.g., the AWS provider, the GCP provider).
- Resources: The infrastructure components you want to create (e.g., `aws_s3_bucket`, `google_compute_instance`).
- Variables: Allow you to parameterize your configurations to make them reusable (e.g., defining the instance type as a variable).
- Outputs: Allow you to extract useful information from your infrastructure after it's created (e.g., the public IP address of a VM).

 Example: Defining an S3 Bucket with Terraform

This simple Terraform configuration file (`main.tf`) defines an S3 bucket in AWS.

```terraform
 1. Configure the AWS provider
provider "aws" {
  region = "us-east-1"
}

 2. Define a resource: an S3 bucket
resource "aws_s3_bucket" "ml_data_bucket" {
  bucket = "my-unique-ml-data-bucket-12345"  Bucket names must be globally unique

  tags = {
    Name    = "ML Data Storage"
    Project = "AI Curriculum"
  }
}

 3. Define an output
output "bucket_name" {
  value = aws_s3_bucket.ml_data_bucket.bucket
}
```

 The Terraform Workflow
1.  `terraform init`: Initializes the directory, downloading the necessary provider plugins. Run this once per project.
2.  `terraform plan`: Shows you what changes Terraform will make to your infrastructure. This is a dry run and is crucial for preventing mistakes.
3.  `terraform apply`: Applies the changes to create, update, or delete resources.
4.  `terraform destroy`: Destroys all the resources managed by the configuration.

By using IaC tools like Terraform, ML teams can ensure their infrastructure is reproducible, version-controlled, and can be automated as part of a larger MLOps pipeline.