
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "Google Cloud region"
  type        = string
  default     = "us-central1"
}

variable "bucket_name" {
  description = "Name of the Google Cloud Storage bucket"
  type        = string
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_storage_bucket" "example" {
  name                        = var.bucket_name
  location                    = "US"
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  versioning {
    enabled = false
  }

  lifecycle_rule {
    condition {
      age = 30
    }

    action {
      type = "Delete"
    }
  }

  labels = {
    project     = "automlops"
    environment = "development"
  }
}

output "bucket_name" {
  description = "Name of the created bucket"
  value       = google_storage_bucket.example.name
}

output "bucket_url" {
  description = "Google Cloud Storage bucket URL"
  value       = google_storage_bucket.example.url
}
