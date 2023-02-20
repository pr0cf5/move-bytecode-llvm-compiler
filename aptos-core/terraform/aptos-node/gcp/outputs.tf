output "helm_release_name" {
  value = helm_release.validator.name
}

output "gke_cluster_endpoint" {
  value = google_container_cluster.aptos.endpoint
}

output "gke_cluster_ca_certificate" {
  value = google_container_cluster.aptos.master_auth[0].cluster_ca_certificate
}
