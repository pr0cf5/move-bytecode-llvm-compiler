resource "google_compute_network" "aptos" {
  name                    = "aptos-${terraform.workspace}"
  auto_create_subnetworks = true
}

# If the google_compute_subnetwork data source resolves immediately after the
# network is created, it doesn't find the subnet and returns null. This results
# in the vault-lb address being created in the default network.
resource "time_sleep" "create-subnetworks" {
  create_duration = "30s"
  depends_on      = [google_compute_network.aptos]
}

data "google_compute_subnetwork" "region" {
  name       = google_compute_network.aptos.name
  depends_on = [time_sleep.create-subnetworks]
}

resource "google_compute_router" "nat" {
  name    = "aptos-${terraform.workspace}-nat"
  network = google_compute_network.aptos.id
}

resource "google_compute_address" "nat" {
  count = var.gke_enable_private_nodes ? 1 : 0
  name  = "aptos-${terraform.workspace}-nat"
}

resource "google_compute_router_nat" "nat" {
  count                              = var.gke_enable_private_nodes ? 1 : 0
  name                               = "aptos-${terraform.workspace}-nat"
  router                             = google_compute_router.nat.name
  nat_ip_allocate_option             = "MANUAL_ONLY"
  nat_ips                            = [google_compute_address.nat[0].self_link]
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_PRIMARY_IP_RANGES"
}
